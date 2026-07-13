#!/usr/bin/env python3
"""Extract button centers from a layout image and print coordinates relative to Start.

Method overview:
1) Detect candidate circles with Hough transform.
2) Validate/refine each circle by sampling many angles around the circumference.
3) Keep only circles with strong full-circumference edge support.
4) OCR text inside each circle (with arrow-shape fallback).
5) Print a table of button coordinates relative to the Start button.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
	import cv2
except ModuleNotFoundError:
	cv2 = None

try:
	import numpy as np
except ModuleNotFoundError:
	np = None

try:
	import pytesseract
except ModuleNotFoundError:
	pytesseract = None


@dataclass
class Circle:
	cx: float
	cy: float
	r: float
	support_ratio: float
	rms_residual: float


@dataclass
class Button:
	label: str
	circle: Circle


EXPECTED_LABELS = [
	"Start",
	"L",
	"R",
	"A",
	"B",
	"X",
	"Y",
	"Z",
	"LS",
	"MS",
	"MX",
	"MY",
	"Dpad Toggle",
	"c Up",
	"c Down",
	"c Left",
	"c Right",
	"UP",
	"DOWN",
	"LEFT",
	"RIGHT",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Detect complete circular buttons and output coordinates relative to Start."
	)
	parser.add_argument(
		"image",
		nargs="?",
		default="resources/lbx-button-layout.png",
		help="Input image path (default: resources/lbx-button-layout.png)",
	)
	parser.add_argument(
		"--samples",
		type=int,
		default=128,
		help="Number of angular samples around each circle (default: 128).",
	)
	parser.add_argument(
		"--upsample",
		type=float,
		default=4.0,
		help="Image upsample scale before detection (default: 4.0).",
	)
	parser.add_argument(
		"--radial-search",
		type=int,
		default=10,
		help="Radial pixel search range around estimated radius (default: 10).",
	)
	parser.add_argument(
		"--min-support",
		type=float,
		default=0.62,
		help="Minimum ratio of successful edge samples for a complete circle (default: 0.62).",
	)
	parser.add_argument(
		"--min-dist",
		type=float,
		default=14.0,
		help="Minimum center distance for Hough circle candidates (pixels).",
	)
	parser.add_argument(
		"--min-radius",
		type=float,
		default=10.5,
		help="Minimum button radius in input-image pixels (default: 10.5).",
	)
	parser.add_argument(
		"--max-radius",
		type=float,
		default=20.5,
		help="Maximum button radius in input-image pixels (default: 20.5).",
	)
	parser.add_argument(
		"--fixed-radius",
		type=float,
		default=18.0,
		help="Keep radius fixed to this input-image pixel value during refinement (default: 18.0).",
	)
	parser.add_argument(
		"--bf-search-px",
		type=int,
		default=8,
		help="Brute-force integer search width in pixels (default: 8 for 8x8 map).",
	)
	parser.add_argument(
		"--subpixel-steps",
		type=int,
		default=100,
		help="Subpixel hypotheses per axis for brute-force refinement (default: 100).",
	)
	parser.add_argument(
		"--max-rms",
		type=float,
		default=1.2,
		help="Maximum RMS radial residual in input-image pixels (default: 1.2).",
	)
	parser.add_argument(
		"--coverage-refine",
		action="store_true",
		help="Refine circles by maximizing dark-ring coverage in grayscale image.",
	)
	parser.add_argument(
		"--ring-half-width",
		type=float,
		default=1.8,
		help="Half width of annulus used for dark-coverage scoring in input px (default: 1.8).",
	)
	parser.add_argument(
		"--coverage-iters",
		type=int,
		default=6,
		help="Iterations for black-coverage local optimization (default: 6).",
	)
	parser.add_argument(
		"--coverage-restarts",
		type=int,
		default=8,
		help="Number of randomized optimizer restarts for coverage refinement (default: 8).",
	)
	parser.add_argument(
		"--coverage-jitter",
		type=float,
		default=1.8,
		help="Random start jitter in upsampled pixels for coverage refinement (default: 1.8).",
	)
	parser.add_argument(
		"--coverage-angles",
		type=int,
		default=1024,
		help="Angular samples used by coverage score (default: 1024).",
	)
	parser.add_argument(
		"--mm-per-px",
		type=float,
		default=None,
		help="Millimeters per pixel for metric output. If omitted, mm columns are blank.",
	)
	parser.add_argument(
		"--csv",
		type=str,
		default=None,
		help="Optional output CSV path.",
	)
	parser.add_argument(
		"--overlay-image",
		type=str,
		default=None,
		help=(
			"Path for the circles-overlay output image. "
			"Default: <input_stem>.overlay.png in the same folder as input image."
		),
	)
	return parser.parse_args()


def require_dependencies() -> None:
	missing = []
	if cv2 is None:
		missing.append("opencv-python")
	if np is None:
		missing.append("numpy")
	if missing:
		print(
			"Missing required Python packages: "
			+ ", ".join(missing)
			+ "\nInstall with: pip install "
			+ " ".join(missing),
			file=sys.stderr,
		)
		raise SystemExit(2)


def fit_circle_least_squares(points: Any) -> tuple[float, float, float] | None:
	"""Kasa circle fit from points Nx2. Returns (cx, cy, r)."""
	if points.shape[0] < 6:
		return None
	x = points[:, 0]
	y = points[:, 1]
	a = np.column_stack((x, y, np.ones_like(x)))
	b = -(x * x + y * y)
	try:
		sol, *_ = np.linalg.lstsq(a, b, rcond=None)
	except np.linalg.LinAlgError:
		return None
	d, e, f = sol
	cx = -d / 2.0
	cy = -e / 2.0
	rad_sq = cx * cx + cy * cy - f
	if rad_sq <= 0:
		return None
	return float(cx), float(cy), float(math.sqrt(rad_sq))


def circle_residuals(points: Any, cx: float, cy: float, r: float) -> Any:
	d = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
	return np.abs(d - r)


def robust_refine_circle(points: Any) -> tuple[float, float, float, float] | None:
	if points.shape[0] < 12:
		return None
	fit = fit_circle_least_squares(points)
	if fit is None:
		return None
	cx, cy, r = fit

	for _ in range(3):
		res = circle_residuals(points, cx, cy, r)
		if res.size < 8:
			break
		med = float(np.median(res))
		mad = float(np.median(np.abs(res - med))) + 1e-6
		thr = max(0.8, med + 2.8 * mad)
		inliers = points[res <= thr]
		if inliers.shape[0] < 8:
			break
		fit2 = fit_circle_least_squares(inliers)
		if fit2 is None:
			break
		cx, cy, r = fit2
		points = inliers

	res_final = circle_residuals(points, cx, cy, r)
	rms = float(np.sqrt(np.mean(res_final * res_final))) if res_final.size else 99.0
	return float(cx), float(cy), float(r), rms


def bilinear_sample(gray: Any, xs: Any, ys: Any) -> Any:
	"""Bilinear grayscale sampling for vectorized coordinates."""
	h, w = gray.shape[:2]
	xs = np.clip(xs, 0.0, w - 1.001)
	ys = np.clip(ys, 0.0, h - 1.001)
	x0 = np.floor(xs).astype(np.int32)
	y0 = np.floor(ys).astype(np.int32)
	x1 = np.clip(x0 + 1, 0, w - 1)
	y1 = np.clip(y0 + 1, 0, h - 1)
	fx = xs - x0
	fy = ys - y0

	v00 = gray[y0, x0].astype(np.float32)
	v10 = gray[y0, x1].astype(np.float32)
	v01 = gray[y1, x0].astype(np.float32)
	v11 = gray[y1, x1].astype(np.float32)
	v0 = v00 * (1.0 - fx) + v10 * fx
	v1 = v01 * (1.0 - fx) + v11 * fx
	return v0 * (1.0 - fy) + v1 * fy


def build_ring_kernel(radius: float, sub_x: float, sub_y: float, size: int = 37) -> Any:
	"""Create a matched-filter kernel for a dark ring at fixed radius with subpixel center offset."""
	if size % 2 == 0:
		raise ValueError("Kernel size must be odd.")
	c = (size - 1) / 2.0
	ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
	d = np.hypot(xs - (c + sub_x), ys - (c + sub_y))

	# Positive ring reward, negative nearby penalties (maximize score).
	ring = (np.abs(d - radius) <= 0.95).astype(np.float32)
	inner = (np.abs(d - (radius - 2.0)) <= 0.85).astype(np.float32)
	outer = (np.abs(d - (radius + 2.0)) <= 0.85).astype(np.float32)

	k = 1.0 * ring - 0.42 * inner - 0.36 * outer
	k -= float(np.mean(k))
	norm = float(np.linalg.norm(k)) + 1e-9
	return k / norm


def brute_force_refine_fixed_radius_center(
	gray: Any,
	cx: float,
	cy: float,
	radius: float,
	search_px: int,
	subpixel_steps: int,
) -> tuple[float, float, float] | None:
	"""Brute-force center search using 37x37 template over 45x45 ROI and 10x10 subpixels."""
	if search_px != 8:
		# Keep behavior explicit for the requested algorithm shape.
		search_px = max(2, search_px)

	patch_size = 37
	roi_size = patch_size + search_px
	h, w = gray.shape[:2]
	half_roi = roi_size // 2

	cx_i = int(round(cx))
	cy_i = int(round(cy))
	x0 = cx_i - half_roi
	y0 = cy_i - half_roi
	x1 = x0 + roi_size
	y1 = y0 + roi_size
	if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
		return None

	# Score in darkness space so high values correspond to black ink.
	roi_gray = gray[y0:y1, x0:x1].astype(np.float32)
	roi = 255.0 - roi_gray
	best_score = float("-inf")
	best_cx = cx
	best_cy = cy

	if subpixel_steps < 2:
		subpixel_steps = 2

	# subpixel_steps x subpixel_steps hypotheses.
	for sy_i in range(subpixel_steps):
		sub_y = sy_i / float(subpixel_steps)
		for sx_i in range(subpixel_steps):
			sub_x = sx_i / float(subpixel_steps)
			k = build_ring_kernel(radius=radius, sub_x=sub_x, sub_y=sub_y, size=patch_size)

			# 8x8 integer search map from valid 37x37 placements in 45x45 ROI.
			best_local = float("-inf")
			best_ox = 0
			best_oy = 0
			for oy in range(search_px):
				for ox in range(search_px):
					patch = roi[oy : oy + patch_size, ox : ox + patch_size]
					score = float(np.sum(patch * k))
					if score > best_local:
						best_local = score
						best_ox = ox
						best_oy = oy

			# Global max over the 100 subpixel maxima.
			if best_local > best_score:
				best_score = best_local
				best_cx = x0 + best_ox + 18.0 + sub_x
				best_cy = y0 + best_oy + 18.0 + sub_y

	return float(best_cx), float(best_cy), float(best_score)


def dark_ring_coverage_score(
	dark_map: Any,
	grad_norm: Any,
	cx: float,
	cy: float,
	r: float,
	ring_half_width: float,
	n_angles: int,
) -> float:
	"""Score dark ring coverage and edge-boundary agreement for a candidate circle."""
	if r <= 2.0:
		return -1e9
	angles = np.linspace(0.0, 2.0 * np.pi, num=n_angles, endpoint=False, dtype=np.float64)
	ux = np.cos(angles)
	uy = np.sin(angles)

	def annulus_dark(map_img: Any, r_mid: float, half_w: float, radial_samples: int) -> float:
		offsets = np.linspace(-half_w, half_w, num=radial_samples, dtype=np.float64)
		vals = []
		for dr in offsets:
			rr = r_mid + dr
			xs = cx + rr * ux
			ys = cy + rr * uy
			g = bilinear_sample(map_img, xs, ys)
			vals.append(g)
		v = np.stack(vals, axis=0)
		return float(np.mean(v))

	main_dark = annulus_dark(dark_map, r, ring_half_width, radial_samples=7)
	inner_dark = annulus_dark(dark_map, max(0.5, r - 2.6 * ring_half_width), ring_half_width * 0.8, radial_samples=5)
	outer_dark = annulus_dark(dark_map, r + 2.6 * ring_half_width, ring_half_width * 0.8, radial_samples=5)

	edge_in = annulus_dark(grad_norm, max(0.5, r - ring_half_width), ring_half_width * 0.35, radial_samples=3)
	edge_out = annulus_dark(grad_norm, r + ring_half_width, ring_half_width * 0.35, radial_samples=3)

	# Encourage dark ring occupancy and strong edge response at both borders.
	return 1.8 * main_dark - 0.7 * inner_dark - 0.55 * outer_dark + 0.35 * (edge_in + edge_out)


def maximize_black_ring_coverage(
	dark_map: Any,
	grad_norm: Any,
	cx: float,
	cy: float,
	r: float,
	ring_half_width: float,
	iters: int,
	restarts: int,
	jitter: float,
	n_angles: int,
) -> tuple[float, float, float]:
	"""Multi-start local search in (cx, cy, r) to maximize dark ring coverage."""
	rng = np.random.default_rng(7)
	global_best = (cx, cy, r)
	global_score = dark_ring_coverage_score(dark_map, grad_norm, cx, cy, r, ring_half_width, n_angles)

	for restart_id in range(max(1, restarts)):
		if restart_id == 0:
			cur = (cx, cy, r)
		else:
			cur = (
				cx + float(rng.normal(0.0, jitter)),
				cy + float(rng.normal(0.0, jitter)),
				max(2.0, r + float(rng.normal(0.0, jitter * 0.6))),
			)

		step_xy = max(0.3, 1.6)
		step_r = max(0.2, 1.0)
		cur_score = dark_ring_coverage_score(
			dark_map,
			grad_norm,
			cur[0],
			cur[1],
			cur[2],
			ring_half_width,
			n_angles,
		)

		for _ in range(max(1, iters)):
			improved = False
			for dx in (0.0, -step_xy, step_xy, -0.5 * step_xy, 0.5 * step_xy):
				for dy in (0.0, -step_xy, step_xy, -0.5 * step_xy, 0.5 * step_xy):
					for dr in (0.0, -step_r, step_r, -0.5 * step_r, 0.5 * step_r):
						if dx == 0.0 and dy == 0.0 and dr == 0.0:
							continue
						cand = (cur[0] + dx, cur[1] + dy, max(2.0, cur[2] + dr))
						s = dark_ring_coverage_score(
							dark_map,
							grad_norm,
							cand[0],
							cand[1],
							cand[2],
							ring_half_width,
							n_angles,
						)
						if s > cur_score:
							cur = cand
							cur_score = s
							improved = True
			if not improved:
				step_xy *= 0.55
				step_r *= 0.55
			if step_xy < 0.06 and step_r < 0.05:
				break

		if cur_score > global_score:
			global_score = cur_score
			global_best = cur

	return float(global_best[0]), float(global_best[1]), float(global_best[2])


def maximize_black_ring_center_fixed_radius(
	dark_map: Any,
	grad_norm: Any,
	cx: float,
	cy: float,
	r_fixed: float,
	ring_half_width: float,
	iters: int,
	restarts: int,
	jitter: float,
	n_angles: int,
) -> tuple[float, float, float]:
	"""Multi-start local search for center only with fixed radius."""
	rng = np.random.default_rng(17)
	global_best = (cx, cy, r_fixed)
	global_score = dark_ring_coverage_score(dark_map, grad_norm, cx, cy, r_fixed, ring_half_width, n_angles)

	for restart_id in range(max(1, restarts)):
		if restart_id == 0:
			curx, cury = cx, cy
		else:
			curx = cx + float(rng.normal(0.0, jitter))
			cury = cy + float(rng.normal(0.0, jitter))

		step_xy = 1.6
		cur_score = dark_ring_coverage_score(
			dark_map,
			grad_norm,
			curx,
			cury,
			r_fixed,
			ring_half_width,
			n_angles,
		)

		for _ in range(max(1, iters)):
			improved = False
			for dx in (0.0, -step_xy, step_xy, -0.5 * step_xy, 0.5 * step_xy):
				for dy in (0.0, -step_xy, step_xy, -0.5 * step_xy, 0.5 * step_xy):
					if dx == 0.0 and dy == 0.0:
						continue
					candx = curx + dx
					candy = cury + dy
					s = dark_ring_coverage_score(
						dark_map,
						grad_norm,
						candx,
						candy,
						r_fixed,
						ring_half_width,
						n_angles,
					)
					if s > cur_score:
						curx, cury = candx, candy
						cur_score = s
						improved = True
			if not improved:
				step_xy *= 0.55
			if step_xy < 0.05:
				break

		if cur_score > global_score:
			global_score = cur_score
			global_best = (curx, cury, r_fixed)

	return float(global_best[0]), float(global_best[1]), float(global_best[2])


def sample_circle_edges(
	edges: Any,
	grad_x: Any,
	grad_y: Any,
	grad_mag: Any,
	cx: float,
	cy: float,
	r: float,
	samples: int,
	radial_search: int,
) -> tuple[Any, float]:
	"""Sample circle edge points while favoring gradient directions normal to the circle."""
	h, w = edges.shape[:2]
	found_points: list[tuple[float, float]] = []
	hits = 0

	def best_hit_for_range(
		dr_values: range,
		base_cx: float,
		base_cy: float,
		base_r: float,
		ux: float,
		uy: float,
	) -> tuple[tuple[float, float] | None, float]:
		best_score = -1.0
		best_hit: tuple[float, float] | None = None
		for dr in dr_values:
			rr = base_r + dr
			x = int(round(base_cx + rr * ux))
			y = int(round(base_cy + rr * uy))
			if x < 0 or y < 0 or x >= w or y >= h:
				continue
			if edges[y, x] <= 0:
				continue
			gx = float(grad_x[y, x])
			gy = float(grad_y[y, x])
			gm = float(grad_mag[y, x])
			if gm < 10.0:
				continue
			align = abs((gx * ux + gy * uy) / gm)
			score = gm * max(0.0, align - 0.45)
			if score > best_score:
				best_score = score
				best_hit = (float(x), float(y))
		return best_hit, best_score

	for i in range(samples):
		theta = 2.0 * math.pi * i / samples
		ux = math.cos(theta)
		uy = math.sin(theta)

		# Pair edge hits on both sides of the ring stroke and use their midpoint.
		inner_hit, inner_score = best_hit_for_range(
			range(-radial_search, 0),
			cx,
			cy,
			r,
			ux,
			uy,
		)
		outer_hit, outer_score = best_hit_for_range(
			range(1, radial_search + 1),
			cx,
			cy,
			r,
			ux,
			uy,
		)

		hit = None
		if inner_hit is not None and outer_hit is not None:
			hit = (
				0.5 * (inner_hit[0] + outer_hit[0]),
				0.5 * (inner_hit[1] + outer_hit[1]),
			)
		else:
			fallback_hit, fallback_score = best_hit_for_range(
				range(-radial_search, radial_search + 1),
				cx,
				cy,
				r,
				ux,
				uy,
			)
			if fallback_hit is not None and max(inner_score, outer_score, fallback_score) > 0:
				hit = fallback_hit

		if hit is not None:
			hits += 1
			found_points.append(hit)
	support_ratio = hits / float(samples)
	if not found_points:
		return np.empty((0, 2), dtype=np.float64), support_ratio
	return np.array(found_points, dtype=np.float64), support_ratio


def refine_and_validate_circle(
	edges: Any,
	dark_map: Any,
	grad_norm: Any,
	grad_x: Any,
	grad_y: Any,
	grad_mag: Any,
	circle: Iterable[float],
	samples: int,
	radial_search: int,
	min_support: float,
	max_rms_up: float,
	coverage_refine: bool,
	ring_half_width_up: float,
	coverage_iters: int,
	coverage_restarts: int,
	coverage_jitter: float,
	coverage_angles: int,
	fixed_radius_up: float | None,
) -> Circle | None:
	cx, cy, r = [float(v) for v in circle]
	if fixed_radius_up is not None:
		r = float(fixed_radius_up)
	points, support = sample_circle_edges(edges, grad_x, grad_y, grad_mag, cx, cy, r, samples, radial_search)
	if support < min_support:
		return None

	if fixed_radius_up is None:
		refined = robust_refine_circle(points)
	else:
		# Fit center from sampled points while keeping radius fixed.
		if points.shape[0] < 8:
			return None
		rcx = float(np.mean(points[:, 0]))
		rcy = float(np.mean(points[:, 1]))
		rr = float(fixed_radius_up)
		res = circle_residuals(points, rcx, rcy, rr)
		rms = float(np.sqrt(np.mean(res * res))) if res.size else 99.0
		refined = (rcx, rcy, rr, rms)
	if refined is None:
		return None
	rcx, rcy, rr, rms = refined
	if coverage_refine:
		if fixed_radius_up is None:
			rcx, rcy, rr = maximize_black_ring_coverage(
				dark_map=dark_map,
				grad_norm=grad_norm,
				cx=rcx,
				cy=rcy,
				r=rr,
				ring_half_width=ring_half_width_up,
				iters=coverage_iters,
				restarts=coverage_restarts,
				jitter=coverage_jitter,
				n_angles=coverage_angles,
			)
		else:
			rcx, rcy, rr = maximize_black_ring_center_fixed_radius(
				dark_map=dark_map,
				grad_norm=grad_norm,
				cx=rcx,
				cy=rcy,
				r_fixed=fixed_radius_up,
				ring_half_width=ring_half_width_up,
				iters=coverage_iters,
				restarts=coverage_restarts,
				jitter=coverage_jitter,
				n_angles=coverage_angles,
			)
		points_cov, support_cov = sample_circle_edges(
			edges,
			grad_x,
			grad_y,
			grad_mag,
			rcx,
			rcy,
			rr,
			samples,
			radial_search,
		)
		if support_cov >= min_support:
			if fixed_radius_up is None:
				refined_cov = robust_refine_circle(points_cov)
				if refined_cov is not None:
					rcx, rcy, rr, rms = refined_cov
			else:
				rcx = float(np.mean(points_cov[:, 0]))
				rcy = float(np.mean(points_cov[:, 1]))
				rr = float(fixed_radius_up)
				res_cov = circle_residuals(points_cov, rcx, rcy, rr)
				rms = float(np.sqrt(np.mean(res_cov * res_cov))) if res_cov.size else 99.0
			if coverage_refine:
					if fixed_radius_up is None:
						rcx, rcy, rr = maximize_black_ring_coverage(
							dark_map=dark_map,
							grad_norm=grad_norm,
							cx=rcx,
							cy=rcy,
							r=rr,
							ring_half_width=ring_half_width_up,
							iters=max(2, coverage_iters // 2),
							restarts=max(2, coverage_restarts // 2),
							jitter=max(0.4, coverage_jitter * 0.55),
							n_angles=max(256, coverage_angles // 2),
						)
					else:
						rcx, rcy, rr = maximize_black_ring_center_fixed_radius(
							dark_map=dark_map,
							grad_norm=grad_norm,
							cx=rcx,
							cy=rcy,
							r_fixed=fixed_radius_up,
							ring_half_width=ring_half_width_up,
							iters=max(2, coverage_iters // 2),
							restarts=max(2, coverage_restarts // 2),
							jitter=max(0.4, coverage_jitter * 0.55),
							n_angles=max(256, coverage_angles // 2),
						)

	if fixed_radius_up is not None:
		rr = float(fixed_radius_up)
		points_f, _ = sample_circle_edges(
			edges,
			grad_x,
			grad_y,
			grad_mag,
			rcx,
			rcy,
			rr,
			samples,
			radial_search,
		)
		if points_f.shape[0] >= 8:
			res_f = circle_residuals(points_f, rcx, rcy, rr)
			rms = float(np.sqrt(np.mean(res_f * res_f))) if res_f.size else rms
	if rms > max_rms_up:
		return None

	points2, support2 = sample_circle_edges(edges, grad_x, grad_y, grad_mag, rcx, rcy, rr, samples, radial_search)
	if support2 < min_support:
		return None
	refined2 = robust_refine_circle(points2)
	if refined2 is None:
		return None
	fcx, fcy, fr, rms2 = refined2
	if rms2 > max_rms_up:
		return None
	return Circle(cx=fcx, cy=fcy, r=fr, support_ratio=support2, rms_residual=rms2)


def deduplicate_circles(circles: list[Circle]) -> list[Circle]:
	circles = sorted(circles, key=lambda c: c.support_ratio, reverse=True)
	kept: list[Circle] = []
	for c in circles:
		duplicate = False
		for k in kept:
			dc = math.hypot(c.cx - k.cx, c.cy - k.cy)
			if dc < 0.45 * min(c.r, k.r) and abs(c.r - k.r) < 4.0:
				duplicate = True
				break
		if not duplicate:
			kept.append(c)
	return kept


def normalize_label(label: str) -> str:
	label = " ".join(label.strip().split())
	mapping = {
		"LS": "LS",
		"MS": "MS",
		"MX": "MX",
		"MY": "MY",
		"DPAD TOGGLE": "Dpad Toggle",
		"DPADTOGGLE": "Dpad Toggle",
		"START": "Start",
		"C UP": "c Up",
		"C DOWN": "c Down",
		"C LEFT": "c Left",
		"C RIGHT": "c Right",
	}
	up = label.upper()
	return mapping.get(up, label)


def canonical_key(s: str) -> str:
	return "".join(ch for ch in s.upper() if ch.isalnum())


def fuzzy_map_label(label: str) -> str:
	if not label:
		return ""
	normalized = normalize_label(label)
	key = canonical_key(normalized)
	if not key:
		return ""

	for candidate in EXPECTED_LABELS:
		if canonical_key(candidate) == key:
			return candidate

	scores = []
	for candidate in EXPECTED_LABELS:
		ckey = canonical_key(candidate)
		ratio = difflib.SequenceMatcher(a=key, b=ckey).ratio()
		scores.append((ratio, candidate))
	best_ratio, best = max(scores, key=lambda t: t[0])
	if best_ratio >= 0.55:
		return best
	return normalized


def detect_arrow_label(roi_gray: Any) -> str | None:
	"""Fallback classifier for triangle-only labels: returns UP/DOWN/LEFT/RIGHT."""
	blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
	_, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return None
	largest = max(contours, key=cv2.contourArea)
	area = cv2.contourArea(largest)
	if area < 40:
		return None

	peri = cv2.arcLength(largest, True)
	approx = cv2.approxPolyDP(largest, 0.05 * peri, True)
	if len(approx) < 3 or len(approx) > 5:
		return None

	pts = largest.reshape(-1, 2).astype(np.float32)
	c = pts.mean(axis=0)
	d2 = ((pts - c) ** 2).sum(axis=1)
	tip = pts[int(np.argmax(d2))]
	vx, vy = tip - c
	if abs(vx) > abs(vy):
		return "RIGHT" if vx > 0 else "LEFT"
	return "DOWN" if vy > 0 else "UP"


def ocr_label(gray: Any, circle: Circle) -> str:
	if pytesseract is None:
		return ""

	h, w = gray.shape[:2]
	pad = int(max(3, round(circle.r * 0.8)))
	x0 = max(0, int(round(circle.cx)) - pad)
	y0 = max(0, int(round(circle.cy)) - pad)
	x1 = min(w, int(round(circle.cx)) + pad)
	y1 = min(h, int(round(circle.cy)) + pad)
	roi = gray[y0:y1, x0:x1]
	if roi.size == 0:
		return ""

	roi = cv2.resize(roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
	roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
	_, roi_bw = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
	text = pytesseract.image_to_string(roi_bw, config=cfg)
	text = fuzzy_map_label(text)
	if text:
		return text

	cfg_multi = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
	text2 = pytesseract.image_to_string(roi_bw, config=cfg_multi)
	text2 = fuzzy_map_label(text2)
	if text2:
		return text2

	arrow = detect_arrow_label(roi)
	return arrow or ""


def detect_buttons(
	image_path: Path,
	samples: int,
	radial_search: int,
	min_support: float,
	min_dist: float,
	upsample: float,
	min_radius: float,
	max_radius: float,
	max_rms: float,
	coverage_refine: bool,
	ring_half_width: float,
	coverage_iters: int,
	coverage_restarts: int,
	coverage_jitter: float,
	coverage_angles: int,
	fixed_radius: float | None,
	bf_search_px: int,
	subpixel_steps: int,
) -> tuple[Any, list[Button]]:
	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	if upsample <= 0:
		raise ValueError("upsample must be > 0")
	gray_up = cv2.resize(gray, None, fx=upsample, fy=upsample, interpolation=cv2.INTER_CUBIC)
	blur = cv2.GaussianBlur(gray_up, (5, 5), 1.0)
	edges = cv2.Canny(blur, 45, 125)
	grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
	grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
	grad_mag = cv2.magnitude(grad_x, grad_y)
	grad_scale = float(np.percentile(grad_mag, 99.0)) + 1e-6
	grad_norm = np.clip(grad_mag / grad_scale, 0.0, 1.0)
	dark_u8 = cv2.adaptiveThreshold(
		gray_up,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY_INV,
		41,
		7,
	)
	dark_map = cv2.GaussianBlur(dark_u8.astype(np.float32) / 255.0, (3, 3), 0.8)

	min_radius_up = max(1, int(round(min_radius * upsample)))
	max_radius_up = max(min_radius_up + 1, int(round(max_radius * upsample)))
	min_dist_up = float(min_dist * upsample)
	max_rms_up = float(max_rms * upsample)
	ring_half_width_up = float(max(0.5, ring_half_width * upsample))
	# Keep early candidate validation flexible; fixed radius is enforced later in original space.
	fixed_radius_up = None

	hough = cv2.HoughCircles(
		blur,
		cv2.HOUGH_GRADIENT,
		dp=1.2,
		minDist=min_dist_up,
		param1=120,
		param2=24,
		minRadius=min_radius_up,
		maxRadius=max_radius_up,
	)

	if hough is None:
		return image, []

	raw = np.round(hough[0, :]).astype(np.float64)
	circles: list[Circle] = []
	for cand in raw:
		c = refine_and_validate_circle(
			edges=edges,
			dark_map=dark_map,
			grad_norm=grad_norm,
			grad_x=grad_x,
			grad_y=grad_y,
			grad_mag=grad_mag,
			circle=cand,
			samples=samples,
			radial_search=radial_search,
			min_support=min_support,
			max_rms_up=max_rms_up,
			coverage_refine=coverage_refine,
			ring_half_width_up=ring_half_width_up,
			coverage_iters=coverage_iters,
			coverage_restarts=coverage_restarts,
			coverage_jitter=coverage_jitter,
			coverage_angles=coverage_angles,
			fixed_radius_up=fixed_radius_up,
		)
		if c is not None:
			circles.append(
				Circle(
					cx=c.cx / upsample,
					cy=c.cy / upsample,
					r=c.r / upsample,
					support_ratio=c.support_ratio,
					rms_residual=c.rms_residual / upsample,
				)
			)

	circles = deduplicate_circles(circles)

	if fixed_radius is not None:
		# Enforce fixed-radius refinement in original-pixel space for every detected circle.
		blur0 = cv2.GaussianBlur(gray, (5, 5), 1.0)
		edges0 = cv2.Canny(blur0, 45, 125)
		gx0 = cv2.Sobel(blur0, cv2.CV_32F, 1, 0, ksize=3)
		gy0 = cv2.Sobel(blur0, cv2.CV_32F, 0, 1, ksize=3)
		gm0 = cv2.magnitude(gx0, gy0)

		fixed_list: list[Circle] = []
		for c in circles:
			refined = brute_force_refine_fixed_radius_center(
				gray=gray,
				cx=c.cx,
				cy=c.cy,
				radius=float(fixed_radius),
				search_px=bf_search_px,
				subpixel_steps=subpixel_steps,
			)
			if refined is None:
				continue
			ncx, ncy, _ = refined
			pts, supp = sample_circle_edges(
				edges=edges0,
				grad_x=gx0,
				grad_y=gy0,
				grad_mag=gm0,
				cx=ncx,
				cy=ncy,
				r=float(fixed_radius),
				samples=max(64, samples // 2),
				radial_search=max(3, radial_search // 2),
			)
			if supp < 0.35 or pts.shape[0] < 8:
				continue
			res = circle_residuals(pts, ncx, ncy, float(fixed_radius))
			rms = float(np.sqrt(np.mean(res * res))) if res.size else 99.0
			if rms > max(3.0, 1.8 * max_rms):
				continue
			fixed_list.append(
				Circle(
					cx=ncx,
					cy=ncy,
					r=float(fixed_radius),
					support_ratio=supp,
					rms_residual=rms,
				)
			)
		circles = deduplicate_circles(fixed_list)

	buttons: list[Button] = []
	for i, c in enumerate(circles, start=1):
		label = ocr_label(gray, c)
		if not label:
			label = f"BTN_{i:02d}"
		buttons.append(Button(label=label, circle=c))
	return image, buttons


def find_start_button(buttons: list[Button]) -> Button:
	for b in buttons:
		if b.label.strip().lower() == "start":
			return b

	# OCR can fail on low-res images. Fallback: choose the most top-center button.
	mean_x = sum(b.circle.cx for b in buttons) / len(buttons)
	best = min(buttons, key=lambda b: (abs(b.circle.cx - mean_x) + 0.55 * b.circle.cy))
	best.label = "Start"
	return best


def sort_buttons(buttons: list[Button], origin: Circle) -> list[Button]:
	return sorted(
		buttons,
		key=lambda b: (
			round(b.circle.cy - origin.cy, 3),
			round(b.circle.cx - origin.cx, 3),
			b.label,
		),
	)


def maybe_write_csv(rows: list[dict[str, str]], out_csv: str | None) -> None:
	if not out_csv:
		return
	fieldnames = [
		"label",
		"x_rel_px",
		"y_rel_px",
		"x_rel_mm",
		"y_rel_mm",
		"radius_px",
		"support_ratio",
		"rms_residual_px",
	]
	with open(out_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def maybe_write_debug_image(image: Any, buttons: list[Button], path: str | None) -> None:
	if not path:
		return
	dbg = image.copy()
	shift_bits = 4
	scale = 1 << shift_bits
	for b in buttons:
		c = b.circle
		center = (int(round(c.cx)), int(round(c.cy)))
		center_fp = (int(round(c.cx * scale)), int(round(c.cy * scale)))
		radius_fp = int(round(c.r * scale))
		cv2.circle(
			dbg,
			center_fp,
			radius_fp,
			(0, 180, 0),
			1,
			lineType=cv2.LINE_AA,
			shift=shift_bits,
		)
		cv2.circle(dbg, center, 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)
		label = f"{b.label} r={c.r:.1f} e={c.rms_residual:.2f}"
		cv2.putText(
			dbg,
			label,
			(center[0] - int(round(c.r)), center[1] - int(round(c.r)) - 3),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.35,
			(20, 20, 20),
			1,
			cv2.LINE_AA,
		)
	cv2.imwrite(path, dbg)


def main() -> None:
	args = parse_args()
	require_dependencies()

	image_path = Path(args.image)
	if not image_path.exists():
		raise SystemExit(f"Image not found: {image_path}")
	if args.samples < 32:
		raise SystemExit("Use at least --samples 32 for robust center estimation.")

	if pytesseract is None:
		print(
			"Warning: pytesseract not installed. Labels may fall back to BTN_## and arrow detection.\n"
			"Install with: pip install pytesseract and system package: tesseract-ocr",
			file=sys.stderr,
		)

	_, buttons = detect_buttons(
		image_path=image_path,
		samples=args.samples,
		radial_search=args.radial_search,
		min_support=args.min_support,
		min_dist=args.min_dist,
		upsample=args.upsample,
		min_radius=args.min_radius,
		max_radius=args.max_radius,
		max_rms=args.max_rms,
		coverage_refine=args.coverage_refine,
		ring_half_width=args.ring_half_width,
		coverage_iters=args.coverage_iters,
		coverage_restarts=args.coverage_restarts,
		coverage_jitter=args.coverage_jitter,
		coverage_angles=args.coverage_angles,
		fixed_radius=args.fixed_radius,
		bf_search_px=args.bf_search_px,
		subpixel_steps=args.subpixel_steps,
	)
	if not buttons:
		raise SystemExit("No complete circular buttons detected.")

	start_button = find_start_button(buttons)
	origin = start_button.circle
	ordered = sort_buttons(buttons, origin)

	rows: list[dict[str, str]] = []
	for b in ordered:
		dx_px = b.circle.cx - origin.cx
		dy_px = b.circle.cy - origin.cy
		if args.mm_per_px is None:
			dx_mm_str = ""
			dy_mm_str = ""
		else:
			dx_mm_str = f"{dx_px * args.mm_per_px:.3f}"
			dy_mm_str = f"{dy_px * args.mm_per_px:.3f}"

		rows.append(
			{
				"label": b.label,
				"x_rel_px": f"{dx_px:.3f}",
				"y_rel_px": f"{dy_px:.3f}",
				"x_rel_mm": dx_mm_str,
				"y_rel_mm": dy_mm_str,
				"radius_px": f"{b.circle.r:.3f}",
				"support_ratio": f"{b.circle.support_ratio:.3f}",
				"rms_residual_px": f"{b.circle.rms_residual:.3f}",
			}
		)

	header = ["label", "x_rel_px", "y_rel_px", "x_rel_mm", "y_rel_mm", "radius_px", "support", "rms_residual_px"]
	print("| " + " | ".join(header) + " |")
	print("| " + " | ".join(["---"] * len(header)) + " |")
	for r in rows:
		row = [
			r["label"],
			r["x_rel_px"],
			r["y_rel_px"],
			r["x_rel_mm"],
			r["y_rel_mm"],
			r["radius_px"],
			r["support_ratio"],
			r["rms_residual_px"],
		]
		print("| " + " | ".join(row) + " |")

	maybe_write_csv(rows, args.csv)

	if args.overlay_image:
		overlay_path = Path(args.overlay_image)
	else:
		overlay_path = image_path.with_name(f"{image_path.stem}.overlay.png")

	# Overlay comes from a second detection pass to keep main path simple.
	image, buttons_dbg = detect_buttons(
		image_path=image_path,
		samples=args.samples,
		radial_search=args.radial_search,
		min_support=args.min_support,
		min_dist=args.min_dist,
		upsample=args.upsample,
		min_radius=args.min_radius,
		max_radius=args.max_radius,
		max_rms=args.max_rms,
		coverage_refine=args.coverage_refine,
		ring_half_width=args.ring_half_width,
		coverage_iters=args.coverage_iters,
		coverage_restarts=args.coverage_restarts,
		coverage_jitter=args.coverage_jitter,
		coverage_angles=args.coverage_angles,
		fixed_radius=args.fixed_radius,
		bf_search_px=args.bf_search_px,
		subpixel_steps=args.subpixel_steps,
	)
	maybe_write_debug_image(image, buttons_dbg, str(overlay_path))
	print(f"Overlay image written: {overlay_path}")


if __name__ == "__main__":
	main()
