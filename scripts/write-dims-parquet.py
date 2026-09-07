#!/usr/bin/env python3
"""Convert lbx-dims.md into a structured Parquet file.

Each markdown table row becomes one row with:
- line_no: original line number
- label: row label
- x_rel_px/y_rel_px/x_rel_mm/y_rel_mm/radius_px/support/rms_residual_px

The measurement columns are optionally scaled before writing.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


TABLE_SEPARATOR_RE = re.compile(r"^\s*\|\s*[-\s|:]+\|\s*$")

MEASUREMENT_COLUMNS = [
    "x_rel_px",
    "y_rel_px",
    "x_rel_mm",
    "y_rel_mm",
    "radius_px",
    "support",
    "rms_residual_px",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert lbx-dims.txt notes into Parquet.")
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Multiply all parsed numeric values by this factor before writing (default: 1.0).",
    )
    return parser.parse_args()


def split_markdown_row(raw_text: str) -> list[str]:
    parts = [part.strip() for part in raw_text.strip().strip("|").split("|")]
    return parts


def parse_optional_float(text: str, scale_factor: float) -> float | None:
    text = text.strip()
    if not text:
        return None
    return float(text) * scale_factor


def parse_dims(text_path: Path, scale_factor: float) -> pa.Table:
    rows = []
    with text_path.open("r", encoding="utf-8") as fh:
        lines = list(enumerate(fh, start=1))

    header = None
    for line_no, raw_line in lines:
        raw_text = raw_line.rstrip("\n")
        if not raw_text.strip() or not raw_text.lstrip().startswith("|"):
            continue
        if TABLE_SEPARATOR_RE.match(raw_text):
            continue
        cells = split_markdown_row(raw_text)
        if header is None:
            header = cells
            continue

        if len(cells) != len(header):
            continue

        row = dict(zip(header, cells))
        parsed = {
            "line_no": line_no,
            "label": row.get("label", ""),
            "x_rel_px": parse_optional_float(row.get("x_rel_px", ""), scale_factor),
            "y_rel_px": parse_optional_float(row.get("y_rel_px", ""), scale_factor),
            "x_rel_mm": parse_optional_float(row.get("x_rel_mm", ""), scale_factor),
            "y_rel_mm": parse_optional_float(row.get("y_rel_mm", ""), scale_factor),
            "radius_px": parse_optional_float(row.get("radius_px", ""), scale_factor),
            "support": parse_optional_float(row.get("support", ""), 1.0),
            "rms_residual_px": parse_optional_float(row.get("rms_residual_px", ""), scale_factor),
        }
        rows.append(parsed)

    return pa.Table.from_pylist(rows, schema=pa.schema([
        pa.field("line_no", pa.int32()),
        pa.field("label", pa.string()),
        pa.field("x_rel_px", pa.float64()),
        pa.field("y_rel_px", pa.float64()),
        pa.field("x_rel_mm", pa.float64()),
        pa.field("y_rel_mm", pa.float64()),
        pa.field("radius_px", pa.float64()),
        pa.field("support", pa.float64()),
        pa.field("rms_residual_px", pa.float64()),
    ]))


def format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:g}"


def print_markdown_table(table: pa.Table) -> None:
    print("| line_no | label | x_rel_px | y_rel_px | x_rel_mm | y_rel_mm | radius_px | support | rms_residual_px |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in table.to_pylist():
        print(
            "| "
            + " | ".join(
                [
                    str(row["line_no"]),
                    row["label"] or "",
                    format_optional_float(row["x_rel_px"]),
                    format_optional_float(row["y_rel_px"]),
                    format_optional_float(row["x_rel_mm"]),
                    format_optional_float(row["y_rel_mm"]),
                    format_optional_float(row["radius_px"]),
                    format_optional_float(row["support"]),
                    format_optional_float(row["rms_residual_px"]),
                ]
            )
            + " |"
        )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "resources" / "lbx-dims.md"
    output_path = repo_root / "resources" / "lbx-dims.parquet"

    table = parse_dims(input_path, scale_factor=args.scale_factor)
    pq.write_table(table, output_path, compression="zstd")
    print(f"Wrote {output_path} ({table.num_rows} rows, scale_factor={args.scale_factor})")
    print_markdown_table(table)


if __name__ == "__main__":
    main()
