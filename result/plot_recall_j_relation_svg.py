#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import html
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


SECTION_RE = re.compile(r"^\s*##\s+(?P<title>.+?)\s*$")
NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
DEFAULT_COLORS = [
    "#D1495B",
    "#2E86AB",
    "#3C9D5D",
    "#E08E45",
    "#8E6CBE",
    "#008B8B",
    "#C05621",
    "#4C6EF5",
    "#AA3A3A",
    "#6B8E23",
]


@dataclass
class RawPoint:
    group: str
    method: str
    recall1: float
    j_value: float


@dataclass
class NormalizedPoint(RawPoint):
    recall1_norm: float
    j_norm: float
    color: str


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot group-normalized recall@1 vs J from record.md into an SVG scatter plot.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=base_dir / "record.md",
        help="Input markdown file, default: result/record.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "recall1_vs_j_normalized.svg",
        help="Output SVG path.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Canvas width.")
    parser.add_argument("--height", type=int, default=860, help="Canvas height.")
    parser.add_argument("--point-radius", type=float, default=6.0, help="Scatter point radius.")
    parser.add_argument("--font-family", default="Arial, Helvetica, sans-serif", help="SVG font family.")
    parser.add_argument("--title", default="", help="Optional plot title. Default: empty.")
    return parser.parse_args(argv)


def clean_cell(text: str) -> str:
    value = text.strip()
    value = value.replace("**", "")
    value = re.sub(r"<[^>]+>", "", value)
    return value.strip()


def normalize_header(text: str) -> str:
    value = clean_cell(text).lower()
    value = value.replace(" ", "")
    value = value.replace("_", "")
    return value


def parse_number(text: str) -> float:
    match = NUMBER_RE.search(clean_cell(text))
    if match is None:
        raise ValueError(f"Cannot parse numeric value from: {text!r}")
    return float(match.group(0))


def split_markdown_row(line: str) -> List[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [part.strip() for part in stripped[1:-1].split("|")]


def is_separator_row(cells: Sequence[str]) -> bool:
    if not cells:
        return False
    for cell in cells:
        compact = cell.replace("-", "").replace(":", "").replace(" ", "")
        if compact:
            return False
    return True


def extract_points_from_table(group: str, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[RawPoint]:
    normalized_headers = [normalize_header(header) for header in headers]
    column_map = {name: idx for idx, name in enumerate(normalized_headers)}

    method_idx = None
    for candidate in ("method", "variant"):
        if candidate in column_map:
            method_idx = column_map[candidate]
            break
    if method_idx is None:
        return []

    recall_idx = None
    for candidate in ("recall@1", "recall1"):
        if candidate in column_map:
            recall_idx = column_map[candidate]
            break
    if recall_idx is None or "j" not in column_map:
        return []

    j_idx = column_map["j"]
    points: List[RawPoint] = []
    for row in rows:
        if len(row) <= max(method_idx, recall_idx, j_idx):
            continue
        method = clean_cell(row[method_idx])
        if not method:
            continue
        points.append(
            RawPoint(
                group=group,
                method=method,
                recall1=parse_number(row[recall_idx]),
                j_value=parse_number(row[j_idx]),
            )
        )
    return points


def parse_record_markdown(text: str) -> List[RawPoint]:
    all_points: List[RawPoint] = []
    current_group: str | None = None
    current_headers: List[str] | None = None
    current_rows: List[List[str]] = []

    def flush_table() -> None:
        nonlocal current_headers, current_rows
        if current_group and current_headers and current_rows:
            all_points.extend(extract_points_from_table(current_group, current_headers, current_rows))
        current_headers = None
        current_rows = []

    for line in text.splitlines():
        section_match = SECTION_RE.match(line)
        if section_match is not None:
            flush_table()
            current_group = clean_cell(section_match.group("title"))
            continue

        if not line.strip().startswith("|"):
            flush_table()
            continue

        cells = split_markdown_row(line)
        if not cells:
            flush_table()
            continue

        if current_headers is None:
            current_headers = cells
            continue

        if is_separator_row(cells):
            continue

        current_rows.append(cells)

    flush_table()

    if not all_points:
        raise ValueError("No valid tables with recall@1 and J were found in the markdown file.")
    return all_points


def minmax_normalize(value: float, vmin: float, vmax: float) -> float:
    if math.isclose(vmax, vmin):
        return 0.5
    return (value - vmin) / (vmax - vmin)


def normalize_points(points: Sequence[RawPoint]) -> List[NormalizedPoint]:
    grouped: Dict[str, List[RawPoint]] = {}
    for point in points:
        grouped.setdefault(point.group, []).append(point)

    normalized: List[NormalizedPoint] = []
    for idx, group in enumerate(grouped):
        group_points = grouped[group]
        recall_values = [point.recall1 for point in group_points]
        j_values = [point.j_value for point in group_points]
        recall_min, recall_max = min(recall_values), max(recall_values)
        j_min, j_max = min(j_values), max(j_values)
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        for point in group_points:
            normalized.append(
                NormalizedPoint(
                    group=point.group,
                    method=point.method,
                    recall1=point.recall1,
                    j_value=point.j_value,
                    recall1_norm=minmax_normalize(point.recall1, recall_min, recall_max),
                    j_norm=minmax_normalize(point.j_value, j_min, j_max),
                    color=color,
                )
            )
    return normalized


def fit_line(points: Sequence[NormalizedPoint]) -> tuple[float, float, float]:
    xs = [point.j_norm for point in points]
    ys = [point.recall1_norm for point in points]
    n = len(points)
    if n == 0:
        raise ValueError("No points available for line fitting.")

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if math.isclose(var_x, 0.0):
        slope = 0.0
    else:
        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x

    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    if math.isclose(ss_tot, 0.0):
        r2 = 1.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return slope, intercept, r2


def clip_line_to_unit_box(slope: float, intercept: float) -> tuple[tuple[float, float], tuple[float, float]]:
    candidates: List[tuple[float, float]] = []

    def add_point(x: float, y: float) -> None:
        if -1e-9 <= x <= 1.0 + 1e-9 and -1e-9 <= y <= 1.0 + 1e-9:
            px = min(max(x, 0.0), 1.0)
            py = min(max(y, 0.0), 1.0)
            point = (px, py)
            if point not in candidates:
                candidates.append(point)

    add_point(0.0, intercept)
    add_point(1.0, slope + intercept)

    if not math.isclose(slope, 0.0):
        add_point((0.0 - intercept) / slope, 0.0)
        add_point((1.0 - intercept) / slope, 1.0)

    if len(candidates) < 2:
        y = min(max(intercept, 0.0), 1.0)
        return (0.0, y), (1.0, y)

    candidates.sort()
    return candidates[0], candidates[-1]


def svg_circle(cx: float, cy: float, r: float, fill: str, stroke: str, stroke_width: float, title: str) -> str:
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" '
        f'fill="{html.escape(fill, quote=True)}" stroke="{html.escape(stroke, quote=True)}" '
        f'stroke-width="{stroke_width:.2f}"><title>{html.escape(title)}</title></circle>'
    )


def svg_text(x: float, y: float, text: str, **attrs: object) -> str:
    attr_parts = [f'x="{x:.2f}"', f'y="{y:.2f}"']
    for key, value in attrs.items():
        attr_name = key.replace("_", "-")
        attr_parts.append(f'{attr_name}="{html.escape(str(value), quote=True)}"')
    return f"<text {' '.join(attr_parts)}>{html.escape(text)}</text>"


def build_svg(
    points: Sequence[NormalizedPoint],
    *,
    slope: float,
    intercept: float,
    r2: float,
    width: int,
    height: int,
    point_radius: float,
    font_family: str,
    title: str,
) -> str:
    outer_left = 90
    outer_top = 60
    outer_bottom = 90
    legend_width = 270
    plot_width = width - outer_left - legend_width - 24
    plot_height = height - outer_top - outer_bottom
    plot_x0 = outer_left
    plot_y0 = outer_top + plot_height
    plot_x1 = plot_x0 + plot_width
    plot_y1 = outer_top

    def map_x(value: float) -> float:
        return plot_x0 + value * plot_width

    def map_y(value: float) -> float:
        return plot_y0 - value * plot_height

    grouped: Dict[str, str] = {}
    for point in points:
        grouped.setdefault(point.group, point.color)

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    if title:
        parts.append(
            svg_text(
                width / 2.0,
                34,
                title,
                font_family=font_family,
                font_size="32",
                font_weight="700",
                text_anchor="middle",
                fill="#111111",
            ),
        )

    for tick in range(6):
        t = tick / 5.0
        x = map_x(t)
        y = map_y(t)
        parts.append(
            f'<line x1="{x:.2f}" y1="{plot_y1:.2f}" x2="{x:.2f}" y2="{plot_y0:.2f}" '
            f'stroke="#e3e3e3" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{plot_x0:.2f}" y1="{y:.2f}" x2="{plot_x1:.2f}" y2="{y:.2f}" '
            f'stroke="#e3e3e3" stroke-width="1"/>'
        )
        parts.append(
            svg_text(
                x,
                plot_y0 + 28,
                f"{t:.1f}",
                font_family=font_family,
                font_size="20",
                text_anchor="middle",
                fill="#444444",
            )
        )
        parts.append(
            svg_text(
                plot_x0 - 16,
                y + 5,
                f"{t:.1f}",
                font_family=font_family,
                font_size="20",
                text_anchor="end",
                fill="#444444",
            )
        )

    parts.append(
        f'<line x1="{plot_x0:.2f}" y1="{plot_y0:.2f}" x2="{plot_x1:.2f}" y2="{plot_y0:.2f}" '
        f'stroke="#222222" stroke-width="1.8"/>'
    )
    parts.append(
        f'<line x1="{plot_x0:.2f}" y1="{plot_y0:.2f}" x2="{plot_x0:.2f}" y2="{plot_y1:.2f}" '
        f'stroke="#222222" stroke-width="1.8"/>'
    )

    parts.append(
        svg_text(
            (plot_x0 + plot_x1) / 2.0,
            height - 28,
            "Normalized J within group",
            font_family=font_family,
            font_size="28",
            text_anchor="middle",
            fill="#111111",
        )
    )
    parts.append(
        f'<text x="28" y="{(plot_y0 + plot_y1) / 2.0:.2f}" transform="rotate(-90 28 {(plot_y0 + plot_y1) / 2.0:.2f})" '
        f'font-family="{html.escape(font_family, quote=True)}" font-size="28" fill="#111111" text-anchor="middle">'
        "Normalized recall@1 within group</text>"
    )

    (x_start, y_start), (x_end, y_end) = clip_line_to_unit_box(slope, intercept)
    parts.append(
        f'<line x1="{map_x(x_start):.2f}" y1="{map_y(y_start):.2f}" x2="{map_x(x_end):.2f}" y2="{map_y(y_end):.2f}" '
        f'stroke="#111111" stroke-width="2.2"/>'
    )

    for point in points:
        tooltip = (
            f"{point.group} | {point.method} | "
            f"recall@1={point.recall1:.4f}, J={point.j_value:.4f}, "
            f"norm_recall@1={point.recall1_norm:.4f}, norm_J={point.j_norm:.4f}"
        )
        parts.append(
            svg_circle(
                map_x(point.j_norm),
                map_y(point.recall1_norm),
                point_radius,
                point.color,
                "#ffffff",
                1.2,
                tooltip,
            )
        )

    legend_x = plot_x1 + 18
    legend_y = outer_top + 12
    parts.append(
        svg_text(
            legend_x,
            legend_y,
            "Groups",
            font_family=font_family,
            font_size="20",
            font_weight="700",
            text_anchor="start",
            fill="#111111",
        )
    )
    offset = 30
    for group, color in grouped.items():
        cy = legend_y + offset - 5
        parts.append(
            f'<circle cx="{legend_x + 8:.2f}" cy="{cy:.2f}" r="6" fill="{html.escape(color, quote=True)}" stroke="#ffffff" stroke-width="1.1"/>'
        )
        parts.append(
            svg_text(
                legend_x + 22,
                legend_y + offset,
                group,
                font_family=font_family,
                font_size="20",
                text_anchor="start",
                fill="#222222",
            )
        )
        offset += 24

    stat_x = legend_x
    stat_y = legend_y + offset + 10
    parts.append(
        f'<rect x="{stat_x - 10:.2f}" y="{stat_y - 25:.2f}" width="276" height="72" fill="#ffffff" fill-opacity="0.92" stroke="#d0d0d0" stroke-width="1"/>'
    )
    parts.append(
        svg_text(
            stat_x,
            stat_y,
            f"fit: y = {slope:.3f}x + {intercept:.3f}",
            font_family=font_family,
            font_size="20",
            text_anchor="start",
            fill="#111111",
        )
    )
    parts.append(
        svg_text(
            stat_x,
            stat_y + 28,
            f"R² = {r2:.3f}",
            font_family=font_family,
            font_size="20",
            text_anchor="start",
            fill="#111111",
        )
    )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    text = args.input.read_text(encoding="utf-8")
    raw_points = parse_record_markdown(text)
    normalized_points = normalize_points(raw_points)
    slope, intercept, r2 = fit_line(normalized_points)
    svg = build_svg(
        normalized_points,
        slope=slope,
        intercept=intercept,
        r2=r2,
        width=int(args.width),
        height=int(args.height),
        point_radius=float(args.point_radius),
        font_family=str(args.font_family),
        title=str(args.title),
    )
    args.output.write_text(svg, encoding="utf-8")
    print(f"wrote svg: {args.output}")
    print(f"points: {len(normalized_points)}")
    print(f"fit: y = {slope:.6f} x + {intercept:.6f}, R^2 = {r2:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
