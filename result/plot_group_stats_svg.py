#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


LINE_RE = re.compile(
    r"""
    group\[(?P<group_id>\d+)\]
    \s+ndims=(?P<ndims>\d+)
    \s+bits=(?P<bits>\d+)
    \s+D_proxy=(?P<dproxy>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
    """,
    re.VERBOSE,
)
HEADER_RE = re.compile(
    r"""
    quantizer=(?P<quantizer>\S+)
    \s+space=(?P<space>.+?)
    \s+entries=(?P<entries>\d+)
    \s+total_dims=(?P<total_dims>\d+)
    \s+total_bits=(?P<total_bits>\d+)
    \s*$
    """,
    re.VERBOSE,
)


@dataclass
class GroupStat:
    group_id: int
    ndims: int
    bits: int
    dproxy: float

    @property
    def dproxy_per_dim(self) -> float:
        if self.ndims <= 0:
            raise ValueError(f"ndims must be positive, got {self.ndims}")
        return float(self.dproxy) / float(self.ndims)


@dataclass
class QuantizerStats:
    quantizer: str
    space: str
    stats: List[GroupStat]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render [group-stats] lines into a segmented SVG bar.",
    )
    parser.add_argument("--d-min", type=float, default=None, help="Lower bound for normalized D/d color mapping. Default: auto from input.")
    parser.add_argument("--d-max", type=float, default=None, help="Upper bound for normalized D/d color mapping. Default: auto from input.")
    parser.add_argument(
        "--auto-range-mode",
        choices=("quantile", "minmax"),
        default="quantile",
        help="How to auto-compute D/d range when --d-min/--d-max are not provided.",
    )
    parser.add_argument("--quantile-low", type=float, default=0.05, help="Lower quantile for robust auto range.")
    parser.add_argument("--quantile-high", type=float, default=0.95, help="Upper quantile for robust auto range.")
    parser.add_argument("--input", type=Path, default=None, help="Input text file. If omitted, read from stdin.")
    parser.add_argument("--output", type=Path, required=True, help="Output SVG path.")
    parser.add_argument("--colorbar-output", type=Path, default=None, help="Optional output path for the colorbar SVG.")
    parser.add_argument("--width", type=int, default=1400, help="SVG canvas width.")
    parser.add_argument("--bar-height", type=int, default=72, help="Height of the segmented bar.")
    parser.add_argument("--left-pad", type=int, default=36, help="Left/right outer padding.")
    parser.add_argument("--top-pad", type=int, default=28, help="Top padding above the bar.")
    parser.add_argument("--bottom-pad", type=int, default=54, help="Bottom padding below labels.")
    parser.add_argument("--gap", type=int, default=2, help="Gap between segments.")
    parser.add_argument("--stroke", default="#ffffff", help="Segment border color.")
    parser.add_argument("--stroke-width", type=float, default=1.0, help="Segment border width.")
    parser.add_argument("--font-family", default="Arial, Helvetica, sans-serif", help="SVG font-family.")
    parser.add_argument("--font-size", type=int, default=20, help="Main label font size.")
    parser.add_argument("--colorbar-font-size", type=int, default=18, help="Colorbar label font size.")
    parser.add_argument(
        "--colorbar-orientation",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="Colorbar layout orientation.",
    )
    parser.add_argument("--bits-suffix", default="b", help="Suffix for bits labels, default: b.")
    return parser.parse_args(argv)


def read_input_text(path: Path | None) -> str:
    if path is None:
        return sys.stdin.read()
    return path.read_text(encoding="utf-8")


def parse_group_stats(text: str) -> List[QuantizerStats]:
    blocks: List[QuantizerStats] = []
    current_name = "group_stats"
    current_space = "unknown"
    current_stats: List[GroupStat] = []

    def flush_current() -> None:
        nonlocal current_stats
        if current_stats:
            current_stats.sort(key=lambda x: x.group_id)
            blocks.append(
                QuantizerStats(
                    quantizer=current_name,
                    space=current_space,
                    stats=current_stats,
                )
            )
            current_stats = []

    for line in text.splitlines():
        line_s = line.strip()
        if not line_s:
            continue

        header = HEADER_RE.search(line_s)
        if header is not None:
            flush_current()
            current_name = str(header.group("quantizer"))
            current_space = str(header.group("space"))
            continue

        match = LINE_RE.search(line_s)
        if match is None:
            continue
        current_stats.append(
            GroupStat(
                group_id=int(match.group("group_id")),
                ndims=int(match.group("ndims")),
                bits=int(match.group("bits")),
                dproxy=float(match.group("dproxy")),
            )
        )

    flush_current()

    if not blocks:
        raise ValueError("No valid [group-stats] lines found in input.")

    return blocks


def clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def lerp_color_rgb(t: float) -> str:
    t = clamp01(t)
    red = (231, 76, 60)
    yellow = (241, 196, 15)
    green = (46, 204, 113)

    if t <= 0.5:
        u = t / 0.5
        rgb = tuple(int(round((1.0 - u) * a + u * b)) for a, b in zip(green, yellow))
    else:
        u = (t - 0.5) / 0.5
        rgb = tuple(int(round((1.0 - u) * a + u * b)) for a, b in zip(yellow, red))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def dproxy_to_color(dproxy_per_dim: float, d_min: float, d_max: float) -> str:
    if d_max <= d_min:
        return lerp_color_rgb(0.5)
    t = (float(dproxy_per_dim) - float(d_min)) / (float(d_max) - float(d_min))
    return lerp_color_rgb(t)


def segment_widths(
    stats: Sequence[GroupStat],
    total_width: float,
    gap: float,
) -> List[float]:
    n = len(stats)
    usable = total_width - max(0, n - 1) * gap
    if usable <= 0:
        raise ValueError("Canvas width too small for the requested number of segments and gaps.")

    total_dims = sum(item.ndims for item in stats)
    if total_dims <= 0:
        raise ValueError("Total ndims must be positive.")

    raw = [usable * item.ndims / total_dims for item in stats]
    widths = [max(1.0, w) for w in raw]
    excess = sum(widths) - usable
    if excess > 0:
        scale = usable / sum(widths)
        widths = [max(1.0, w * scale) for w in widths]
    return widths


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    font_size: int,
    font_family: str,
    fill: str,
    dominant_baseline: str | None = None,
) -> str:
    baseline_attr = ""
    if dominant_baseline is not None:
        baseline_attr = f' dominant-baseline="{html.escape(dominant_baseline, quote=True)}"'
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="middle"{baseline_attr} '
        f'font-family="{html.escape(font_family, quote=True)}" font-size="{font_size}" '
        f'fill="{fill}">{html.escape(text)}</text>'
    )


def svg_under_bracket(
    x: float,
    y_top: float,
    width: float,
    depth: float,
    *,
    inset: float,
    stroke: str,
    stroke_width: float,
) -> str:
    inset_i = max(0.0, float(inset))
    half_sw = max(0.0, float(stroke_width)) / 2.0
    x0 = x + inset_i + half_sw
    x1 = x + width - inset_i - half_sw
    y1 = y_top + depth
    ym = y_top + depth / 2.0
    return (
        f'<path d="M {x0:.2f} {y_top:.2f} '
        f'L {x0:.2f} {y1:.2f} '
        f'M {x0:.2f} {ym:.2f} '
        f'L {x1:.2f} {ym:.2f} '
        f'M {x1:.2f} {y_top:.2f} '
        f'L {x1:.2f} {y1:.2f}" '
        f'fill="none" stroke="{html.escape(stroke, quote=True)}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def build_svg(
    stats: Sequence[GroupStat],
    *,
    d_min: float,
    d_max: float,
    width: int,
    bar_height: int,
    left_pad: int,
    top_pad: int,
    bottom_pad: int,
    gap: int,
    stroke: str,
    stroke_width: float,
    font_family: str,
    font_size: int,
    bits_suffix: str,
) -> str:
    bar_y = float(top_pad)
    height = int(top_pad + bar_height + bottom_pad)
    bar_width = float(width - 2 * left_pad)
    widths = segment_widths(stats, total_width=bar_width, gap=float(gap))

    center_text_y = bar_y + bar_height / 2.0
    bracket_top_y = bar_y + bar_height + 6.0
    bracket_depth = 10.0
    bottom_text_y = bracket_top_y + bracket_depth + 20.0

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
    ]

    x = float(left_pad)
    for item, seg_w in zip(stats, widths):
        color = dproxy_to_color(item.dproxy_per_dim, d_min=d_min, d_max=d_max)
        parts.append(
            f'<rect x="{x:.2f}" y="{bar_y:.2f}" width="{seg_w:.2f}" height="{bar_height:.2f}" '
            f'rx="2" ry="2" fill="{color}" stroke="{html.escape(stroke, quote=True)}" '
            f'stroke-width="{stroke_width}"/>'
        )
        cx = x + seg_w / 2.0
        parts.append(
            svg_text(
                cx,
                center_text_y,
                f"{item.bits}{bits_suffix}",
                font_size=font_size,
                font_family=font_family,
                fill="#111111",
                dominant_baseline="middle",
            )
        )
        parts.append(
            svg_under_bracket(
                x,
                bracket_top_y,
                seg_w,
                bracket_depth,
                inset=0.0,
                stroke="#333333",
                stroke_width=max(1.2, stroke_width),
            )
        )
        parts.append(
            svg_text(
                cx,
                bottom_text_y,
                f"{item.ndims}d",
                font_size=font_size,
                font_family=font_family,
                fill="#111111",
            )
        )
        x += seg_w + gap

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def default_colorbar_output_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_colorbar.svg")


def _format_range_for_filename(x: float) -> str:
    s = f"{x:.6f}"
    s = s.rstrip("0").rstrip(".")
    if not s:
        s = "0"
    return s.replace("-", "neg").replace(".", "p")


def default_shared_colorbar_output_path(output: Path, *, d_min: float, d_max: float) -> Path:
    d_mid = 0.5 * (float(d_min) + float(d_max))
    return output.with_name(
        f"{output.stem}_colorbar"
        f"_min_{_format_range_for_filename(d_min)}"
        f"_mid_{_format_range_for_filename(d_mid)}"
        f"_max_{_format_range_for_filename(d_max)}"
        f"{output.suffix}"
    )


def slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    s = s.strip("._-").lower()
    return s or "quantizer"


def resolve_output_path(base_output: Path, quantizer_name: str, *, multi: bool) -> Path:
    if not multi:
        return base_output
    return base_output.with_name(f"{base_output.stem}_{slugify(quantizer_name)}{base_output.suffix}")


def compute_auto_range(blocks: Sequence[QuantizerStats]) -> tuple[float, float]:
    vals: List[float] = []
    for block in blocks:
        for item in block.stats:
            vals.append(item.dproxy_per_dim)
    if not vals:
        raise ValueError("No group values available to compute auto D/d range.")
    return min(vals), max(vals)


def compute_quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("No values for quantile computation.")
    q1 = min(max(float(q), 0.0), 1.0)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q1 * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def compute_quantile_range(
    blocks: Sequence[QuantizerStats],
    *,
    q_low: float,
    q_high: float,
) -> tuple[float, float]:
    vals: List[float] = []
    for block in blocks:
        for item in block.stats:
            vals.append(item.dproxy_per_dim)
    if not vals:
        raise ValueError("No group values available to compute quantile D/d range.")
    vals.sort()
    lo = min(max(float(q_low), 0.0), 1.0)
    hi = min(max(float(q_high), 0.0), 1.0)
    if lo > hi:
        lo, hi = hi, lo
    return compute_quantile(vals, lo), compute_quantile(vals, hi)


def build_colorbar_svg(
    *,
    d_min: float,
    d_max: float,
    orientation: str,
    font_family: str,
    font_size: int,
) -> str:
    orientation_s = str(orientation).lower()
    if orientation_s == "vertical":
        width = 120
        height = 360
        bar_x = 42
        bar_y = 24
        bar_w = 36
        bar_h = 300
        gradient = (
            f'<linearGradient id="dproxy-gradient" x1="0%" y1="100%" x2="0%" y2="0%">\n'
            f'  <stop offset="0%" stop-color="{lerp_color_rgb(0.0)}"/>\n'
            f'  <stop offset="50%" stop-color="{lerp_color_rgb(0.5)}"/>\n'
            f'  <stop offset="100%" stop-color="{lerp_color_rgb(1.0)}"/>\n'
            f'</linearGradient>'
        )
        body = [
            f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{bar_h}" rx="3" ry="3" fill="url(#dproxy-gradient)" stroke="#999999" stroke-width="1"/>',
        ]
    else:
        width = 720
        height = 64
        bar_x = 24
        bar_y = 18
        bar_w = width - 2 * bar_x
        bar_h = 28
        gradient = (
            f'<linearGradient id="dproxy-gradient" x1="0%" y1="0%" x2="100%" y2="0%">\n'
            f'  <stop offset="0%" stop-color="{lerp_color_rgb(0.0)}"/>\n'
            f'  <stop offset="50%" stop-color="{lerp_color_rgb(0.5)}"/>\n'
            f'  <stop offset="100%" stop-color="{lerp_color_rgb(1.0)}"/>\n'
            f'</linearGradient>'
        )
        body = [
            f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{bar_h}" rx="3" ry="3" fill="url(#dproxy-gradient)" stroke="#999999" stroke-width="1"/>',
        ]

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<defs>\n{gradient}\n</defs>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    text = read_input_text(args.input)
    blocks = parse_group_stats(text)
    if args.d_min is not None or args.d_max is not None:
        auto_d_min, auto_d_max = compute_auto_range(blocks)
        d_min = float(args.d_min) if args.d_min is not None else float(auto_d_min)
        d_max = float(args.d_max) if args.d_max is not None else float(auto_d_max)
        range_mode_used = "manual-mixed"
    else:
        if str(args.auto_range_mode) == "minmax":
            d_min, d_max = compute_auto_range(blocks)
        else:
            d_min, d_max = compute_quantile_range(
                blocks,
                q_low=float(args.quantile_low),
                q_high=float(args.quantile_high),
            )
        range_mode_used = str(args.auto_range_mode)
    multi = len(blocks) > 1
    shared_colorbar_output = (
        args.colorbar_output
        if args.colorbar_output is not None else
        default_shared_colorbar_output_path(args.output, d_min=d_min, d_max=d_max)
    )
    colorbar_svg = build_colorbar_svg(
        d_min=d_min,
        d_max=d_max,
        orientation=str(args.colorbar_orientation),
        font_family=str(args.font_family),
        font_size=int(args.colorbar_font_size),
    )

    for block in blocks:
        output_path = resolve_output_path(args.output, block.quantizer, multi=multi)

        svg = build_svg(
            block.stats,
            d_min=d_min,
            d_max=d_max,
            width=int(args.width),
            bar_height=int(args.bar_height),
            left_pad=int(args.left_pad),
            top_pad=int(args.top_pad),
            bottom_pad=int(args.bottom_pad),
            gap=int(args.gap),
            stroke=str(args.stroke),
            stroke_width=float(args.stroke_width),
            font_family=str(args.font_family),
            font_size=int(args.font_size),
            bits_suffix=str(args.bits_suffix),
        )
        output_path.write_text(svg, encoding="utf-8")
        print(f"wrote svg: {output_path}")

    shared_colorbar_output.write_text(colorbar_svg, encoding="utf-8")
    print(f"wrote shared colorbar svg: {shared_colorbar_output}")
    if range_mode_used == "quantile":
        print(
            f"using D/d range: min={d_min:.10f} max={d_max:.10f} "
            f"(mode=quantile q_low={float(args.quantile_low):.3f} q_high={float(args.quantile_high):.3f})"
        )
    else:
        print(f"using D/d range: min={d_min:.10f} max={d_max:.10f} (mode={range_mode_used})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
