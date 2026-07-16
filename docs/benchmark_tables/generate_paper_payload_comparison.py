#!/usr/bin/env python3
"""Generate the paper's fixed-budget payload comparison (Figure 1)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parents[2]
DEFAULT_OUTPUT = WORKSPACE_ROOT / "paper" / "assets" / "comparison.pdf"


@dataclass(frozen=True)
class Method:
    name: str
    design: str
    bits: tuple[int, ...]
    dims: tuple[int, ...]
    recall10: float
    color: str
    trailing_note: str = ""
    tail_bits: int = 0


METHODS = (
    Method(
        "PQ",
        "fixed descriptor",
        (8,) * 8,
        (12,) * 8,
        0.2356,
        "#e99a1a",
    ),
    Method(
        "OPQ",
        "fixed descriptor\n+ learned rotation",
        (8,) * 8,
        (12,) * 8,
        0.3819,
        "#65b84a",
    ),
    Method(
        "BAPQ",
        "fixed 4-D blocks\n+ adaptive allocation",
        (8, 7, 7, 6, 6, 5, 5, 4, 4, 4, 3, 3, 2),
        (4,) * 13,
        0.3571,
        "#d4aa18",
        trailing_note="every block is 4-D; 11 zero-bit blocks are omitted",
    ),
    Method(
        "EPQ",
        "partition-guided descriptor\n+ learned transform",
        (12, 11, 11, 11, 11),
        (15, 17, 16, 17, 31),
        0.4753,
        "#28b779",
        tail_bits=8,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def add_payload(
    ax: plt.Axes,
    method: Method,
    y: float,
    left: float,
    width: float,
    height: float,
) -> None:
    total_bits = sum(method.bits) + method.tail_bits
    if total_bits != 64:
        raise ValueError(f"{method.name}: expected a 64-bit payload, got {total_bits}")

    cursor = left
    for index, (bits, dims) in enumerate(zip(method.bits, method.dims)):
        block_width = width * bits / total_bits
        alpha = 0.96 if index % 2 == 0 else 0.82
        ax.add_patch(
            Rectangle(
                (cursor, y - height / 2),
                block_width,
                height,
                facecolor=method.color,
                edgecolor="white",
                linewidth=1.1,
                alpha=alpha,
            )
        )
        if method.name == "BAPQ" or bits >= 4:
            ax.text(
                cursor + block_width / 2,
                y + 0.005,
                str(bits) if method.name == "BAPQ" else f"{bits}b",
                ha="center",
                va="center",
                fontsize=9.3 if method.name == "BAPQ" else 10.2,
                color="#111111",
            )
        if method.name != "BAPQ":
            ax.text(
                cursor + block_width / 2,
                y - height / 2 - 0.021,
                f"{dims}d",
                ha="center",
                va="top",
                fontsize=9.7,
                color="#333333",
            )
        cursor += block_width

    if method.tail_bits:
        tail_width = width * method.tail_bits / total_bits
        ax.add_patch(
            Rectangle(
                (cursor, y - height / 2),
                tail_width,
                height,
                facecolor="#a6adb8",
                edgecolor="white",
                linewidth=1.1,
            )
        )
        ax.text(
            cursor + tail_width / 2,
            y + 0.005,
            f"{method.tail_bits}b",
            ha="center",
            va="center",
            fontsize=9.8,
            color="#111111",
        )
        ax.text(
            cursor + tail_width / 2,
            y - height / 2 - 0.021,
            "tail",
            ha="center",
            va="top",
            fontsize=9.0,
            color="#333333",
        )

    if method.trailing_note:
        ax.text(
            left,
            y - height / 2 - 0.021,
            f"Bits/block; {method.trailing_note}",
            ha="left",
            va="top",
            fontsize=9.4,
            color="#555555",
        )


def add_recall(ax: plt.Axes, method: Method, y: float, left: float, width: float) -> None:
    scale_max = 0.55
    bar_width = width * method.recall10 / scale_max
    face = "#cfe7d5" if method.name == "EPQ" else "#eeeeee"
    edge = "#52a66c" if method.name == "EPQ" else "#8a8a8a"
    ax.add_patch(
        Rectangle(
            (left, y - 0.039),
            bar_width,
            0.078,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
        )
    )
    ax.text(
        left + bar_width - 0.006,
        y,
        f"{100 * method.recall10:.2f}%",
        ha="right",
        va="center",
        fontsize=10.3,
        fontweight="bold" if method.name == "EPQ" else "normal",
        color="#111111",
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(10.4, 4.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.015,
        0.955,
        r"DEEP10M  |  $d=96$  |  total payload $B=64$ bits",
        ha="left",
        va="top",
        fontsize=13.5,
        fontweight="bold",
    )
    ax.text(0.015, 0.835, "Method", fontsize=11.4, fontweight="bold")
    ax.text(
        0.260,
        0.835,
        "Stored code (block width is proportional to bits)",
        fontsize=11.4,
        fontweight="bold",
    )
    ax.text(0.835, 0.835, "Recall@10", fontsize=11.4, fontweight="bold")

    row_y = (0.710, 0.520, 0.330, 0.140)
    payload_left = 0.260
    payload_width = 0.500
    payload_height = 0.095
    recall_left = 0.835
    recall_width = 0.145

    for method, y in zip(METHODS, row_y):
        ax.text(
            0.015,
            y + 0.038,
            method.name,
            ha="left",
            va="center",
            fontsize=15.0,
            fontweight="bold",
        )
        ax.text(
            0.015,
            y - 0.006,
            method.design,
            ha="left",
            va="top",
            fontsize=9.4,
            color="#444444",
            linespacing=1.08,
        )
        add_payload(ax, method, y, payload_left, payload_width, payload_height)
        add_recall(ax, method, y, recall_left, recall_width)

    ax.plot([0.80, 0.80], [0.095, 0.82], color="#b8b8b8", linewidth=0.9)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
