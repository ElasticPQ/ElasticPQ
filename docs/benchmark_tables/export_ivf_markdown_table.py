#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


ROOT = Path(__file__).resolve().parent
METHOD_FILES = {
    "PQ": ROOT / "IVF-PQ.csv",
    "OPQ": ROOT / "IVF-OPQ.csv",
    "BAPQ": ROOT / "IVF-BAPQ.csv",
    "RaBitQ": ROOT / "IVF-RaBitQ.csv",
    "EPQ": ROOT / "IVF-EPQ.csv",
}
METHOD_ORDER = ["PQ", "OPQ", "BAPQ", "RaBitQ", "EPQ"]
DATASET_ORDER = {"sift1M": 0, "gist1M": 1, "deep10M": 2}
RA_BITQ_EXCLUDED_METHOD = "RaBitQ"
REFINE_TOKEN = "refine=IndexRefineFlat"
NOMINAL_BITS_RE = re.compile(r"nominal_budget_bits=(\d+)")
LOG_STEM_BITS_RE = re.compile(r"log_stem=joint_[^_]+_(\d+)b_rabitq_")


@dataclass(frozen=True, order=True)
class CaseKey:
    dataset: str
    bits: int
    nprobe: int


@dataclass
class RowMetrics:
    timestamp: datetime
    recall1: float
    recall10: float
    qps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a full IVF-X markdown comparison table from the benchmark CSVs."
        )
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=ROOT,
        help="directory containing IVF-{PQ,OPQ,BAPQ,RaBitQ,EPQ}.csv",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="export refine rows instead of the default no-refine rows",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write markdown to this file instead of stdout",
    )
    return parser.parse_args()


def parse_timestamp(text: str) -> datetime:
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def is_refine_row(row: Dict[str, str]) -> bool:
    return REFINE_TOKEN in row.get("notes", "")


def resolve_bits(method: str, row: Dict[str, str]) -> int:
    if method != "RaBitQ":
        return int(float(row["budget_b"]))

    notes = row.get("notes", "")
    match = NOMINAL_BITS_RE.search(notes)
    if match:
        return int(match.group(1))

    match = LOG_STEM_BITS_RE.search(notes)
    if match:
        return int(match.group(1))

    return int(float(row["budget_b"]))


def load_method_rows(
    method: str,
    path: Path,
    want_refine: bool,
) -> Dict[CaseKey, RowMetrics]:
    rows_by_key: Dict[CaseKey, RowMetrics] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if is_refine_row(row) != want_refine:
                continue

            key = CaseKey(
                dataset=row["dataset"],
                bits=resolve_bits(method, row),
                nprobe=int(float(row["nprobe"])),
            )
            metrics = RowMetrics(
                timestamp=parse_timestamp(row["timestamp"]),
                recall1=float(row["recall_1"]),
                recall10=float(row["recall_10"]),
                qps=float(row["qps"]),
            )
            prev = rows_by_key.get(key)
            if prev is None or metrics.timestamp >= prev.timestamp:
                rows_by_key[key] = metrics
    return rows_by_key


def load_all_methods(
    csv_dir: Path,
    want_refine: bool,
) -> Dict[str, Dict[CaseKey, RowMetrics]]:
    all_rows: Dict[str, Dict[CaseKey, RowMetrics]] = {}
    for method, default_path in METHOD_FILES.items():
        path = csv_dir / default_path.name
        all_rows[method] = load_method_rows(method, path, want_refine)
    return all_rows


def iter_all_keys(method_rows: Dict[str, Dict[CaseKey, RowMetrics]]) -> Iterable[CaseKey]:
    keys = set()
    for rows in method_rows.values():
        keys.update(rows.keys())
    return sorted(
        keys,
        key=lambda key: (
            DATASET_ORDER.get(key.dataset, 999),
            key.dataset,
            key.bits,
            key.nprobe,
        ),
    )


def best_metric(
    method_rows: Dict[str, Dict[CaseKey, RowMetrics]],
    key: CaseKey,
    metric_name: str,
) -> Optional[float]:
    values = []
    for method, rows in method_rows.items():
        if method == RA_BITQ_EXCLUDED_METHOD:
            continue
        metrics = rows.get(key)
        if metrics is None:
            continue
        values.append(getattr(metrics, metric_name))
    if not values:
        return None
    return max(values)


def fmt_recall(value: float) -> str:
    return f"{value:.3f}"


def fmt_qps(value: float) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.1f}k"
    return f"{value:.1f}"


def maybe_bold(text: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"**{text}**"


def build_cell(
    method: str,
    metrics: Optional[RowMetrics],
    best_r1: Optional[float],
    best_r10: Optional[float],
) -> str:
    if metrics is None:
        return "N/A"

    can_bold = method != RA_BITQ_EXCLUDED_METHOD
    r1_wins = can_bold and best_r1 is not None and abs(metrics.recall1 - best_r1) < 1e-12
    r10_wins = can_bold and best_r10 is not None and abs(metrics.recall10 - best_r10) < 1e-12
    r1_text = maybe_bold(fmt_recall(metrics.recall1), r1_wins)
    r10_text = maybe_bold(fmt_recall(metrics.recall10), r10_wins)
    return f"{r1_text} / {r10_text} / {fmt_qps(metrics.qps)}"


def render_table(method_rows: Dict[str, Dict[CaseKey, RowMetrics]]) -> str:
    lines = []
    lines.append("| Dataset | Budget | nprobe | PQ | OPQ | BAPQ | RaBitQ | EPQ |")
    lines.append("| --- | ---: | ---: | --- | --- | --- | --- | --- |")
    for key in iter_all_keys(method_rows):
        best_r1 = best_metric(method_rows, key, "recall1")
        best_r10 = best_metric(method_rows, key, "recall10")
        row = [key.dataset, f"{key.bits}b", str(key.nprobe)]
        for method in METHOD_ORDER:
            row.append(build_cell(method, method_rows[method].get(key), best_r1, best_r10))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    method_rows = load_all_methods(args.csv_dir, args.refine)
    mode_name = "refine" if args.refine else "no-refine"
    for method, rows in method_rows.items():
        if rows:
            continue
        print(
            f"warning: no {mode_name} rows found for {method} under {args.csv_dir}",
            file=sys.stderr,
        )
    markdown = render_table(method_rows)
    if args.output is None:
        print(markdown, end="")
        return

    args.output.write_text(markdown)


if __name__ == "__main__":
    main()
