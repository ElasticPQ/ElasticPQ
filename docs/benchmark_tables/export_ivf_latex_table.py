#!/usr/bin/env python3
"""Export paper-facing LaTeX tables for IVF benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parent

METHOD_FILES = {
    "PQ": "IVF-PQ.csv",
    "OPQ": "IVF-OPQ.csv",
    "DP-OPQ": "IVF-DP-OPQ.csv",
    "BAPQ": "IVF-BAPQ.csv",
    "VAQ": "IVF-VAQ.csv",
    "EPQ": "IVF-EPQ.csv",
    "RaBitQ": "IVF-RaBitQ.csv",
    "RQ": "IVF-RQ.csv",
    "LSQ": "IVF-LSQ.csv",
}
DEFAULT_METHODS = ["PQ", "OPQ", "DP-OPQ", "BAPQ", "VAQ", "RQ", "LSQ", "EPQ"]
DEFAULT_DATASETS = ["sift1M", "gist1M", "deep10M"]
DEFAULT_BITS = [64, 128]
DEFAULT_METRICS = ["recall_1", "recall_10", "recall_100", "overlap_1000", "qps"]
DATASET_ORDER = {name: i for i, name in enumerate(DEFAULT_DATASETS)}
DATASET_DISPLAY = {
    "sift1M": "SIFT1M",
    "gist1M": "GIST1M",
    "deep10M": "DEEP10M",
}
REFINE_TOKEN = "refine=IndexRefineFlat"
NOMINAL_BITS_RE = re.compile(r"nominal_budget_bits=(\d+)")
LOG_STEM_BITS_RE = re.compile(r"log_stem=joint_[^_]+_(\d+)b_rabitq_")


@dataclass(frozen=True)
class CaseKey:
    method: str
    dataset: str
    bits: int
    nlist: int
    nprobe: int


@dataclass
class MetricSpec:
    header: str
    getter: Callable[[dict], Optional[float | str]]
    formatter: Callable[[float | str], str]
    best: Optional[str] = None
    bold: bool = True


@dataclass
class SelectedRow:
    timestamp: datetime
    row: dict


def parse_timestamp(text: str) -> datetime:
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def dataset_display(dataset: str) -> str:
    return DATASET_DISPLAY.get(dataset, dataset)


def safe_float(row: dict, key: str) -> Optional[float]:
    value = row.get(key, "").strip()
    if not value or value == "N/A":
        return None
    return float(value)


def fmt_recall(value: float | str) -> str:
    return f"{float(value):.4f}"


def fmt_qps(value: float | str) -> str:
    number = float(value)
    if number >= 1000:
        return f"{number:.0f}"
    return f"{number:.1f}"


def fmt_ms(value: float | str) -> str:
    return f"{float(value):.3f}"


def fmt_seconds(value: float | str) -> str:
    number = float(value)
    if number >= 100:
        return f"{number:.1f}"
    return f"{number:.3f}"


def fmt_mb(value: float | str) -> str:
    return f"{float(value):.1f}"


def fmt_candidates(value: float | str) -> str:
    number = float(value)
    if number >= 1000:
        return f"{number:.0f}"
    return f"{number:.1f}"


def is_refine_row(row: dict) -> bool:
    return REFINE_TOKEN in row.get("notes", "")


def resolve_bits(method: str, row: dict) -> int:
    if method != "RaBitQ":
        return int(float(row["budget_b"]))
    notes = row.get("notes", "")
    match = NOMINAL_BITS_RE.search(notes) or LOG_STEM_BITS_RE.search(notes)
    if match:
        return int(match.group(1))
    return int(float(row["budget_b"]))


def build_metric_specs() -> Dict[str, MetricSpec]:
    return {
        "recall_1": MetricSpec(
            header=r"R@1",
            getter=lambda row: safe_float(row, "recall_1"),
            formatter=fmt_recall,
            best="max",
        ),
        "recall_10": MetricSpec(
            header=r"R@10",
            getter=lambda row: safe_float(row, "recall_10"),
            formatter=fmt_recall,
            best="max",
        ),
        "recall_100": MetricSpec(
            header=r"R@100",
            getter=lambda row: safe_float(row, "recall_100"),
            formatter=fmt_recall,
            best="max",
        ),
        "overlap_1000": MetricSpec(
            header=r"Overlap@1k",
            getter=lambda row: safe_float(row, "overlap_1000"),
            formatter=fmt_recall,
            best="max",
        ),
        "qps": MetricSpec(
            header=r"QPS",
            getter=lambda row: safe_float(row, "qps"),
            formatter=fmt_qps,
            best=None,
            bold=False,
        ),
        "search_ms_per_q": MetricSpec(
            header=r"ms/q",
            getter=lambda row: safe_float(row, "search_ms_per_q"),
            formatter=fmt_ms,
            best=None,
            bold=False,
        ),
        "avg_candidates_per_q": MetricSpec(
            header=r"Cand./q",
            getter=lambda row: safe_float(row, "avg_candidates_per_q"),
            formatter=fmt_candidates,
            best=None,
            bold=False,
        ),
        "train_total_s": MetricSpec(
            header=r"Train (s)",
            getter=lambda row: safe_float(row, "train_total_s"),
            formatter=fmt_seconds,
            best=None,
            bold=False,
        ),
        "index_size_mb": MetricSpec(
            header=r"Size (MiB)",
            getter=lambda row: safe_float(row, "index_size_mb"),
            formatter=fmt_mb,
            best=None,
            bold=False,
        ),
    }


def load_rows(
    csv_dir: Path,
    methods: Iterable[str],
    want_refine: bool,
) -> Dict[CaseKey, SelectedRow]:
    selected: Dict[CaseKey, SelectedRow] = {}
    for method in methods:
        csv_name = METHOD_FILES.get(method)
        if csv_name is None:
            raise SystemExit(f"unknown method {method}; known methods: {', '.join(METHOD_FILES)}")
        path = csv_dir / csv_name
        if not path.exists():
            print(f"warning: missing CSV for {method}: {path}", file=sys.stderr)
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if is_refine_row(row) != want_refine:
                    continue
                row["_method"] = method
                key = CaseKey(
                    method=method,
                    dataset=row["dataset"],
                    bits=resolve_bits(method, row),
                    nlist=int(float(row["nlist"])),
                    nprobe=int(float(row["nprobe"])),
                )
                item = SelectedRow(parse_timestamp(row["timestamp"]), row)
                previous = selected.get(key)
                if previous is None or item.timestamp >= previous.timestamp:
                    selected[key] = item
    return selected


def iter_available_cases(
    rows: Dict[CaseKey, SelectedRow],
    methods: list[str],
    datasets: list[str],
    bits_list: list[int],
) -> list[tuple[str, int, int, int]]:
    cases = set()
    method_set = set(methods)
    dataset_set = set(datasets)
    bits_set = set(bits_list)
    for key in rows:
        if key.method not in method_set:
            continue
        if key.dataset not in dataset_set:
            continue
        if key.bits not in bits_set:
            continue
        cases.add((key.dataset, key.bits, key.nlist, key.nprobe))
    return sorted(
        cases,
        key=lambda item: (
            DATASET_ORDER.get(item[0], 999),
            item[0],
            item[1],
            item[2],
            item[3],
        ),
    )


def filter_cases(
    cases: list[tuple[str, int, int, int]],
    policy: str,
    explicit_nprobes: Optional[list[int]],
) -> list[tuple[str, int, int, int]]:
    if policy == "all":
        return cases
    if policy == "explicit":
        wanted = set(explicit_nprobes or [])
        return [case for case in cases if case[3] in wanted]
    return [case for case in cases if case[3] == max(1, case[2] // 256)]


def collect_best_values(
    rows: Dict[CaseKey, SelectedRow],
    methods: list[str],
    dataset: str,
    bits: int,
    nlist: int,
    nprobe: int,
    metric_names: list[str],
    specs: Dict[str, MetricSpec],
) -> Dict[str, Optional[float]]:
    best: Dict[str, Optional[float]] = {}
    for metric_name in metric_names:
        spec = specs[metric_name]
        if spec.best is None:
            best[metric_name] = None
            continue
        values = []
        for method in methods:
            item = rows.get(CaseKey(method, dataset, bits, nlist, nprobe))
            if item is None:
                continue
            value = spec.getter(item.row)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if not values:
            best[metric_name] = None
        elif spec.best == "min":
            best[metric_name] = min(values)
        else:
            best[metric_name] = max(values)
    return best


def render_cell(
    row: Optional[dict],
    metric_name: str,
    spec: MetricSpec,
    best_value: Optional[float],
    bold: bool,
) -> str:
    if row is None:
        return "--"
    value = spec.getter(row)
    if value is None:
        return "--"
    text = spec.formatter(value)
    if (
        bold
        and spec.bold
        and best_value is not None
        and isinstance(value, (int, float))
        and abs(float(value) - best_value) < 1e-12
    ):
        return rf"\textbf{{{text}}}"
    return text


def render_table(
    rows: Dict[CaseKey, SelectedRow],
    methods: list[str],
    cases: list[tuple[str, int, int, int]],
    metric_names: list[str],
    specs: Dict[str, MetricSpec],
    caption: str,
    label: str,
    bold: bool,
    mode_name: str,
) -> str:
    colspec = "lllrr" + "r" * len(metric_names)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        rf"\caption{{{latex_escape(caption)} ({mode_name}).}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        "Dataset & Budget & Method & nlist & nprobe & "
        + " & ".join(specs[name].header for name in metric_names)
        + r" \\",
        r"\midrule",
    ]

    previous_case: Optional[tuple[str, int, int, int]] = None
    for dataset, bits, nlist, nprobe in cases:
        if previous_case is not None:
            lines.append(r"\midrule")
        previous_case = (dataset, bits, nlist, nprobe)
        best = collect_best_values(
            rows,
            methods,
            dataset,
            bits,
            nlist,
            nprobe,
            metric_names,
            specs,
        )
        for method_index, method in enumerate(methods):
            item = rows.get(CaseKey(method, dataset, bits, nlist, nprobe))
            row = item.row if item else None
            dataset_cell = latex_escape(dataset_display(dataset)) if method_index == 0 else ""
            budget_cell = f"{bits}b" if method_index == 0 else ""
            nlist_cell = str(nlist) if method_index == 0 else ""
            nprobe_cell = str(nprobe) if method_index == 0 else ""
            cells = [
                dataset_cell,
                budget_cell,
                latex_escape(method),
                nlist_cell,
                nprobe_cell,
            ]
            for metric_name in metric_names:
                cells.append(
                    render_cell(
                        row,
                        metric_name,
                        specs[metric_name],
                        best[metric_name],
                        bold,
                    )
                )
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-facing LaTeX tables from IVF benchmark CSV logs."
    )
    parser.add_argument("--csv-dir", type=Path, default=ROOT)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--bits", default=",".join(str(bit) for bit in DEFAULT_BITS))
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--refine", action="store_true", help="use IVF+RefineFlat rows")
    parser.add_argument(
        "--nprobe-policy",
        choices=("single", "all", "explicit"),
        default="single",
        help="single means nprobe=nlist/256; explicit uses --nprobes",
    )
    parser.add_argument("--nprobes", default="", help="comma-separated nprobe list for explicit policy")
    parser.add_argument(
        "--caption",
        default="End-to-end IVF recall-throughput comparison at matched payload budget",
    )
    parser.add_argument("--label", default="tab:ivf-generated")
    parser.add_argument("--no-bold", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_str_list(args.methods)
    datasets = parse_str_list(args.datasets)
    bits_list = parse_int_list(args.bits)
    metric_names = parse_str_list(args.metrics)
    explicit_nprobes = parse_int_list(args.nprobes) if args.nprobes.strip() else None
    specs = build_metric_specs()
    unknown_metrics = [name for name in metric_names if name not in specs]
    if unknown_metrics:
        raise SystemExit(
            "unknown metrics: "
            + ", ".join(unknown_metrics)
            + "; known metrics: "
            + ", ".join(specs)
        )

    rows = load_rows(args.csv_dir, methods, args.refine)
    cases = iter_available_cases(rows, methods, datasets, bits_list)
    cases = filter_cases(cases, args.nprobe_policy, explicit_nprobes)
    if not cases:
        raise SystemExit("no IVF rows matched the requested filters")

    mode_name = "with RefineFlat" if args.refine else "no refine"
    latex = (
        render_table(
            rows,
            methods,
            cases,
            metric_names,
            specs,
            args.caption,
            args.label,
            not args.no_bold,
            mode_name,
        )
        + "\n"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(latex)
    else:
        print(latex, end="")


if __name__ == "__main__":
    main()
