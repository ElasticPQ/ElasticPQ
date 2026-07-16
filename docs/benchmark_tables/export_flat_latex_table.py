#!/usr/bin/env python3
"""Export paper-facing LaTeX tables for flat ADC benchmark CSVs."""

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
    "PQ": "PQ.csv",
    "OPQ": "OPQ.csv",
    "DP-OPQ": "DP-OPQ.csv",
    "BAPQ": "BAPQ.csv",
    "VAQ": "VAQ.csv",
    "RQ": "RQ.csv",
    "LSQ": "LSQ.csv",
    "EPQ": "AREPQ.csv",
    "EPQ w/o Residual Tail": "EPQ.csv",
    "RaBitQ": "RaBitQ.csv",
}
DEFAULT_METHODS = ["PQ", "OPQ", "DP-OPQ", "BAPQ", "VAQ", "RQ", "LSQ", "EPQ"]
DEFAULT_DATASETS = ["sift1M", "gist1M", "deep10M"]
DEFAULT_BITS = [64, 128]
DEFAULT_METRICS = ["M", "overlap_1000", "recall_1", "recall_10", "recall_100", "J"]
DATASET_ORDER = {name: i for i, name in enumerate(DEFAULT_DATASETS)}
DATASET_DISPLAY = {
    "sift1M": "SIFT1M",
    "gist1M": "GIST1M",
    "deep10M": "DEEP10M",
}


@dataclass(frozen=True)
class CaseKey:
    method: str
    dataset: str
    bits: int


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


def fmt_j(value: float | str) -> str:
    number = float(value)
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if abs(number) >= 100:
        return f"{number:.1f}"
    return f"{number:.4f}"


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


def fmt_text(value: float | str) -> str:
    return latex_escape(str(value))


def parse_note_value(notes: str, key: str) -> Optional[str]:
    match = re.search(rf"(?:^|[;,]\s*){re.escape(key)}=([^;,\s]+)", notes)
    return match.group(1) if match else None


def parse_log_components(csv_dir: Path, row: dict) -> Optional[int]:
    notes = row.get("notes", "")
    run_tag = parse_note_value(notes, "run_tag")
    log_stem = parse_note_value(notes, "log_stem")
    if not run_tag or not log_stem:
        return None
    log_path = csv_dir / "logs" / run_tag / f"{log_stem}.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    for pattern in (
        r"^\s*M:\s*(\d+)\s*$",
        r"^\s*components:\s*(\d+)\s*$",
        r"\[profile\]\s+backend=epq\s+groups=(\d+)",
    ):
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return int(match.group(1))
    return None


def resolve_components(csv_dir: Path, method: str, row: dict) -> Optional[int]:
    impl = row.get("impl_backend", "")
    notes = row.get("notes", "")
    if method in {"PQ", "OPQ", "DP-OPQ", "RQ", "LSQ"}:
        match = re.search(r"\bM=(\d+)", impl) or re.search(r"\bM=(\d+)", notes)
        return int(match.group(1)) if match else None
    if method == "BAPQ":
        match = re.search(r"\bq=(\d+)", impl) or re.search(r"\bq=(\d+)", notes)
        d = safe_float(row, "d")
        if match and d:
            return int(d) // int(match.group(1))
        return None
    if method == "VAQ":
        subspaces = parse_note_value(notes, "subspaces")
        return int(subspaces) if subspaces else parse_log_components(csv_dir, row)
    if method in {"EPQ", "EPQ w/o Residual Tail"}:
        components = parse_log_components(csv_dir, row)
        if components is None or method != "EPQ":
            return components
        tail_stages = parse_note_value(notes, "tail_stages")
        return components - int(tail_stages or 1)
    return None


def build_metric_specs(csv_dir: Path) -> Dict[str, MetricSpec]:
    return {
        "M": MetricSpec(
            header=r"$M$",
            getter=lambda row: resolve_components(csv_dir, row["_method"], row),
            formatter=lambda value: "--" if value is None else str(int(value)),
            best=None,
            bold=False,
        ),
        "overlap_1000": MetricSpec(
            header=r"Overlap@1k",
            getter=lambda row: safe_float(row, "overlap_1000"),
            formatter=fmt_recall,
            best="max",
        ),
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
        "J": MetricSpec(
            header=r"$J$",
            getter=lambda row: safe_float(row, "J"),
            formatter=fmt_j,
            best="min",
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


def load_rows(csv_dir: Path, methods: Iterable[str]) -> Dict[CaseKey, SelectedRow]:
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
                row["_method"] = method
                key = CaseKey(
                    method=method,
                    dataset=row["dataset"],
                    bits=int(float(row["budget_b"])),
                )
                item = SelectedRow(parse_timestamp(row["timestamp"]), row)
                previous = selected.get(key)
                if previous is None or item.timestamp >= previous.timestamp:
                    selected[key] = item
    return selected


def collect_best_values(
    rows: Dict[CaseKey, SelectedRow],
    methods: list[str],
    dataset: str,
    bits: int,
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
            item = rows.get(CaseKey(method, dataset, bits))
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


def render_dataset_table(
    rows: Dict[CaseKey, SelectedRow],
    methods: list[str],
    dataset: str,
    bits_list: list[int],
    metric_names: list[str],
    specs: Dict[str, MetricSpec],
    label_prefix: str,
    caption_prefix: str,
    bold: bool,
) -> str:
    ncols = 1 + len(metric_names)
    colspec = "l" + "r" * len(metric_names)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{latex_escape(caption_prefix)} on {latex_escape(dataset_display(dataset))} under the flat ADC protocol.}}",
        rf"\label{{{label_prefix}-{dataset.lower()}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        "Method & " + " & ".join(specs[name].header for name in metric_names) + r" \\",
        r"\midrule",
    ]
    for bit_index, bits in enumerate(bits_list):
        if bit_index:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{ncols}}}{{c}}{{\textbf{{{bits} bits}}}} \\")
        best = collect_best_values(rows, methods, dataset, bits, metric_names, specs)
        for method in methods:
            item = rows.get(CaseKey(method, dataset, bits))
            row = item.row if item else None
            cells = [latex_escape(method)]
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
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def render_combined_table(
    rows: Dict[CaseKey, SelectedRow],
    methods: list[str],
    datasets: list[str],
    bits_list: list[int],
    metric_names: list[str],
    specs: Dict[str, MetricSpec],
    label: str,
    caption: str,
    bold: bool,
) -> str:
    colspec = "ll" + "r" * len(metric_names)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        "Dataset / Method & Budget & "
        + " & ".join(specs[name].header for name in metric_names)
        + r" \\",
        r"\midrule",
    ]
    first_block = True
    for dataset in datasets:
        for bits in bits_list:
            if not first_block:
                lines.append(r"\midrule")
            first_block = False
            best = collect_best_values(rows, methods, dataset, bits, metric_names, specs)
            for method_index, method in enumerate(methods):
                item = rows.get(CaseKey(method, dataset, bits))
                row = item.row if item else None
                dataset_cell = latex_escape(dataset_display(dataset)) if method_index == 0 else ""
                label_cell = f"{dataset_cell} / {latex_escape(method)}" if dataset_cell else f"/ {latex_escape(method)}"
                cells = [label_cell, f"{bits}b"]
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
        description="Generate paper-facing LaTeX tables from flat benchmark CSV logs."
    )
    parser.add_argument("--csv-dir", type=Path, default=ROOT)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--bits", default=",".join(str(bit) for bit in DEFAULT_BITS))
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument(
        "--layout",
        choices=("by-dataset", "combined"),
        default="by-dataset",
        help="emit one table per dataset or one wide combined table",
    )
    parser.add_argument("--caption-prefix", default="Retrieval accuracy")
    parser.add_argument("--caption", default="Flat ADC retrieval accuracy and cost.")
    parser.add_argument("--label-prefix", default="tab:flat-generated")
    parser.add_argument("--label", default="tab:flat-generated")
    parser.add_argument("--no-bold", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_str_list(args.methods)
    datasets = parse_str_list(args.datasets)
    bits_list = parse_int_list(args.bits)
    metric_names = parse_str_list(args.metrics)
    specs = build_metric_specs(args.csv_dir)
    unknown_metrics = [name for name in metric_names if name not in specs]
    if unknown_metrics:
        raise SystemExit(
            "unknown metrics: "
            + ", ".join(unknown_metrics)
            + "; known metrics: "
            + ", ".join(specs)
        )

    rows = load_rows(args.csv_dir, methods)
    if args.layout == "by-dataset":
        latex = "\n\n".join(
            render_dataset_table(
                rows,
                methods,
                dataset,
                bits_list,
                metric_names,
                specs,
                args.label_prefix,
                args.caption_prefix,
                not args.no_bold,
            )
            for dataset in sorted(datasets, key=lambda item: DATASET_ORDER.get(item, 999))
        )
    else:
        latex = render_combined_table(
            rows,
            methods,
            sorted(datasets, key=lambda item: DATASET_ORDER.get(item, 999)),
            bits_list,
            metric_names,
            specs,
            args.label,
            args.caption,
            not args.no_bold,
        )

    latex += "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(latex)
    else:
        print(latex, end="")


if __name__ == "__main__":
    main()
