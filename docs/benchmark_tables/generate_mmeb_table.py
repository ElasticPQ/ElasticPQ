#!/usr/bin/env python3
"""Generate paper-facing MMEB retrieval tables from C++ benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "mmeb_v2_bench" / "cpp_runs" / "latest"
DEFAULT_CSV = ROOT / "mmeb_mm_core12_128b_table.csv"
DEFAULT_TEX = ROOT / "mmeb_mm_core12_128b_paper_table.tex"
DEFAULT_DOC_TEX = ROOT / "mmeb_mm_core12_128b_table.tex"

METHODS = ["pq", "opq", "bapq", "rq", "lsq", "arepq"]
METHOD_LABELS = {
    "pq": "PQ",
    "opq": "OPQ",
    "bapq": "BAPQ",
    "epq": r"EPQ w/o AR",
    "rq": "RQ",
    "lsq": "LSQ",
    "arepq": "EPQ",
}
TASKS = [
    "MSCOCO_t2i",
    "ImageNet-1K",
    "Kinetics-700",
    "QVHighlight",
    "ViDoRe_docvqa",
    "MMLongBench-doc",
]
TASK_LABELS = {
    "MSCOCO_t2i": r"MSCOCO\_t2i",
    "ImageNet-1K": "ImageNet-1K",
    "Kinetics-700": "Kinetics-700",
    "QVHighlight": "QVHighlight",
    "ViDoRe_docvqa": r"ViDoRe\_docvqa",
    "MMLongBench-doc": "MMLongBench-doc",
}
METRICS = [
    ("hit@1", "Hit@1"),
    ("mrr@1", "MRR@1"),
    ("hit@10", "Hit@10"),
    ("mrr@10", "MRR@10"),
    ("hit@50", "Hit@50"),
    ("mrr@50", "MRR@50"),
    ("hit@100", "Hit@100"),
    ("mrr@100", "MRR@100"),
]
PAPER_METRICS = [
    ("hit@1", "Hit@1"),
    ("hit@10", "Hit@10"),
    ("mrr@10", "MRR@10"),
    ("mrr@100", "MRR@100"),
]


@dataclass(frozen=True)
class MethodRun:
    method: str
    path: Path
    rows: dict[str, dict]
    wall_time: str
    maxrss_kb: int | None


def parse_methods(text: str) -> list[str]:
    methods = [part.strip().lower() for part in text.split(",") if part.strip()]
    unknown = [method for method in methods if method not in METHOD_LABELS]
    if unknown:
        raise SystemExit(f"unknown methods: {', '.join(unknown)}")
    return methods


def parse_tasks(text: str) -> list[str]:
    tasks = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [task for task in tasks if task not in TASK_LABELS]
    if unknown:
        raise SystemExit(f"unknown tasks: {', '.join(unknown)}")
    return tasks


def parse_bits_from_run_dir(method_dir: Path) -> int:
    match = re.search(r"_(\d+)b$", method_dir.name)
    if not match:
        raise RuntimeError(f"cannot parse bit budget from {method_dir}")
    return int(match.group(1))


def parse_time_log(path: Path) -> tuple[str, int | None]:
    if not path.exists():
        raise RuntimeError(f"missing log: {path}")
    wall_time = ""
    maxrss_kb: int | None = None
    saw_exit_ok = False
    for line in path.read_text(errors="replace").splitlines():
        if "Command terminated by signal" in line:
            raise RuntimeError(f"benchmark failed in {path}: {line.strip()}")
        if "Elapsed (wall clock) time" in line:
            match = re.search(r"\):\s*(.+)$", line)
            if match:
                wall_time = match.group(1).strip()
        elif "Maximum resident set size" in line:
            maxrss_kb = int(line.rsplit(":", 1)[-1].strip())
        elif "Exit status:" in line:
            saw_exit_ok = line.rsplit(":", 1)[-1].strip() == "0"
    if not saw_exit_ok:
        raise RuntimeError(f"log does not report Exit status: 0: {path}")
    return wall_time, maxrss_kb


def load_method_run(run_dir: Path, method: str, bits: int) -> MethodRun:
    method_dir = run_dir / f"{method}_{bits}b"
    summary_path = method_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"missing summary: {summary_path}")
    rows = json.loads(summary_path.read_text())
    by_task = {row["task"]: row for row in rows}
    log_path = run_dir / "logs" / f"{method}_{bits}b.log"
    wall_time, maxrss_kb = parse_time_log(log_path)
    return MethodRun(method, method_dir, by_task, wall_time, maxrss_kb)


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


def fmt_metric(value: float) -> str:
    return f"{value:.3f}"


def metric_value(runs: dict[str, MethodRun], method: str, task: str, metric: str) -> float:
    row = runs[method].rows.get(task)
    if row is None:
        raise RuntimeError(f"missing task {task} for method {method}")
    if metric not in row:
        raise RuntimeError(f"missing metric {metric} for {method}/{task}")
    return float(row[metric])


def average_metric(
    runs: dict[str, MethodRun],
    method: str,
    tasks: Iterable[str],
    metric: str,
) -> float:
    values = [metric_value(runs, method, task, metric) for task in tasks]
    return sum(values) / len(values)


def bold_best(values: dict[str, float], method: str, text: str) -> str:
    best = max(values.values())
    if abs(values[method] - best) <= 5e-13:
        return rf"\textbf{{{text}}}"
    return text


def render_table(
    runs: dict[str, MethodRun],
    methods: list[str],
    tasks: list[str],
    bits: int,
    run_dir: Path,
    metrics: list[tuple[str, str]],
    compact: bool,
) -> str:
    environment = "table" if compact else "table*"
    width = r"\columnwidth" if compact else r"\textwidth"
    font_size = r"\tiny" if compact else r"\scriptsize"
    tabcolsep = "1.2pt" if compact else "3pt"
    lines = [
        "% Generated by docs/benchmark_tables/generate_mmeb_table.py",
        rf"\begin{{{environment}}}[!t]",
        r"\centering",
        font_size,
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        r"\renewcommand{\arraystretch}{0.88}" if compact else r"\renewcommand{\arraystretch}{1.0}",
        (
            r"\caption{"
            + (
                f"MMEB-V2 retrieval at a matched {bits}-bit code budget. "
                if compact
                else f"Retrieval performance on MMEB-V2 tasks at a matched {bits}-bit code budget. "
            )
            + (
                ""
                if compact
                else r"All methods use the same Gemini embedding bundle and shared mixed MMEB training pool. "
            )
            + r"Bold marks the numerically highest value for each task and metric.}"
        ),
        r"\label{tab:mm-main}",
        rf"\begin{{tabular*}}{{{width}}}{{@{{\extracolsep{{\fill}}}}llrrrrrr@{{}}}}",
        r"\toprule",
        "Task & Metric & " + " & ".join(METHOD_LABELS[m] for m in methods) + r" \\",
        r"\midrule",
    ]
    first = True
    for task in tasks:
        if not first:
            lines.append(r"\midrule")
        first = False
        for mi, (metric_key, metric_label) in enumerate(metrics):
            prefix = rf"\multirow{{{len(metrics)}}}{{*}}{{{TASK_LABELS[task]}}}" if mi == 0 else ""
            values = {
                method: metric_value(runs, method, task, metric_key)
                for method in methods
            }
            cells = [
                bold_best(values, method, fmt_metric(values[method]))
                for method in methods
            ]
            lines.append(f"{prefix} & {metric_label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for mi, (metric_key, metric_label) in enumerate(metrics):
        prefix = rf"\multirow{{{len(metrics)}}}{{*}}{{Average}}" if mi == 0 else ""
        values = {
            method: average_metric(runs, method, tasks, metric_key)
            for method in methods
        }
        cells = [
            bold_best(values, method, fmt_metric(values[method]))
            for method in methods
        ]
        lines.append(f"{prefix} & {metric_label} & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            rf"\end{{{environment}}}",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(
    path: Path,
    runs: dict[str, MethodRun],
    methods: list[str],
    tasks: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "metric", *[METHOD_LABELS[m] for m in methods]])
        for task in tasks:
            for metric_key, metric_label in METRICS:
                writer.writerow(
                    [
                        task,
                        metric_label,
                        *[
                            fmt_metric(metric_value(runs, method, task, metric_key))
                            for method in methods
                        ],
                    ]
                )
        writer.writerow([])
        for metric_key, metric_label in METRICS:
            writer.writerow(
                [
                    "Average",
                    metric_label,
                    *[
                        fmt_metric(average_metric(runs, method, tasks, metric_key))
                        for method in methods
                    ],
                ]
            )
        writer.writerow([])
        writer.writerow(["method", "wall_time", "maxrss_kb", "run_dir"])
        for method in methods:
            run = runs[method]
            writer.writerow([METHOD_LABELS[method], run.wall_time, run.maxrss_kb, run.path])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--bits", type=int, default=128)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--doc-tex", type=Path, default=DEFAULT_DOC_TEX)
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    tasks = parse_tasks(args.tasks)
    runs = {
        method: load_method_run(args.run_dir, method, args.bits)
        for method in methods
    }
    for run in runs.values():
        parsed_bits = parse_bits_from_run_dir(run.path)
        if parsed_bits != args.bits:
            raise RuntimeError(f"bit mismatch for {run.path}: {parsed_bits} != {args.bits}")

    write_csv(args.csv, runs, methods, tasks)
    tex = render_table(
        runs,
        methods,
        tasks,
        args.bits,
        args.run_dir,
        PAPER_METRICS,
        compact=True,
    )
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.tex.write_text(tex)
    doc_tex = render_table(
        runs,
        methods,
        tasks,
        args.bits,
        args.run_dir,
        METRICS,
        compact=False,
    )
    args.doc_tex.parent.mkdir(parents=True, exist_ok=True)
    args.doc_tex.write_text(doc_tex)
    print(f"wrote {args.csv}")
    print(f"wrote {args.tex}")
    print(f"wrote {args.doc_tex}")


if __name__ == "__main__":
    main()
