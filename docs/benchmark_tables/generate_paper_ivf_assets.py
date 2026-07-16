#!/usr/bin/env python3
"""Generate paper-facing IVF plots and tables from benchmark CSVs or logs.

The default source is the committed unified CSVs. The corresponding frozen
no-refine matrix logs are also supported for audit and backfill:

- PQ/OPQ/BAPQ from the matched IVF-protocol baseline run on 2026-06-24.
- DP-OPQ from the dedicated local-reproduction matrix on 2026-07-14.
- RQ/LSQ from the dedicated IVF AQ matrix run on 2026-07-09.
- EPQ from the AREPQ no-refine run; the paper-facing EPQ label denotes the
  full payload with the auxiliary residual byte inside the same budget.

RaBitQ is intentionally not part of the default method set because its actual
footprint is not matched to the 64/128-bit PQ-family budgets.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parents[2]
PAPER_ROOT = WORKSPACE_ROOT / "paper"
LOG_ROOT = ROOT / "logs"

DEFAULT_BASELINE_LOG_DIR = LOG_ROOT / (
    "joint_topk1000_fullsplits_serial_t12_host_"
    "cpuset1x3x5x7x9x11x13x15x17x19x21x23_norefine_20260624"
)
DEFAULT_EPQ_LOG_DIR = LOG_ROOT / (
    "joint_arepq_inductive_full_topk1000_serial_t12_host_even_"
    "20260713T142805Z"
)
DEFAULT_RQLSQ_LOG_DIR = LOG_ROOT / (
    "joint_topk1000_fullsplits_serial_t12_host_"
    "cpuset1x3x5x7x9x11x13x15x17x19x21x23_rqlsq_20260709"
)
DEFAULT_VAQ_LOG_DIR = LOG_ROOT / (
    "joint_vaq_full_topk1000_fullsplits_serial_t12_host_"
    "cpuset1x3x5x7x9x11x13x15x17x19x21x23_cfgstandard_20260714T132628Z"
)
DEFAULT_DPOPQ_LOG_DIR = LOG_ROOT / (
    "joint_dpopq_full_noalign_topk1000_serial_t12_host_"
    "cpuset1x3x5x7x9x11x13x15x17x19x21x23_20260714T0924Z"
)

METHOD_FILES = {
    "PQ": "IVF-PQ.csv",
    "OPQ": "IVF-OPQ.csv",
    "DP-OPQ": "IVF-DP-OPQ.csv",
    "BAPQ": "IVF-BAPQ.csv",
    "VAQ": "IVF-VAQ.csv",
    "EPQ": "IVF-AREPQ.csv",
    "EPQ without AR": "IVF-EPQ.csv",
    "RaBitQ": "IVF-RaBitQ.csv",
    "RQ": "IVF-RQ.csv",
    "LSQ": "IVF-LSQ.csv",
}
METHOD_TARGETS = {
    "PQ": "IVF+PQ",
    "OPQ": "IVF+OPQ",
    "DP-OPQ": "IVF+DPOPQ",
    "BAPQ": "IVF+BAPQ",
    "VAQ": "IVF+VAQ",
    "EPQ": "IVF+AREPQ",
    "EPQ without AR": "IVF+EPQ",
    "RaBitQ": "IVF+RaBitQ",
    "RQ": "IVF+RQ",
    "LSQ": "IVF+LSQ",
}
TARGET_METHODS = {value: key for key, value in METHOD_TARGETS.items()}

DEFAULT_METHODS = ["PQ", "OPQ", "DP-OPQ", "BAPQ", "VAQ", "RQ", "LSQ", "EPQ"]
REFINE_TABLE_METHODS = ["PQ", "OPQ", "BAPQ", "EPQ"]
DEFAULT_DATASETS = ["sift1M", "gist1M", "deep10M"]
DEFAULT_BITS = [64, 128]
PRODUCT_CODE_METHODS = ["PQ", "OPQ", "DP-OPQ", "BAPQ", "VAQ"]
ADDITIVE_METHODS = ["RQ", "LSQ"]
DATASET_DISPLAY = {
    "sift1M": "SIFT1M",
    "gist1M": "GIST1M",
    "deep10M": "DEEP10M",
}
DATASET_ORDER = {name: i for i, name in enumerate(DEFAULT_DATASETS)}
EXPECTED_NLIST = {
    "sift1M": 4096,
    "gist1M": 4096,
    "deep10M": 16384,
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
class RowItem:
    timestamp: datetime
    row: dict


def parse_str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_timestamp(text: str) -> datetime:
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


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


def safe_float(row: dict, name: str) -> Optional[float]:
    value = str(row.get(name, "")).strip()
    if not value or value == "N/A":
        return None
    return float(value)


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


def load_rows_from_csvs(
    csv_dir: Path,
    methods: Iterable[str],
) -> dict[CaseKey, RowItem]:
    selected: dict[CaseKey, RowItem] = {}
    for method in methods:
        path = csv_dir / METHOD_FILES[method]
        if not path.exists():
            raise RuntimeError(f"missing CSV for {method}: {path}")
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if is_refine_row(row):
                    continue
                key = CaseKey(
                    method=method,
                    dataset=row["dataset"],
                    bits=resolve_bits(method, row),
                    nlist=int(float(row["nlist"])),
                    nprobe=int(float(row["nprobe"])),
                )
                item = RowItem(parse_timestamp(row["timestamp"]), row)
                previous = selected.get(key)
                if previous is None or item.timestamp >= previous.timestamp:
                    selected[key] = item
    return selected


def load_rows_from_logs(
    methods: Iterable[str],
    baseline_log_dir: Path,
    dpopq_log_dir: Path,
    rqlsq_log_dir: Path,
    epq_log_dir: Path,
    vaq_log_dir: Path,
) -> dict[CaseKey, RowItem]:
    sys.path.insert(0, str(ROOT))
    import backfill_ivf_unified_csvs as backfill  # pylint: disable=import-error

    selected: dict[CaseKey, RowItem] = {}

    method_set = set(methods)
    baseline_targets = {
        METHOD_TARGETS[method]
        for method in method_set
        if method not in {"DP-OPQ", "EPQ", "RQ", "LSQ", "VAQ"}
    }
    dpopq_targets = {
        METHOD_TARGETS[method]
        for method in method_set
        if method == "DP-OPQ"
    }
    rqlsq_targets = {
        METHOD_TARGETS[method]
        for method in method_set
        if method in {"RQ", "LSQ"}
    }
    epq_targets = {METHOD_TARGETS["EPQ"]} if "EPQ" in method_set else set()
    vaq_targets = {METHOD_TARGETS["VAQ"]} if "VAQ" in method_set else set()

    source_groups: list[tuple[Path, set[str]]] = []
    if baseline_targets:
        source_groups.append((baseline_log_dir, baseline_targets))
    if dpopq_targets:
        source_groups.append((dpopq_log_dir, dpopq_targets))
    if rqlsq_targets:
        source_groups.append((rqlsq_log_dir, rqlsq_targets))
    if epq_targets:
        source_groups.append((epq_log_dir, epq_targets))
    if vaq_targets:
        source_groups.append((vaq_log_dir, vaq_targets))

    for log_dir, targets in source_groups:
        if not log_dir.exists():
            raise RuntimeError(f"log directory does not exist: {log_dir}")
        rows_by_target = backfill.load_rows_from_dirs([log_dir], targets)
        for target in targets:
            method = TARGET_METHODS[target]
            for row in rows_by_target[target]:
                if is_refine_row(row):
                    continue
                key = CaseKey(
                    method=method,
                    dataset=row["dataset"],
                    bits=resolve_bits(method, row),
                    nlist=int(float(row["nlist"])),
                    nprobe=int(float(row["nprobe"])),
                )
                item = RowItem(parse_timestamp(row["timestamp"]), row)
                previous = selected.get(key)
                if previous is None or item.timestamp >= previous.timestamp:
                    selected[key] = item
    return selected


def expected_nprobes(dataset: str) -> tuple[int, list[int]]:
    nlist = EXPECTED_NLIST[dataset]
    return nlist, [nlist // 1024, nlist // 512, nlist // 256, nlist // 128]


def validate_coverage(
    rows: dict[CaseKey, RowItem],
    methods: list[str],
    datasets: list[str],
    bits_list: list[int],
) -> list[tuple[str, int, int, list[int]]]:
    cases: list[tuple[str, int, int, list[int]]] = []
    missing: list[str] = []
    for dataset in datasets:
        if dataset not in EXPECTED_NLIST:
            raise RuntimeError(f"no frozen IVF nlist rule for dataset {dataset}")
        nlist, nprobes = expected_nprobes(dataset)
        for bits in bits_list:
            for method in methods:
                for nprobe in nprobes:
                    key = CaseKey(method, dataset, bits, nlist, nprobe)
                    if key not in rows:
                        missing.append(
                            f"{method} {dataset} {bits}b nlist={nlist} nprobe={nprobe}"
                        )
            cases.append((dataset, bits, nlist, nprobes))
    if missing:
        sample = "\n".join(f"  - {item}" for item in missing[:32])
        more = "" if len(missing) <= 32 else f"\n  ... {len(missing) - 32} more"
        raise RuntimeError(f"missing IVF rows:\n{sample}{more}")
    return cases


def fmt_metric(value: float) -> str:
    return f"{value:.4f}"


def fmt_qps(value: float) -> str:
    return f"{value:.0f}" if value >= 1000 else f"{value:.1f}"


def fmt_seconds(value: float) -> str:
    return f"{value:.1f}"


def fmt_microseconds(value: float) -> str:
    return f"{value:.2f}"


def fmt_gib(value: float) -> str:
    return f"{value:.1f}"


def fmt_mib(value: float) -> str:
    return f"{value:.1f}"


def required_float(row: dict, name: str) -> float:
    value = safe_float(row, name)
    if value is None:
        raise RuntimeError(f"missing metric {name} in row {row}")
    return value


def fmt_range(values: list[float], formatter) -> str:
    low = min(values)
    high = max(values)
    low_text = formatter(low)
    high_text = formatter(high)
    if low_text == high_text:
        return low_text
    return f"{low_text}--{high_text}"


def ivf_payload_build_seconds(row: dict) -> float:
    return required_float(row, "train_total_s") + required_float(row, "add_encode_s")


def ivf_non_codebook_train_seconds(row: dict) -> float:
    return required_float(row, "prep_s")


def ivf_group_values(
    rows: dict[CaseKey, RowItem],
    methods: list[str],
    dataset: str,
    bits: int,
    nlist: int,
    nprobe: int,
    value_fn,
) -> list[float]:
    return [
        value_fn(rows[CaseKey(method, dataset, bits, nlist, nprobe)].row)
        for method in methods
    ]


def ivf_group_sweep_values(
    rows: dict[CaseKey, RowItem],
    methods: list[str],
    dataset: str,
    bits: int,
    nlist: int,
    nprobes: list[int],
    value_fn,
) -> list[float]:
    values: list[float] = []
    for method in methods:
        method_values = [
            value_fn(rows[CaseKey(method, dataset, bits, nlist, nprobe)].row)
            for nprobe in nprobes
        ]
        values.append(max(method_values))
    return values


def metric_header(metric: str) -> str:
    return {
        "recall_1": "R@1",
        "recall_10": "R@10",
        "recall_100": "R@100",
        "recall_1000": "R@1000",
        "overlap_1000": "Overlap@1k",
    }[metric]


def render_cell(
    row: dict,
    metric: str,
    best_value: Optional[float],
    bold: bool = True,
) -> str:
    value = safe_float(row, metric)
    if value is None:
        return "--"
    if metric == "qps":
        return fmt_qps(value)
    text = fmt_metric(value)
    if bold and best_value is not None and abs(value - best_value) < 1e-12:
        return rf"\textbf{{{text}}}"
    return text


def render_ivf_main_table(
    rows: dict[CaseKey, RowItem],
    methods: list[str],
    cases: list[tuple[str, int, int, list[int]]],
) -> str:
    single_cases = [
        (dataset, bits, nlist, max(1, nlist // 256))
        for dataset, bits, nlist, _ in cases
    ]

    def fmt_kqps(value: float) -> str:
        value /= 1000.0
        return f"{value:.1f}" if value >= 10.0 else f"{value:.2f}"

    lines = [
        "% Generated by docs/benchmark_tables/generate_paper_ivf_assets.py",
        r"\begin{table*}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Representative no-refine IVF quality and throughput at matched "
        r"payload budgets and $nprobe=nlist/256$. Each cell reports "
        r"Recall@10~/~thousands of queries per second (kQPS); "
        r"Figure~\ref{fig:ivf-recall-nprobe} reports the complete matched-routing sweep.}",
        r"\label{tab:ivf-main}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}",
        r"\toprule",
        "Method & "
        + " & ".join(
            f"{latex_escape(DATASET_DISPLAY.get(dataset, dataset))}/{bits}b"
            for dataset, bits, _, _ in single_cases
        )
        + r" \\",
        r"\midrule",
    ]

    for method in methods:
        cells = [latex_escape(method)]
        for dataset, bits, nlist, nprobe in single_cases:
            row = rows[CaseKey(method, dataset, bits, nlist, nprobe)].row
            recall = required_float(row, "recall_10")
            qps = required_float(row, "qps")
            cells.append(rf"${fmt_metric(recall)}/{fmt_kqps(qps)}$")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""])
    return "\n".join(lines)


def render_ivf_refine_table(
    rows: dict[CaseKey, RowItem],
    cases: list[tuple[str, int, int, list[int]]],
) -> str:
    methods = REFINE_TABLE_METHODS
    metric = "recall_1000"
    lines = [
        "% Generated by docs/benchmark_tables/generate_paper_ivf_assets.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Diagnostic exact-vector-reranked IVF Recall@1 at $nprobe=nlist/256$. "
        r"Exact reranking uses the same candidate depth of 1,000, for which "
        r"Recall@10 equals candidate-set coverage.}",
        r"\label{tab:ivf-refine}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrrrrr@{}}",
        r"\toprule",
        "Dataset & Budget & nprobe & " + " & ".join(methods) + r" \\",
        r"\midrule",
    ]
    for dataset, bits, nlist, _ in cases:
        nprobe = max(1, nlist // 256)
        values = [
            safe_float(rows[CaseKey(method, dataset, bits, nlist, nprobe)].row, metric)
            for method in methods
        ]
        best = max(value for value in values if value is not None)
        cells = [
            latex_escape(DATASET_DISPLAY.get(dataset, dataset)),
            f"{bits}b",
            str(nprobe),
        ]
        for value in values:
            if value is None:
                cells.append("--")
            elif abs(value - best) < 1e-12:
                cells.append(rf"\textbf{{{fmt_metric(value)}}}")
            else:
                cells.append(fmt_metric(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""])
    return "\n".join(lines)


def render_ivf_epq_time_table(
    rows: dict[CaseKey, RowItem],
    cases: list[tuple[str, int, int, list[int]]],
) -> str:
    method = "EPQ"
    lines = [
        "% Generated by docs/benchmark_tables/generate_paper_ivf_assets.py",
        r"\begin{table*}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        r"\caption{IVF build and search diagnostics at $nprobe=nlist/256$ under "
        r"the common C++/AVX2 no-refine evaluation protocol. Coarse is the "
        r"separately timed IVF $k$-means stage for the EPQ run; Structure is its "
        r"one-time partition-search cost; RAM is peak RSS, and Size is "
        r"the serialized index. Tail Aux., dashes, training protocols, and other "
        r"columns are defined in Section~\ref{sec:setup}.}",
        r"\label{tab:ivf-epq-time}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrlrrrrrrrrrr@{}}",
        r"\toprule",
        r"Dataset & Budget & nprobe & Coarse & Method & \multicolumn{3}{c}{Train (s)} "
        r"& Add & Enc. & Search & QPS & RAM & Size & Tail Aux. \\",
        r"\cmidrule(lr){6-8}",
        r"& & & (s) & & Structure & Prep & Codebook & (s) "
        r"& ($\mu$s/vec) & (ms/q) & & (GiB) & (MiB) & (MiB) \\",
        r"\midrule",
    ]
    first_case = True
    for dataset, bits, nlist, nprobes in cases:
        if not first_case:
            lines.append(r"\midrule")
        first_case = False
        nprobe = max(1, nlist // 256)
        rep = rows[CaseKey(method, dataset, bits, nlist, nprobe)].row
        sweep_rows = [
            rows[CaseKey(method, dataset, bits, nlist, probe)].row
            for probe in nprobes
        ]
        structure = max(
            value for value in (safe_float(row, "structure_s") for row in sweep_rows)
            if value is not None
        )
        ram = max(
            value for value in (safe_float(row, "peak_ram_gb") for row in sweep_rows)
            if value is not None
        )

        def range_cells(methods: list[str]) -> list[str]:
            return [
                "0.0",
                fmt_range(
                    ivf_group_values(
                        rows,
                        methods,
                        dataset,
                        bits,
                        nlist,
                        nprobe,
                        ivf_non_codebook_train_seconds,
                    ),
                    fmt_seconds,
                ),
                fmt_range(
                    ivf_group_values(
                        rows,
                        methods,
                        dataset,
                        bits,
                        nlist,
                        nprobe,
                        lambda item: required_float(item, "codebook_s"),
                    ),
                    fmt_seconds,
                ),
                fmt_range(
                    ivf_group_values(
                        rows,
                        methods,
                        dataset,
                        bits,
                        nlist,
                        nprobe,
                        lambda item: required_float(item, "add_encode_s"),
                    ),
                    fmt_seconds,
                ),
                fmt_range(
                    ivf_group_values(
                        rows,
                        methods,
                        dataset,
                        bits,
                        nlist,
                        nprobe,
                        lambda item: required_float(item, "encode_us_per_vec"),
                    ),
                    fmt_microseconds,
                ),
                fmt_range(
                    ivf_group_values(
                        rows,
                        methods,
                        dataset,
                        bits,
                        nlist,
                        nprobe,
                        lambda item: required_float(item, "search_ms_per_q"),
                    ),
                    lambda value: f"{value:.3f}",
                ),
                fmt_range(
                    ivf_group_values(
                        rows,
                        methods,
                        dataset,
                        bits,
                        nlist,
                        nprobe,
                        lambda item: required_float(item, "qps"),
                    ),
                    fmt_qps,
                ),
                fmt_range(
                    ivf_group_sweep_values(
                        rows,
                        methods,
                        dataset,
                        bits,
                        nlist,
                        nprobes,
                        lambda item: required_float(item, "peak_ram_gb"),
                    ),
                    fmt_gib,
                ),
                fmt_range(
                    ivf_group_values(
                        rows,
                        methods,
                        dataset,
                        bits,
                        nlist,
                        nprobe,
                        lambda item: required_float(item, "index_size_mb"),
                    ),
                    fmt_mib,
                ),
                "--",
            ]

        lines.append(
            " & ".join(
                [
                    latex_escape(DATASET_DISPLAY.get(dataset, dataset)),
                    f"{bits}b",
                    str(nprobe),
                    fmt_seconds(safe_float(rep, "coarse_train_s") or 0.0),
                    "PC",
                    *range_cells(PRODUCT_CODE_METHODS),
                ]
            )
            + r" \\"
        )
        lines.append(
            " & ".join(["", "", "", "", "Add.", *range_cells(ADDITIVE_METHODS)])
            + r" \\"
        )
        epq_cells = [
            "",
            "",
            "",
            "",
            "EPQ",
            fmt_seconds(structure),
            fmt_seconds(required_float(rep, "prep_s")),
            fmt_seconds(required_float(rep, "codebook_s")),
            fmt_seconds(safe_float(rep, "add_encode_s") or 0.0),
            fmt_microseconds(safe_float(rep, "encode_us_per_vec") or 0.0),
            f"{required_float(rep, 'search_ms_per_q'):.3f}",
            fmt_qps(safe_float(rep, "qps") or 0.0),
            fmt_gib(ram),
            fmt_mib(safe_float(rep, "index_size_mb") or 0.0),
            fmt_mib(
                (
                    required_float(rep, "tail_resident_auxiliary_table_bytes")
                    + required_float(rep, "tail_query_lut_bytes_per_query")
                )
                / (1024.0 * 1024.0)
            ),
        ]
        lines.append(" & ".join(epq_cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""])
    return "\n".join(lines)


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PdfCanvas:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.commands: list[str] = []

    def raw(self, command: str) -> None:
        self.commands.append(command)

    def line_width(self, width: float) -> None:
        self.raw(f"{width:.3f} w")

    def stroke_rgb(self, color: tuple[float, float, float]) -> None:
        self.raw(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG")

    def fill_rgb(self, color: tuple[float, float, float]) -> None:
        self.raw(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg")

    def line(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.raw(f"{x1:.3f} {y1:.3f} m {x2:.3f} {y2:.3f} l S")

    def polyline(self, points: list[tuple[float, float]]) -> None:
        if len(points) < 2:
            return
        parts = [f"{points[0][0]:.3f} {points[0][1]:.3f} m"]
        parts.extend(f"{x:.3f} {y:.3f} l" for x, y in points[1:])
        parts.append("S")
        self.raw(" ".join(parts))

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 8.0,
        font: str = "F1",
    ) -> None:
        self.raw(
            f"BT /{font} {size:.2f} Tf {x:.3f} {y:.3f} Td "
            f"({pdf_escape(text)}) Tj ET"
        )

    def text_center(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 8.0,
        font: str = "F1",
    ) -> None:
        self.text(x - 0.25 * size * len(text), y, text, size, font)

    def text_right(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 8.0,
        font: str = "F1",
    ) -> None:
        self.text(x - 0.50 * size * len(text), y, text, size, font)

    def marker(
        self,
        x: float,
        y: float,
        shape: str,
        size: float,
        color: tuple[float, float, float],
    ) -> None:
        self.stroke_rgb(color)
        self.fill_rgb((1.0, 1.0, 1.0))
        self.line_width(0.8)
        s = size
        if shape == "circle":
            k = 0.5522847498 * s
            self.raw(
                f"{x + s:.3f} {y:.3f} m "
                f"{x + s:.3f} {y + k:.3f} {x + k:.3f} {y + s:.3f} {x:.3f} {y + s:.3f} c "
                f"{x - k:.3f} {y + s:.3f} {x - s:.3f} {y + k:.3f} {x - s:.3f} {y:.3f} c "
                f"{x - s:.3f} {y - k:.3f} {x - k:.3f} {y - s:.3f} {x:.3f} {y - s:.3f} c "
                f"{x + k:.3f} {y - s:.3f} {x + s:.3f} {y - k:.3f} {x + s:.3f} {y:.3f} c B"
            )
        elif shape == "square":
            self.raw(
                f"{x - s:.3f} {y - s:.3f} m {x + s:.3f} {y - s:.3f} l "
                f"{x + s:.3f} {y + s:.3f} l {x - s:.3f} {y + s:.3f} l h B"
            )
        elif shape == "triangle":
            self.raw(
                f"{x:.3f} {y + s:.3f} m {x + s:.3f} {y - s:.3f} l "
                f"{x - s:.3f} {y - s:.3f} l h B"
            )
        else:
            self.raw(
                f"{x:.3f} {y + s:.3f} m {x + s:.3f} {y:.3f} l "
                f"{x:.3f} {y - s:.3f} l {x - s:.3f} {y:.3f} l h B"
            )

    def stream(self) -> bytes:
        return ("\n".join(self.commands) + "\n").encode("latin-1")


def write_pdf(path: Path, width: float, height: float, stream: bytes) -> None:
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.3f} {height:.3f}] "
            f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> "
            f"/Contents 6 0 R >>"
        ).encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream",
    ]
    chunks = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref_offset = sum(len(chunk) for chunk in chunks)
    chunks.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    chunks.append(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        chunks.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    chunks.append(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(chunks))


def nice_qps_ticks(min_value: float, max_value: float) -> list[float]:
    candidates = [
        500,
        1000,
        2000,
        3000,
        5000,
        10000,
        20000,
        30000,
        50000,
        70000,
    ]
    ticks = [value for value in candidates if min_value <= value <= max_value]
    if len(ticks) >= 2:
        return ticks
    return [min_value, max_value] if min_value < max_value else [min_value]


def format_qps_tick(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def metric_value(row: dict, metric: str) -> float:
    value = safe_float(row, metric)
    if value is None:
        raise RuntimeError(f"missing metric {metric} in row {row}")
    return value


def draw_ivf_figure(
    rows: dict[CaseKey, RowItem],
    methods: list[str],
    cases: list[tuple[str, int, int, list[int]]],
    metric: str,
    x_axis: str,
    output: Path,
) -> None:
    width = 540.0
    height = 400.0
    canvas = PdfCanvas(width, height)
    colors = {
        "PQ": (0.10, 0.10, 0.10),
        "OPQ": (0.00, 0.30, 0.65),
        "DP-OPQ": (0.43, 0.24, 0.62),
        "BAPQ": (0.78, 0.22, 0.15),
        "VAQ": (0.58, 0.18, 0.42),
        "RQ": (0.66, 0.11, 0.16),
        "LSQ": (0.82, 0.49, 0.00),
        "EPQ": (0.00, 0.47, 0.24),
        "RaBitQ": (0.50, 0.25, 0.60),
    }
    shapes = {
        "PQ": "circle",
        "OPQ": "square",
        "DP-OPQ": "triangle",
        "BAPQ": "triangle",
        "VAQ": "square",
        "RQ": "square",
        "LSQ": "triangle",
        "EPQ": "diamond",
        "RaBitQ": "circle",
    }

    title = "No-refine IVF payload ranking under matched routing parameters"
    if x_axis == "qps":
        title = "No-refine IVF recall-QPS diagnostic"
    canvas.text_center(width / 2, height - 16, title, 10, "F2")
    legend_font_size = 8.0
    legend_line_w = 18.0
    legend_text_gap = 5.0
    legend_item_gap = 17.0

    def legend_text_width(label: str) -> float:
        return 0.58 * legend_font_size * len(label)

    legend_widths = [
        legend_line_w + legend_text_gap + legend_text_width(method)
        for method in methods
    ]
    legend_total_w = sum(legend_widths) + legend_item_gap * max(0, len(methods) - 1)
    legend_x = (width - legend_total_w) / 2
    for method, legend_w in zip(methods, legend_widths):
        x = legend_x
        y = height - 34
        color = colors.get(method, (0, 0, 0))
        canvas.stroke_rgb(color)
        canvas.line_width(1.2 if method != "EPQ" else 1.7)
        canvas.line(x, y + 3, x + 18, y + 3)
        canvas.marker(x + 9, y + 3, shapes.get(method, "circle"), 2.6, color)
        canvas.fill_rgb((0, 0, 0))
        canvas.text(x + 23, y, method, 8.0, "F1")
        legend_x += legend_w + legend_item_gap

    left = 42.0
    right = 12.0
    bottom = 36.0
    top = 56.0
    gap_x = 22.0
    gap_y = 32.0
    panel_w = (width - left - right - 2 * gap_x) / 3.0
    panel_h = (height - top - bottom - gap_y) / 2.0

    panel_cases = {
        (dataset, bits): (dataset, bits, nlist, nprobes)
        for dataset, bits, nlist, nprobes in cases
    }

    for row_index, bits in enumerate(DEFAULT_BITS):
        if bits not in {case[1] for case in cases}:
            continue
        for col_index, dataset in enumerate(DEFAULT_DATASETS):
            if (dataset, bits) not in panel_cases:
                continue
            _, _, nlist, nprobes = panel_cases[(dataset, bits)]
            x0 = left + col_index * (panel_w + gap_x)
            y0 = height - top - (row_index + 1) * panel_h - row_index * gap_y

            panel_points: dict[str, list[tuple[int, float, float]]] = {}
            x_values: list[float] = []
            y_values: list[float] = []
            for method in methods:
                values = []
                for nprobe in nprobes:
                    row = rows[CaseKey(method, dataset, bits, nlist, nprobe)].row
                    qps = metric_value(row, "qps")
                    recall = metric_value(row, metric)
                    x_value = math.log10(qps) if x_axis == "qps" else float(nprobe)
                    values.append((nprobe, x_value, recall))
                    x_values.append(x_value)
                    y_values.append(recall)
                panel_points[method] = values

            x_min = min(x_values)
            x_max = max(x_values)
            if abs(x_max - x_min) < 1e-9:
                x_min -= 1.0
                x_max += 1.0
            x_pad = 0.05 * (x_max - x_min)
            x_min -= x_pad
            x_max += x_pad
            y_min = max(0.0, math.floor((min(y_values) - 0.02) / 0.05) * 0.05)
            y_max = min(1.0, math.ceil((max(y_values) + 0.02) / 0.05) * 0.05)
            if y_max <= y_min:
                y_max = min(1.0, y_min + 0.1)

            def map_x(value: float) -> float:
                return x0 + (value - x_min) / (x_max - x_min) * panel_w

            def map_y(value: float) -> float:
                return y0 + (value - y_min) / (y_max - y_min) * panel_h

            canvas.stroke_rgb((0.82, 0.82, 0.82))
            canvas.line_width(0.35)
            for tick_index in range(3):
                y_tick = y_min + tick_index * (y_max - y_min) / 2.0
                y = map_y(y_tick)
                canvas.line(x0, y, x0 + panel_w, y)
            if x_axis == "qps":
                qps_values = [10 ** value for value in x_values]
                for tick in nice_qps_ticks(min(qps_values), max(qps_values)):
                    x = map_x(math.log10(tick))
                    canvas.line(x, y0, x, y0 + panel_h)
            else:
                for nprobe in nprobes:
                    x = map_x(float(nprobe))
                    canvas.line(x, y0, x, y0 + panel_h)

            canvas.stroke_rgb((0.0, 0.0, 0.0))
            canvas.line_width(0.7)
            canvas.line(x0, y0, x0 + panel_w, y0)
            canvas.line(x0, y0, x0, y0 + panel_h)

            for tick_index in range(3):
                y_tick = y_min + tick_index * (y_max - y_min) / 2.0
                y = map_y(y_tick)
                canvas.stroke_rgb((0.0, 0.0, 0.0))
                canvas.line_width(0.6)
                canvas.line(x0 - 2, y, x0, y)
                canvas.fill_rgb((0, 0, 0))
                canvas.text_right(x0 - 4, y - 2.2, f"{y_tick:.2f}", 6.5)

            if x_axis == "qps":
                qps_values = [10 ** value for value in x_values]
                for tick in nice_qps_ticks(min(qps_values), max(qps_values)):
                    x = map_x(math.log10(tick))
                    canvas.stroke_rgb((0.0, 0.0, 0.0))
                    canvas.line_width(0.6)
                    canvas.line(x, y0, x, y0 - 2)
                    canvas.fill_rgb((0, 0, 0))
                    canvas.text_center(x, y0 - 10, format_qps_tick(tick), 6.5)
            else:
                for nprobe in nprobes:
                    x = map_x(float(nprobe))
                    canvas.stroke_rgb((0.0, 0.0, 0.0))
                    canvas.line_width(0.6)
                    canvas.line(x, y0, x, y0 - 2)
                    canvas.fill_rgb((0, 0, 0))
                    canvas.text_center(x, y0 - 10, str(nprobe), 6.5)

            canvas.fill_rgb((0, 0, 0))
            title = f"{DATASET_DISPLAY.get(dataset, dataset)} {bits}b"
            canvas.text_center(x0 + panel_w / 2, y0 + panel_h + 8, title, 8.5, "F2")
            if col_index == 0:
                canvas.text(x0, y0 + panel_h + 1, metric_header(metric), 7.0, "F1")
            if row_index == 1:
                xlabel = "QPS (log scale)" if x_axis == "qps" else "nprobe"
                canvas.text_center(x0 + panel_w / 2, y0 - 22, xlabel, 7.0)

            for method in methods:
                color = colors.get(method, (0, 0, 0))
                points = [(map_x(x_value), map_y(recall)) for _, x_value, recall in panel_points[method]]
                canvas.stroke_rgb(color)
                canvas.line_width(1.0 if method != "EPQ" else 1.5)
                canvas.polyline(points)
                for x, y in points:
                    canvas.marker(x, y, shapes.get(method, "circle"), 2.4, color)

    write_pdf(output, width, height, canvas.stream())


def write_manifest(
    output: Path,
    methods: list[str],
    source: str,
    csv_dir: Path,
    baseline_log_dir: Path,
    dpopq_log_dir: Path,
    rqlsq_log_dir: Path,
    epq_log_dir: Path,
    vaq_log_dir: Path,
    figure_output: Path,
    table_output: Path,
    refine_table_output: Path,
    epq_time_table_output: Path,
) -> None:
    def display_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve()))
        except ValueError:
            return str(path)

    lines = [
        "# IVF Paper Assets",
        "",
        f"source: {source}",
        f"methods: {', '.join(methods)}",
    ]
    if source == "csv":
        lines.append(f"csv_dir: {display_path(csv_dir)}")
    else:
        lines.extend(
            [
                f"baseline_log_dir: {display_path(baseline_log_dir)}",
                f"dpopq_log_dir: {display_path(dpopq_log_dir)}",
                f"rqlsq_log_dir: {display_path(rqlsq_log_dir)}",
                f"epq_log_dir: {display_path(epq_log_dir)}",
                f"vaq_log_dir: {display_path(vaq_log_dir)}",
            ]
        )
    lines.extend(
        [
            f"figure: {display_path(figure_output)}",
            f"table: {display_path(table_output)}",
            f"refine_table: {display_path(refine_table_output)}",
            f"epq_time_table: {display_path(epq_time_table_output)}",
            "",
            "RaBitQ is excluded from the default output because its actual footprint is not a matched 64/128-bit budget.",
            "Use --plot-x qps with a different --figure-output to regenerate the recall-QPS diagnostic variant.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper IVF recall-nprobe figure and compact QPS table."
    )
    parser.add_argument("--source", choices=("logs", "csv"), default="csv")
    parser.add_argument("--csv-dir", type=Path, default=ROOT)
    parser.add_argument("--baseline-log-dir", type=Path, default=DEFAULT_BASELINE_LOG_DIR)
    parser.add_argument("--dpopq-log-dir", type=Path, default=DEFAULT_DPOPQ_LOG_DIR)
    parser.add_argument("--rqlsq-log-dir", type=Path, default=DEFAULT_RQLSQ_LOG_DIR)
    parser.add_argument("--epq-log-dir", type=Path, default=DEFAULT_EPQ_LOG_DIR)
    parser.add_argument("--vaq-log-dir", type=Path, default=DEFAULT_VAQ_LOG_DIR)
    parser.add_argument("--paper-dir", type=Path, default=PAPER_ROOT)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--bits", default=",".join(str(bit) for bit in DEFAULT_BITS))
    parser.add_argument(
        "--metric",
        choices=("recall_1", "recall_10", "recall_100", "recall_1000", "overlap_1000"),
        default="recall_10",
    )
    parser.add_argument("--plot-x", choices=("qps", "nprobe"), default="nprobe")
    parser.add_argument("--figure-output", type=Path, default=None)
    parser.add_argument("--table-output", type=Path, default=None)
    parser.add_argument("--refine-table-output", type=Path, default=None)
    parser.add_argument("--epq-time-table-output", type=Path, default=None)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--no-refine-table", action="store_true")
    parser.add_argument("--no-epq-time-table", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_str_list(args.methods)
    datasets = parse_str_list(args.datasets)
    bits_list = parse_int_list(args.bits)
    unknown = [method for method in methods if method not in METHOD_FILES]
    if unknown:
        raise SystemExit(f"unknown methods: {', '.join(unknown)}")
    if "RaBitQ" in methods:
        print(
            "warning: RaBitQ actual footprint is not matched to 64/128-bit budgets",
            file=sys.stderr,
        )

    default_figure_name = "ivf-recall-nprobe.pdf" if args.plot_x == "nprobe" else "ivf-recall-qps.pdf"
    figure_output = args.figure_output or args.paper_dir / "generated" / default_figure_name
    table_output = args.table_output or args.paper_dir / "generated" / "ivf_main_table.tex"
    refine_table_output = (
        args.refine_table_output
        or args.paper_dir / "generated" / "ivf_refine_table.tex"
    )
    epq_time_table_output = (
        args.epq_time_table_output
        or args.paper_dir / "generated" / "ivf_epq_time_table.tex"
    )
    manifest_output = args.manifest_output or args.paper_dir / "generated" / "ivf_asset_manifest.md"

    if args.source == "logs":
        rows = load_rows_from_logs(
            methods,
            args.baseline_log_dir,
            args.dpopq_log_dir,
            args.rqlsq_log_dir,
            args.epq_log_dir,
            args.vaq_log_dir,
        )
    else:
        rows = load_rows_from_csvs(args.csv_dir, methods)
    cases = validate_coverage(rows, methods, datasets, bits_list)

    if not args.no_table:
        table_output.parent.mkdir(parents=True, exist_ok=True)
        table_output.write_text(render_ivf_main_table(rows, methods, cases))
        print(f"wrote {table_output}")
    if not args.no_refine_table:
        missing_refine_methods = [
            method for method in REFINE_TABLE_METHODS if method not in methods
        ]
        if missing_refine_methods:
            raise RuntimeError(
                "cannot render IVF RefineFlat table without methods: "
                + ", ".join(missing_refine_methods)
            )
        refine_table_output.parent.mkdir(parents=True, exist_ok=True)
        refine_table_output.write_text(render_ivf_refine_table(rows, cases))
        print(f"wrote {refine_table_output}")
    if not args.no_epq_time_table:
        if "EPQ" not in methods:
            raise RuntimeError("cannot render IVF EPQ time table without EPQ")
        epq_time_table_output.parent.mkdir(parents=True, exist_ok=True)
        epq_time_table_output.write_text(render_ivf_epq_time_table(rows, cases))
        print(f"wrote {epq_time_table_output}")
    if not args.no_figure:
        draw_ivf_figure(rows, methods, cases, args.metric, args.plot_x, figure_output)
        print(f"wrote {figure_output}")
    write_manifest(
        manifest_output,
        methods,
        args.source,
        args.csv_dir,
        args.baseline_log_dir,
        args.dpopq_log_dir,
        args.rqlsq_log_dir,
        args.epq_log_dir,
        args.vaq_log_dir,
        figure_output,
        table_output,
        refine_table_output,
        epq_time_table_output,
    )
    print(f"wrote {manifest_output}")


if __name__ == "__main__":
    main()
