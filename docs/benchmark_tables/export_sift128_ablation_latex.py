#!/usr/bin/env python3
"""Export a LaTeX ablation table from SIFT1M 128-bit flat benchmark logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_LOG_DIR = ROOT / "tmp_sift128_ablation_odd_20260711T125320Z"
DEFAULT_OUTPUT = ROOT / "sift128_ablation_table.tex"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    title: str
    log_name: str
    expected_family: str
    expected_builder: Optional[str]
    expected_uneven: bool
    expected_tail: bool


@dataclass
class CaseResult:
    spec: CaseSpec
    dataset: str
    bits: int
    train_rows: int
    base_rows: int
    query_rows: int
    maxtrain: int
    target: str
    config_path: str
    method: dict
    m: int
    structure_s: float
    prep_s: float
    codebook_s: float
    train_s: float
    add_s: float
    search_s: float
    qps: float
    recall_1: float
    recall_10: float
    recall_100: float
    recall_1000: float
    overlap_1000: float
    recon_error: float
    index_mib: Optional[float]
    exit_status: int


CASE_SPECS = [
    CaseSpec(
        key="full_arepq",
        title="Full EPQ",
        log_name="full_arepq.log",
        expected_family="arepq",
        expected_builder="refined",
        expected_uneven=True,
        expected_tail=True,
    ),
    CaseSpec(
        key="no_structure",
        title=r"w/o Structure Learning",
        log_name="no_structure.log",
        expected_family="arepq",
        expected_builder="balanced",
        expected_uneven=True,
        expected_tail=True,
    ),
    CaseSpec(
        key="no_uneven",
        title=r"w/o UnevenOPQ",
        log_name="no_uneven_fresh.log",
        expected_family="arepq",
        expected_builder="refined",
        expected_uneven=False,
        expected_tail=True,
    ),
    CaseSpec(
        key="no_residual_tail",
        title=r"w/o Residual Tail",
        log_name="no_residual_tail.log",
        expected_family="epq",
        expected_builder="refined",
        expected_uneven=True,
        expected_tail=False,
    ),
]


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


def parse_json_line(text: str, prefix: str) -> dict:
    pattern = re.compile(rf"^{re.escape(prefix)}\s+(.+)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"missing {prefix}")
    return json.loads(match.group(1))


def parse_required_float(text: str, pattern: str, name: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing {name}")
    return float(match.group(1))


def parse_optional_float(text: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_required_int(text: str, pattern: str, name: str) -> int:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing {name}")
    return int(match.group(1))


def parse_dataset_header(text: str) -> tuple[str, int, int, int, int, int]:
    match = re.search(
        r"^dataset=(\S+)\s+d=\d+\s+nb=(\d+)\s+nq=(\d+)\s+nt=(\d+)"
        r".*?\sbits=(\d+).*?\smaxtrain=(\d+)\b",
        text,
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError("missing dataset header")
    return (
        match.group(1),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
        int(match.group(5)),
        int(match.group(6)),
    )


def main_meta(method: dict) -> dict:
    return method.get("main", method)


def builder_type(method: dict) -> str:
    return str(main_meta(method).get("builder", {}).get("type", ""))


def use_uneven(method: dict) -> bool:
    return bool(main_meta(method).get("use_uneven_transform", False))


def has_tail(method: dict) -> bool:
    return method.get("family") == "arepq" and int(method.get("tail_bits", 0)) > 0


def validate_case(result: CaseResult) -> None:
    errors: list[str] = []
    if result.exit_status != 0:
        errors.append(f"exit_status={result.exit_status}")
    if result.dataset != "sift1M" or result.bits != 128:
        errors.append(f"unexpected dataset/bits: {result.dataset} {result.bits}")
    if result.maxtrain != 0:
        errors.append(f"maxtrain={result.maxtrain}, expected 0")
    if result.train_rows != 100000:
        errors.append(f"train_rows={result.train_rows}, expected 100000")
    if result.method.get("family") != result.spec.expected_family:
        errors.append(
            f"family={result.method.get('family')}, "
            f"expected {result.spec.expected_family}"
        )
    actual_builder = builder_type(result.method)
    if result.spec.expected_builder and actual_builder != result.spec.expected_builder:
        errors.append(
            f"builder={actual_builder}, expected {result.spec.expected_builder}"
        )
    if use_uneven(result.method) != result.spec.expected_uneven:
        errors.append(
            f"use_uneven_transform={use_uneven(result.method)}, "
            f"expected {result.spec.expected_uneven}"
        )
    if has_tail(result.method) != result.spec.expected_tail:
        errors.append(
            f"residual_tail={has_tail(result.method)}, "
            f"expected {result.spec.expected_tail}"
        )
    if errors:
        joined = "; ".join(errors)
        raise RuntimeError(f"{result.spec.key}: validation failed: {joined}")


def parse_case(log_dir: Path, spec: CaseSpec) -> CaseResult:
    log_path = log_dir / spec.log_name
    if not log_path.exists():
        raise RuntimeError(f"missing log: {log_path}")
    text = log_path.read_text(errors="replace")
    dataset, base_rows, query_rows, train_rows, bits, maxtrain = parse_dataset_header(text)
    run = parse_json_line(text, "meta.run")
    method = parse_json_line(text, "meta.method")
    component_count = parse_required_int(text, r"^\s*M:\s*(\d+)\s*$", "M")
    m = component_count - int(method.get("tail_stages", 0)) if has_tail(method) else component_count
    recall = re.search(
        r"^\s*recall@1:\s*([0-9.]+)\s+recall@10:\s*([0-9.]+)"
        r"\s+recall@100:\s*([0-9.]+)\s+recall@1000:\s*([0-9.]+)\s*$",
        text,
        re.MULTILINE,
    )
    if not recall:
        raise RuntimeError(f"{spec.key}: missing recall line")
    result = CaseResult(
        spec=spec,
        dataset=dataset,
        bits=bits,
        train_rows=train_rows,
        base_rows=base_rows,
        query_rows=query_rows,
        maxtrain=maxtrain,
        target=str(run.get("targets", [""])[0]),
        config_path=str(run.get("config_path", "")),
        method=method,
        m=m,
        structure_s=parse_required_float(
            text, r"^\s*structure time:\s*([0-9.]+)\s*s$", "structure time"
        ),
        prep_s=parse_required_float(
            text, r"^\s*preparation time:\s*([0-9.]+)\s*s$", "preparation time"
        ),
        codebook_s=parse_required_float(
            text, r"^\s*codebook time:\s*([0-9.]+)\s*s$", "codebook time"
        ),
        train_s=parse_required_float(
            text, r"^\s*training total:\s*([0-9.]+)\s*s$", "training total"
        ),
        add_s=parse_required_float(
            text, r"^\s*add/encode time:\s*([0-9.]+)\s*s$", "add time"
        ),
        search_s=parse_required_float(
            text, r"^\s*search time:\s*([0-9.]+)\s*s$", "search time"
        ),
        qps=parse_required_float(text, r"^\s*QPS:\s*([0-9.]+)\s*$", "QPS"),
        recall_1=float(recall.group(1)),
        recall_10=float(recall.group(2)),
        recall_100=float(recall.group(3)),
        recall_1000=float(recall.group(4)),
        overlap_1000=parse_required_float(
            text,
            r"^\s*overlap@1000\(gt=1000\):\s*([0-9.]+)\s*$",
            "overlap@1000",
        ),
        recon_error=parse_required_float(
            text,
            r"^\s*reconstruction error \(sample\):\s*([0-9.]+)\s*$",
            "reconstruction error",
        ),
        index_mib=parse_optional_float(
            text, r"^\s*serialized index size:\s*([0-9.]+)\s*MiB$"
        ),
        exit_status=parse_required_int(
            text, r"^\s*Exit status:\s*(\d+)\s*$", "exit status"
        ),
    )
    validate_case(result)
    return result


def fmt_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def fmt_time(value: float) -> str:
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def fmt_recon(value: float) -> str:
    return f"{value:.0f}"


def fmt_qps(value: float) -> str:
    return f"{value:.0f}" if value >= 100 else f"{value:.1f}"


@dataclass(frozen=True)
class Column:
    key: str
    header: str
    getter: Callable[[CaseResult], float | int | str]
    formatter: Callable[[float | int | str], str]
    best: Optional[str] = None


def build_columns() -> list[Column]:
    return [
        Column("variant", "Variant", lambda r: r.spec.title, str, None),
        Column("m", r"$M$", lambda r: r.m, lambda x: str(int(x)), None),
        Column("r1", "R@1", lambda r: r.recall_1, lambda x: fmt_float(float(x), 4), "max"),
        Column("r10", "R@10", lambda r: r.recall_10, lambda x: fmt_float(float(x), 4), "max"),
        Column("r100", "R@100", lambda r: r.recall_100, lambda x: fmt_float(float(x), 4), "max"),
        Column("overlap", "Overlap@1k", lambda r: r.overlap_1000, lambda x: fmt_float(float(x), 4), "max"),
        Column("j", r"$J$", lambda r: r.recon_error, lambda x: fmt_recon(float(x)), "min"),
        Column("train", "Train (s)", lambda r: r.train_s, lambda x: fmt_time(float(x)), None),
        Column("add", "Add (s)", lambda r: r.add_s, lambda x: fmt_time(float(x)), None),
        Column("search", "Search (s)", lambda r: r.search_s, lambda x: fmt_time(float(x)), None),
        Column("qps", "QPS", lambda r: r.qps, lambda x: fmt_qps(float(x)), None),
    ]


def best_values(rows: list[CaseResult], columns: list[Column]) -> dict[str, float]:
    out: dict[str, float] = {}
    for column in columns:
        if column.best is None:
            continue
        values = [float(column.getter(row)) for row in rows]
        out[column.key] = max(values) if column.best == "max" else min(values)
    return out


def maybe_bold(value: str, raw: float | int | str, column: Column, best: dict[str, float]) -> str:
    if column.key not in best:
        return value
    if math.isclose(float(raw), best[column.key], rel_tol=1e-12, abs_tol=1e-12):
        return rf"\textbf{{{value}}}"
    return value


def repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def write_csv(rows: list[CaseResult], output: Path) -> None:
    fields = [
        "variant",
        "target",
        "config_path",
        "builder",
        "use_uneven_transform",
        "residual_tail",
        "M",
        "train_s",
        "structure_s",
        "preparation_s",
        "codebook_s",
        "add_s",
        "search_s",
        "qps",
        "recall_1",
        "recall_10",
        "recall_100",
        "recall_1000",
        "overlap_1000",
        "reconstruction_error",
        "index_mib",
        "train_rows",
        "base_rows",
        "query_rows",
        "maxtrain",
        "exit_status",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant": row.spec.key,
                    "target": row.target,
                    "config_path": repo_relative(row.config_path),
                    "builder": builder_type(row.method),
                    "use_uneven_transform": use_uneven(row.method),
                    "residual_tail": has_tail(row.method),
                    "M": row.m,
                    "train_s": row.train_s,
                    "structure_s": row.structure_s,
                    "preparation_s": row.prep_s,
                    "codebook_s": row.codebook_s,
                    "add_s": row.add_s,
                    "search_s": row.search_s,
                    "qps": row.qps,
                    "recall_1": row.recall_1,
                    "recall_10": row.recall_10,
                    "recall_100": row.recall_100,
                    "recall_1000": row.recall_1000,
                    "overlap_1000": row.overlap_1000,
                    "reconstruction_error": row.recon_error,
                    "index_mib": row.index_mib,
                    "train_rows": row.train_rows,
                    "base_rows": row.base_rows,
                    "query_rows": row.query_rows,
                    "maxtrain": row.maxtrain,
                    "exit_status": row.exit_status,
                }
            )


def write_latex(rows: list[CaseResult], output: Path, caption: str, label: str) -> None:
    columns = build_columns()
    best = best_values(rows, columns)
    align = "l" + "r" * (len(columns) - 1)
    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(column.header for column in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells: list[str] = []
        for column in columns:
            raw = column.getter(row)
            rendered = column.formatter(raw)
            if column.key == "variant":
                rendered = latex_escape(rendered)
            else:
                rendered = maybe_bold(rendered, raw, column, best)
            cells.append(rendered)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    output.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV summary path. Defaults to output with .csv suffix.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Ablation study on SIFT1M at 128 bits under the flat ADC protocol."
        ),
    )
    parser.add_argument("--label", default="tab:sift128-ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [parse_case(args.log_dir, spec) for spec in CASE_SPECS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_latex(rows, args.output, args.caption, args.label)
    csv_output = args.csv_output or args.output.with_suffix(".csv")
    write_csv(rows, csv_output)
    print(f"wrote {args.output}")
    print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
