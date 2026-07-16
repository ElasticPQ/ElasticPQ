#!/usr/bin/env python3
"""Export SIFT1M 128-bit random-seed robustness tables from benchmark logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_OUTPUT = ROOT / "sift128_seed_robustness_table.tex"
DEFAULT_LOGS = [
    ROOT / "tmp_sift128_ablation_odd_20260711T125320Z" / "full_arepq.log",
    ROOT / "tmp_sift128_seed_robustness_20260711T141328Z" / "seed_456.log",
    ROOT / "tmp_sift128_seed_robustness_20260711T141328Z" / "seed_789.log",
]


@dataclass
class SeedResult:
    seed: int
    m: int
    recall_1: float
    recall_10: float
    recall_100: float
    overlap_1000: float
    recon_error: float
    train_s: float
    add_s: float
    search_s: float
    qps: float
    exit_status: int
    log_path: Path


def parse_json_line(text: str, prefix: str) -> dict:
    match = re.search(rf"^{re.escape(prefix)}\s+(.+)$", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing {prefix}")
    return json.loads(match.group(1))


def parse_float(text: str, pattern: str, name: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing {name}")
    return float(match.group(1))


def parse_int(text: str, pattern: str, name: str) -> int:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing {name}")
    return int(match.group(1))


def parse_seed(text: str, method: dict) -> int:
    match = re.search(r"^seed=(\d+)$", text, re.MULTILINE)
    if match:
        return int(match.group(1))
    main = method.get("main", method)
    return int(main.get("builder", {}).get("seed", -1))


def validate_common(text: str, method: dict, row: SeedResult) -> None:
    errors: list[str] = []
    if row.exit_status != 0:
        errors.append(f"exit_status={row.exit_status}")
    if not re.search(r"^dataset=sift1M\b.*\bbits=128\b.*\bmaxtrain=0\b", text, re.MULTILINE):
        errors.append("dataset/bits/maxtrain header mismatch")
    if method.get("family") != "arepq":
        errors.append(f"family={method.get('family')}")
    main = method.get("main", {})
    if main.get("builder", {}).get("type") != "refined":
        errors.append(f"builder={main.get('builder', {}).get('type')}")
    if not main.get("use_uneven_transform", False):
        errors.append("use_uneven_transform=false")
    if int(method.get("tail_bits", 0)) <= 0:
        errors.append("missing residual tail")
    if int(method.get("total_bits", 0)) != 128:
        errors.append(f"total_bits={method.get('total_bits')}")
    if errors:
        raise RuntimeError(f"{row.log_path}: " + "; ".join(errors))


def parse_log(path: Path) -> SeedResult:
    text = path.read_text(errors="replace")
    method = parse_json_line(text, "meta.method")
    component_count = parse_int(text, r"^\s*M:\s*(\d+)\s*$", "M")
    product_groups = component_count - int(method.get("tail_stages", 0))
    recall_match = re.search(
        r"^\s*recall@1:\s*([0-9.]+)\s+recall@10:\s*([0-9.]+)"
        r"\s+recall@100:\s*([0-9.]+)\s+recall@1000:\s*([0-9.]+)\s*$",
        text,
        re.MULTILINE,
    )
    if not recall_match:
        raise RuntimeError(f"{path}: missing recall line")
    row = SeedResult(
        seed=parse_seed(text, method),
        m=product_groups,
        recall_1=float(recall_match.group(1)),
        recall_10=float(recall_match.group(2)),
        recall_100=float(recall_match.group(3)),
        overlap_1000=parse_float(
            text,
            r"^\s*overlap@1000\(gt=1000\):\s*([0-9.]+)\s*$",
            "overlap@1000",
        ),
        recon_error=parse_float(
            text,
            r"^\s*reconstruction error \(sample\):\s*([0-9.]+)\s*$",
            "reconstruction error",
        ),
        train_s=parse_float(text, r"^\s*training total:\s*([0-9.]+)\s*s$", "train"),
        add_s=parse_float(text, r"^\s*add/encode time:\s*([0-9.]+)\s*s$", "add"),
        search_s=parse_float(text, r"^\s*search time:\s*([0-9.]+)\s*s$", "search"),
        qps=parse_float(text, r"^\s*QPS:\s*([0-9.]+)\s*$", "QPS"),
        exit_status=parse_int(text, r"^\s*Exit status:\s*(\d+)\s*$", "exit status"),
        log_path=path,
    )
    validate_common(text, method, row)
    return row


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def fmt_recall(value: float) -> str:
    return f"{value:.4f}"


def fmt_j(value: float) -> str:
    return f"{value:.0f}"


def fmt_mean_std(mean: float, std: float, digits: int) -> str:
    return rf"${mean:.{digits}f}{{\pm}}{std:.{digits}f}$"


def fmt_j_mean_std(mean: float, std: float) -> str:
    return rf"${mean:.0f}{{\pm}}{std:.0f}$"


def repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def write_csv(rows: list[SeedResult], output: Path) -> None:
    fields = [
        "seed",
        "M",
        "recall_1",
        "recall_10",
        "recall_100",
        "overlap_1000",
        "reconstruction_error",
        "train_s",
        "add_s",
        "search_s",
        "qps",
        "exit_status",
        "log_path",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "seed": row.seed,
                    "M": row.m,
                    "recall_1": row.recall_1,
                    "recall_10": row.recall_10,
                    "recall_100": row.recall_100,
                    "overlap_1000": row.overlap_1000,
                    "reconstruction_error": row.recon_error,
                    "train_s": row.train_s,
                    "add_s": row.add_s,
                    "search_s": row.search_s,
                    "qps": row.qps,
                    "exit_status": row.exit_status,
                    "log_path": repo_relative(row.log_path),
                }
            )


def write_latex(rows: list[SeedResult], output: Path) -> None:
    rows = sorted(rows, key=lambda item: item.seed)
    mean_r1, std_r1 = mean_std([row.recall_1 for row in rows])
    mean_r10, std_r10 = mean_std([row.recall_10 for row in rows])
    mean_r100, std_r100 = mean_std([row.recall_100 for row in rows])
    mean_overlap, std_overlap = mean_std([row.overlap_1000 for row in rows])
    mean_j, std_j = mean_std([row.recon_error for row in rows])
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\caption{Random-seed robustness of full EPQ on SIFT1M at 128 bits.}",
        r"\label{tab:seed-robustness}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}c c c c c c c@{}}",
        r"\toprule",
        r"Seed & $M$ & Recall@1 & Recall@10 & Recall@100 & Overlap@1k & $J$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.seed} & {row.m} & {fmt_recall(row.recall_1)} & "
            f"{fmt_recall(row.recall_10)} & {fmt_recall(row.recall_100)} & "
            f"{fmt_recall(row.overlap_1000)} & {fmt_j(row.recon_error)} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            "Mean $\\pm$ Std & -- & "
            f"{fmt_mean_std(mean_r1, std_r1, 4)} & "
            f"{fmt_mean_std(mean_r10, std_r10, 4)} & "
            f"{fmt_mean_std(mean_r100, std_r100, 4)} & "
            f"{fmt_mean_std(mean_overlap, std_overlap, 4)} & "
            f"{fmt_j_mean_std(mean_j, std_j)} \\\\",
            r"\bottomrule",
            r"\end{tabular*}",
            r"\end{table}",
            "",
        ]
    )
    output.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV path. Defaults to output with .csv suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_paths = args.log or DEFAULT_LOGS
    rows = [parse_log(path) for path in log_paths]
    if sorted(row.seed for row in rows) != [123, 456, 789]:
        raise RuntimeError(f"unexpected seed set: {[row.seed for row in rows]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_latex(rows, args.output)
    csv_output = args.csv_output or args.output.with_suffix(".csv")
    write_csv(sorted(rows, key=lambda item: item.seed), csv_output)
    print(f"wrote {args.output}")
    print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
