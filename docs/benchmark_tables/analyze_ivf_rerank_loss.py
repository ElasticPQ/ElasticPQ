#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = os.path.dirname(os.path.abspath(__file__))
EPQ_NOREFINE_GLOB = (
    ROOT
    + "/logs/joint_topk1000*_epq_perfvalidated_norefine_20260627/joint_*.json"
)
EPQ_REFINE_GLOB = (
    ROOT
    + "/logs/joint_topk1000*_epq_perfvalidated_refine_20260627/joint_*.json"
)


@dataclass(frozen=True)
class CaseKey:
    dataset: str
    bits: int
    nprobe: int


@dataclass
class MethodMetrics:
    method: str
    dataset: str
    bits: int
    nprobe: int
    top1: float
    top10: float
    top100: float
    top1000: float
    upper1: float
    upper10: float
    upper100: float
    upper1000: float
    candidate_hit_rate: float
    avg_candidates: float

    def keep_ratio(self, k: int) -> float:
        if k == 1:
            num, den = self.top1, self.upper1
        elif k == 10:
            num, den = self.top10, self.upper10
        elif k == 100:
            num, den = self.top100, self.upper100
        elif k == 1000:
            num, den = self.top1000, self.upper1000
        else:
            raise ValueError(f"unsupported k={k}")
        if den <= 0.0:
            return float("nan")
        return num / den

    def loss(self, k: int) -> float:
        if k == 1:
            return self.upper1 - self.top1
        if k == 10:
            return self.upper10 - self.top10
        if k == 100:
            return self.upper100 - self.top100
        if k == 1000:
            return self.upper1000 - self.top1000
        raise ValueError(f"unsupported k={k}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze IVF rerank loss by comparing no-refine vs refine runs on the "
            "same coarse candidate set."
        )
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--bits", type=int, default=None)
    parser.add_argument("--nprobe", type=int, default=None)
    parser.add_argument(
        "--methods",
        default="epq,pq,bapq,opq",
        help="comma-separated subset of: epq,pq,bapq,opq",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="max number of cases to print when filters are not specified",
    )
    return parser.parse_args()


def parse_float(text: str) -> float:
    text = text.strip()
    if not text or text.upper() == "N/A":
        return float("nan")
    return float(text)


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def is_refine_row(row: Dict[str, str]) -> bool:
    notes = row.get("notes", "")
    return "refine=IndexRefineFlat" in notes


def case_matches(key: CaseKey, args: argparse.Namespace) -> bool:
    if args.dataset is not None and key.dataset != args.dataset:
        return False
    if args.bits is not None and key.bits != args.bits:
        return False
    if args.nprobe is not None and key.nprobe != args.nprobe:
        return False
    return True


def build_metrics_from_csv(
    method: str,
    norefine: Dict[str, str],
    refine: Dict[str, str],
) -> MethodMetrics:
    return MethodMetrics(
        method=method,
        dataset=norefine["dataset"],
        bits=int(norefine["budget_b"]),
        nprobe=int(norefine["nprobe"]),
        top1=parse_float(norefine["recall_1"]),
        top10=parse_float(norefine["recall_10"]),
        top100=parse_float(norefine["recall_100"]),
        top1000=parse_float(norefine["recall_1000"]),
        upper1=parse_float(refine["recall_1"]),
        upper10=parse_float(refine["recall_10"]),
        upper100=parse_float(refine["recall_100"]),
        upper1000=parse_float(refine["recall_1000"]),
        candidate_hit_rate=parse_float(norefine["candidate_hit_rate"]),
        avg_candidates=parse_float(norefine["avg_candidates_per_q"]),
    )


def load_non_epq_method(method: str) -> Dict[CaseKey, MethodMetrics]:
    rows = load_csv_rows(os.path.join(ROOT, f"IVF-{method.upper()}.csv"))
    norefine_rows: Dict[CaseKey, Dict[str, str]] = {}
    refine_rows: Dict[CaseKey, Dict[str, str]] = {}
    for row in rows:
        key = CaseKey(
            dataset=row["dataset"],
            bits=int(row["budget_b"]),
            nprobe=int(row["nprobe"]),
        )
        if is_refine_row(row):
            refine_rows[key] = row
        else:
            norefine_rows[key] = row
    out: Dict[CaseKey, MethodMetrics] = {}
    for key, noref in norefine_rows.items():
        ref = refine_rows.get(key)
        if ref is None:
            continue
        out[key] = build_metrics_from_csv(method, noref, ref)
    return out


def metric_from_epq_json(
    method: str,
    norefine: Dict,
    refine: Dict,
) -> MethodMetrics:
    return MethodMetrics(
        method=method,
        dataset=norefine["dataset"],
        bits=int(norefine["nominal_bits"]),
        nprobe=int(norefine["nprobe"]),
        top1=float(norefine["recall1"]),
        top10=float(norefine["recall10"]),
        top100=float(norefine["recall100"]),
        top1000=float(norefine["recall1000"]),
        upper1=float(refine["recall1"]),
        upper10=float(refine["recall10"]),
        upper100=float(refine["recall100"]),
        upper1000=float(refine["recall1000"]),
        candidate_hit_rate=float(norefine["coarse"]["candidate_hit_rate"]),
        avg_candidates=float(norefine["coarse"]["avg_candidates"]),
    )


def load_epq() -> Dict[CaseKey, MethodMetrics]:
    norefine_rows: Dict[CaseKey, Dict] = {}
    refine_rows: Dict[CaseKey, Dict] = {}

    for path in glob.glob(EPQ_NOREFINE_GLOB):
        with open(path) as handle:
            row = json.load(handle)
        key = CaseKey(
            dataset=row["dataset"],
            bits=int(row["nominal_bits"]),
            nprobe=int(row["nprobe"]),
        )
        norefine_rows[key] = row

    for path in glob.glob(EPQ_REFINE_GLOB):
        with open(path) as handle:
            row = json.load(handle)
        key = CaseKey(
            dataset=row["dataset"],
            bits=int(row["nominal_bits"]),
            nprobe=int(row["nprobe"]),
        )
        refine_rows[key] = row

    out: Dict[CaseKey, MethodMetrics] = {}
    for key, noref in norefine_rows.items():
        ref = refine_rows.get(key)
        if ref is None:
            continue
        out[key] = metric_from_epq_json("epq", noref, ref)
    return out


def fmt(x: float) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.4f}"


def print_case_table(key: CaseKey, metrics: List[MethodMetrics]) -> None:
    print(
        f"\nCASE {key.dataset} {key.bits}b nprobe={key.nprobe} "
        f"(avg_candidates={fmt(metrics[0].avg_candidates)}, "
        f"candidate_upper={fmt(metrics[0].candidate_hit_rate)})"
    )
    print(
        "method   "
        "top1   upper1 keep1  loss1   "
        "top10  upper10 keep10 loss10   "
        "top100 upper100 keep100 loss100"
    )
    for item in metrics:
        print(
            f"{item.method:<6} "
            f"{fmt(item.top1):>6} {fmt(item.upper1):>7} {fmt(item.keep_ratio(1)):>5} {fmt(item.loss(1)):>6}   "
            f"{fmt(item.top10):>6} {fmt(item.upper10):>7} {fmt(item.keep_ratio(10)):>6} {fmt(item.loss(10)):>6}   "
            f"{fmt(item.top100):>6} {fmt(item.upper100):>8} {fmt(item.keep_ratio(100)):>7} {fmt(item.loss(100)):>7}"
        )


def print_summary(metrics_by_method: Dict[str, Dict[CaseKey, MethodMetrics]], args: argparse.Namespace) -> None:
    methods = [m for m in args.methods.split(",") if m]
    common_keys: Optional[set] = None
    for method in methods:
        keys = {key for key in metrics_by_method[method] if case_matches(key, args)}
        common_keys = keys if common_keys is None else common_keys & keys
    if not common_keys:
        raise SystemExit("no matching cases")

    ordered_keys = sorted(common_keys, key=lambda x: (x.dataset, x.bits, x.nprobe))
    if args.dataset is None and args.bits is None and args.nprobe is None:
        ordered_keys = ordered_keys[: args.top]

    for key in ordered_keys:
        rows = [metrics_by_method[method][key] for method in methods]
        rows.sort(key=lambda item: item.keep_ratio(10), reverse=True)
        print_case_table(key, rows)


def main() -> None:
    args = parse_args()
    methods = [m for m in args.methods.split(",") if m]
    valid = {"epq", "pq", "bapq", "opq"}
    unknown = [m for m in methods if m not in valid]
    if unknown:
        raise SystemExit(f"unknown methods: {', '.join(unknown)}")

    metrics_by_method: Dict[str, Dict[CaseKey, MethodMetrics]] = {}
    for method in methods:
        if method == "epq":
            metrics_by_method[method] = load_epq()
        else:
            metrics_by_method[method] = load_non_epq_method(method)

    print_summary(metrics_by_method, args)


if __name__ == "__main__":
    main()
