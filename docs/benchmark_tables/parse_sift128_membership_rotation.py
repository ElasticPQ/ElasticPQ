#!/usr/bin/env python3
"""Parse and summarize the SIFT1M 128b membership/rotation experiment."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


FLOAT = r"[-+0-9.eE]+"
METRICS = ("reconstruction_error", "recall1", "recall10", "recall100", "overlap1000")


def search_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing pattern: {pattern}")
    return float(match.group(1))


def search_int(pattern: str, text: str) -> int:
    return int(search_float(pattern, text))


def parse_header(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.*)$", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing header key: {key}")
    return match.group(1).strip()


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text()
    if not re.search(r"^\s*Exit status: 0$", text, re.MULTILINE):
        raise RuntimeError(f"non-zero or missing exit status: {path}")
    recall = re.search(
        rf"recall@1: ({FLOAT}) recall@10: ({FLOAT}) "
        rf"recall@100: ({FLOAT}) recall@1000: ({FLOAT})",
        text,
    )
    if not recall:
        raise RuntimeError(f"missing recall line: {path}")
    method_line = re.search(r"^meta\.method (\{.*\})$", text, re.MULTILINE)
    if not method_line:
        raise RuntimeError(f"missing method metadata: {path}")
    method = json.loads(method_line.group(1))
    main = method["main"]
    structure_path = Path(parse_header(text, "structure_path"))
    structure = json.loads(structure_path.read_text())
    sizes = [len(group["dims"]) for group in structure["groups"]]
    bits = [group["nbits"] for group in structure["groups"]]
    return {
        "case_id": parse_header(text, "case_id"),
        "protocol": parse_header(text, "protocol"),
        "membership": parse_header(text, "membership"),
        "membership_seed": int(parse_header(text, "membership_seed")),
        "init_mode": parse_header(text, "init_mode"),
        "init_seed": int(parse_header(text, "init_seed")),
        "structure_path": str(structure_path.relative_to(REPO_ROOT)),
        "group_count": len(structure["groups"]),
        "group_sizes": ",".join(map(str, sizes)),
        "group_bits": ",".join(map(str, bits)),
        "main_bits": structure["total_bits"],
        "training_seconds": search_float(rf"training total:\s+({FLOAT}) s", text),
        "add_seconds": search_float(rf"add/encode time:\s+({FLOAT}) s", text),
        "search_seconds": search_float(rf"search time:\s+({FLOAT}) s", text),
        "qps": search_float(rf"QPS:\s+({FLOAT})", text),
        "reconstruction_error": search_float(
            rf"reconstruction error \(sample\):\s+({FLOAT})", text
        ),
        "recall1": float(recall.group(1)),
        "recall10": float(recall.group(2)),
        "recall100": float(recall.group(3)),
        "recall1000": float(recall.group(4)),
        "overlap1000": search_float(rf"overlap@1000\(gt=1000\):\s+({FLOAT})", text),
        "max_rss_kib": search_int(r"Maximum resident set size \(kbytes\):\s+(\d+)", text),
        "transform_init_mode_meta": main["transform_init_mode"],
        "transform_init_seed_meta": main["transform_init_seed"],
        "transform_init_orthogonality_error": main[
            "transform_init_orthogonality_error"
        ],
        "tail_alt_final_mse": method["tail_alt_final_mse"],
        "exit_status": 0,
        "log_path": str(path.resolve().relative_to(REPO_ROOT)),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def bootstrap_mean_ci(values: list[float], seed: int = 20260714) -> tuple[float, float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(20000):
        samples.append(statistics.mean(rng.choice(values) for _ in values))
    samples.sort()
    return samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]


def exact_sign_flip_paired_p(diffs: list[float]) -> float:
    observed = abs(statistics.mean(diffs))
    count = 0
    total = 1 << len(diffs)
    for mask in range(total):
        permuted = [
            value if (mask >> index) & 1 else -value
            for index, value in enumerate(diffs)
        ]
        if abs(statistics.mean(permuted)) >= observed - 1e-15:
            count += 1
    return count / total


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def group_by_seed(rows: list[dict[str, object]]) -> dict[int, dict[str, dict[str, object]]]:
    grouped: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[int(row["init_seed"])][str(row["membership"])] = row
    return grouped


def summarize(matrix_dir: Path, rows: list[dict[str, object]]) -> tuple[dict, str]:
    protocols = defaultdict(list)
    for row in rows:
        protocols[str(row["protocol"])].append(row)

    signatures = {
        (row["group_count"], row["group_sizes"], row["group_bits"], row["main_bits"])
        for row in rows
    }
    if len(signatures) != 1:
        raise RuntimeError(f"architecture mismatch: {signatures}")

    identity = protocols["production_identity"]
    haar = protocols["literal_haar_r"]
    matched = protocols["matched_physical"]
    random_names = sorted(
        {str(row["membership"]) for row in rows if str(row["membership"]).startswith("random_")}
    )

    summary: dict[str, object] = {
        "case_count": len(rows),
        "protocol_counts": {key: len(value) for key, value in protocols.items()},
        "architecture_signature": list(next(iter(signatures))),
    }
    lines = [
        "# SIFT1M 128b fixed-architecture membership/rotation results",
        "",
        f"All {len(rows)} cases completed with exit status 0. All structures share one architecture signature.",
        "",
        "## Production identity initialization",
        "",
        "| Membership | J | R@1 | R@10 | R@100 | Overlap@1k | Train(s) | QPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    identity_by_name = {str(row["membership"]): row for row in identity}
    for name in ["learned", "contiguous", *random_names]:
        row = identity_by_name[name]
        lines.append(
            f"| {name} | {row['reconstruction_error']:.3f} | {row['recall1']:.4f} | "
            f"{row['recall10']:.4f} | {row['recall100']:.4f} | {row['overlap1000']:.4f} | "
            f"{row['training_seconds']:.3f} | {row['qps']:.1f} |"
        )

    identity_random = [identity_by_name[name] for name in random_names]
    identity_comparisons = {}
    for metric in METRICS:
        random_values = [float(row[metric]) for row in identity_random]
        learned_value = float(identity_by_name["learned"][metric])
        contiguous_value = float(identity_by_name["contiguous"][metric])
        identity_comparisons[metric] = {
            "learned": learned_value,
            "contiguous": contiguous_value,
            "random_mean": statistics.mean(random_values),
            "random_std": statistics.stdev(random_values),
            "random_min": min(random_values),
            "random_max": max(random_values),
            "learned_minus_random_mean": learned_value - statistics.mean(random_values),
            "learned_minus_contiguous": learned_value - contiguous_value,
        }
    summary["production_identity"] = identity_comparisons
    lines.extend(["", "Identity comparison against five random memberships:", ""])
    lines.append("| Metric | Learned | Contiguous | Random mean±sd | Learned−random | Learned−contiguous |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for metric in METRICS:
        item = identity_comparisons[metric]
        lines.append(
            f"| {metric} | {item['learned']:.6f} | {item['contiguous']:.6f} | "
            f"{item['random_mean']:.6f}±{item['random_std']:.6f} | "
            f"{item['learned_minus_random_mean']:+.6f} | {item['learned_minus_contiguous']:+.6f} |"
        )

    lines.extend([
        "",
        "## Literal Haar-R initialization",
        "",
        "Each learned/contiguous row has 10 Haar seeds. Random is the per-seed mean over five fixed random memberships.",
        "",
        "| Membership class | J mean±sd | R@1 mean±sd | R@10 mean±sd | Overlap mean±sd |",
        "|---|---:|---:|---:|---:|",
    ])
    haar_by_seed = group_by_seed(haar)
    haar_series: dict[str, dict[str, list[float]]] = {
        "learned": defaultdict(list),
        "contiguous": defaultdict(list),
        "random_mean": defaultdict(list),
    }
    for seed in sorted(haar_by_seed):
        seed_rows = haar_by_seed[seed]
        for metric in METRICS:
            haar_series["learned"][metric].append(float(seed_rows["learned"][metric]))
            haar_series["contiguous"][metric].append(float(seed_rows["contiguous"][metric]))
            haar_series["random_mean"][metric].append(
                statistics.mean(float(seed_rows[name][metric]) for name in random_names)
            )
    for name in ("learned", "contiguous", "random_mean"):
        j_mean, j_std = mean_std(haar_series[name]["reconstruction_error"])
        r1_mean, r1_std = mean_std(haar_series[name]["recall1"])
        r10_mean, r10_std = mean_std(haar_series[name]["recall10"])
        ov_mean, ov_std = mean_std(haar_series[name]["overlap1000"])
        lines.append(
            f"| {name} | {j_mean:.2f}±{j_std:.2f} | {r1_mean:.4f}±{r1_std:.4f} | "
            f"{r10_mean:.4f}±{r10_std:.4f} | {ov_mean:.4f}±{ov_std:.4f} |"
        )

    paired = {}
    lines.extend([
        "",
        "Paired differences over the same 10 Haar seeds (two-sided exact sign-flip p):",
        "",
        "| Metric | Learned−contiguous mean [95% bootstrap CI] | p | Learned−random mean [CI] | p |",
        "|---|---:|---:|---:|---:|",
    ])
    for metric in METRICS:
        lc = [
            a - b
            for a, b in zip(
                haar_series["learned"][metric], haar_series["contiguous"][metric]
            )
        ]
        lr = [
            a - b
            for a, b in zip(
                haar_series["learned"][metric], haar_series["random_mean"][metric]
            )
        ]
        lc_ci = bootstrap_mean_ci(lc)
        lr_ci = bootstrap_mean_ci(lr, seed=20260715)
        paired[metric] = {
            "learned_minus_contiguous": {
                "mean": statistics.mean(lc),
                "ci95": list(lc_ci),
                "exact_sign_flip_p": exact_sign_flip_paired_p(lc),
                "values": lc,
            },
            "learned_minus_random_mean": {
                "mean": statistics.mean(lr),
                "ci95": list(lr_ci),
                "exact_sign_flip_p": exact_sign_flip_paired_p(lr),
                "values": lr,
            },
        }
        lines.append(
            f"| {metric} | {statistics.mean(lc):+.6f} [{lc_ci[0]:+.6f}, {lc_ci[1]:+.6f}] | "
            f"{exact_sign_flip_paired_p(lc):.4f} | {statistics.mean(lr):+.6f} "
            f"[{lr_ci[0]:+.6f}, {lr_ci[1]:+.6f}] | {exact_sign_flip_paired_p(lr):.4f} |"
        )
    summary["literal_haar"] = {
        "series": haar_series,
        "paired_comparisons": paired,
    }

    matched_by_seed = group_by_seed(matched)
    matched_spreads = {}
    lines.extend([
        "",
        "## Matched physical initialization",
        "",
        "| Seed | Metric | learned | contiguous | random_101 | max−min |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for seed in sorted(matched_by_seed):
        matched_spreads[str(seed)] = {}
        seed_rows = matched_by_seed[seed]
        for metric in METRICS:
            values = [float(seed_rows[name][metric]) for name in ("learned", "contiguous", "random_101")]
            spread = max(values) - min(values)
            matched_spreads[str(seed)][metric] = spread
            lines.append(
                f"| {seed} | {metric} | {values[0]:.6f} | {values[1]:.6f} | "
                f"{values[2]:.6f} | {spread:.9f} |"
            )
    summary["matched_physical_max_spreads"] = matched_spreads

    max_orth_error = max(float(row["transform_init_orthogonality_error"]) for row in rows)
    summary["max_initial_orthogonality_error"] = max_orth_error
    lines.extend([
        "",
        f"Maximum recorded initial orthogonality error: `{max_orth_error:.9g}`.",
        "",
    ])
    return summary, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_dir", type=Path)
    parser.add_argument(
        "--stable-csv",
        type=Path,
        help="also write a stable parsed CSV at this path",
    )
    args = parser.parse_args()
    log_paths = sorted((args.matrix_dir / "logs").glob("*.log"))
    rows = [parse_log(path) for path in log_paths]
    if len(rows) != 86:
        raise RuntimeError(f"expected 86 logs, found {len(rows)}")

    raw_csv = args.matrix_dir / "results.csv"
    with raw_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if args.stable_csv is not None:
        args.stable_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.stable_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    summary, markdown = summarize(args.matrix_dir, rows)
    (args.matrix_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.matrix_dir / "summary.md").write_text(markdown + "\n")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "results_csv": str(raw_csv),
                "summary_json": str(args.matrix_dir / "summary.json"),
                "summary_md": str(args.matrix_dir / "summary.md"),
            }
        )
    )


if __name__ == "__main__":
    main()
