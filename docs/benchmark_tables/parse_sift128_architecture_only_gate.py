#!/usr/bin/env python3
"""Parse and summarize the SIFT1M/128b architecture-only gate."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FLOAT = r"[-+0-9.eE]+"
METRICS = ("reconstruction_error", "recall1", "recall10", "recall100", "overlap1000")
LEARNED = "learned_architecture"
ORDER = (
    LEARNED,
    "learned_dims_balanced_bits",
    "balanced_dims_dp_bits_matched_m",
    "balanced_dims_balanced_bits_matched_m",
    "grid_best_dp_bits",
)
CONTRASTS = (
    (
        "dimensions_at_dp_bits",
        "learned_architecture",
        "balanced_dims_dp_bits_matched_m",
    ),
    (
        "dimensions_at_balanced_bits",
        "learned_dims_balanced_bits",
        "balanced_dims_balanced_bits_matched_m",
    ),
    (
        "dp_bits_at_learned_dimensions",
        "learned_architecture",
        "learned_dims_balanced_bits",
    ),
    (
        "dp_bits_at_balanced_dimensions",
        "balanced_dims_dp_bits_matched_m",
        "balanced_dims_balanced_bits_matched_m",
    ),
    (
        "searched_descriptor_vs_proxy_grid",
        "learned_architecture",
        "grid_best_dp_bits",
    ),
)


def search_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing pattern: {pattern}")
    return float(match.group(1))


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
    dataset_line = re.search(r"^meta\.dataset (\{.*\})$", text, re.MULTILINE)
    if not method_line or not dataset_line:
        raise RuntimeError(f"missing metadata: {path}")
    method = json.loads(method_line.group(1))
    dataset = json.loads(dataset_line.group(1))
    structure_path = Path(parse_header(text, "structure_path"))
    structure = json.loads(structure_path.read_text())
    sizes = [len(group["dims"]) for group in structure["groups"]]
    bits = [group["nbits"] for group in structure["groups"]]
    return {
        "case_id": parse_header(text, "case_id"),
        "architecture": parse_header(text, "architecture"),
        "init_mode": parse_header(text, "init_mode"),
        "init_seed": int(parse_header(text, "init_seed")),
        "structure_path": str(structure_path.resolve().relative_to(REPO_ROOT)),
        "group_count": len(structure["groups"]),
        "group_sizes": ",".join(map(str, sizes)),
        "group_bits": ",".join(map(str, bits)),
        "main_bits": structure["total_bits"],
        "design_proxy_j": float(structure["meta"]["design_proxy_j"]),
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
        "max_rss_kib": int(
            search_float(r"Maximum resident set size \(kbytes\):\s+(\d+)", text)
        ),
        "train_rows": int(dataset["train_rows"]),
        "transform_init_mode_meta": method["transform_init_mode"],
        "transform_init_seed_meta": int(method["transform_init_seed"]),
        "transform_init_orthogonality_error": float(
            method["transform_init_orthogonality_error"]
        ),
        "exit_status": 0,
        "log_path": str(path.resolve().relative_to(REPO_ROOT)),
    }


def mean_ci(values: list[float], seed: int = 20260715) -> tuple[float, float]:
    rng = random.Random(seed)
    samples = sorted(
        statistics.mean(rng.choice(values) for _ in values) for _ in range(20000)
    )
    return samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]


def sign_flip_p(diffs: list[float]) -> float:
    observed = abs(statistics.mean(diffs))
    count = 0
    total = 1 << len(diffs)
    for mask in range(total):
        value = statistics.mean(
            diff if (mask >> index) & 1 else -diff
            for index, diff in enumerate(diffs)
        )
        if abs(value) >= observed - 1e-15:
            count += 1
    return count / total


def preferred_advantage(metric: str, preferred: float, reference: float) -> float:
    if metric == "reconstruction_error":
        return reference - preferred
    return preferred - reference


def summarize(rows: list[dict[str, object]]) -> tuple[dict, str]:
    by_arch: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_seed: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_arch[str(row["architecture"])].append(row)
        by_seed[int(row["init_seed"])][str(row["architecture"])] = row
    expected = set(ORDER)
    if set(by_arch) != expected:
        raise RuntimeError(f"architecture set mismatch: {set(by_arch)}")
    for seed, seed_rows in by_seed.items():
        if set(seed_rows) != expected:
            raise RuntimeError(f"incomplete seed {seed}: {set(seed_rows)}")

    summary: dict[str, object] = {
        "case_count": len(rows),
        "seeds": sorted(by_seed),
        "architectures": {},
        "paired": {},
    }
    lines = [
        "# SIFT1M/128b product-only architecture-only gate",
        "",
        f"All {len(rows)} fixed-structure cases completed with exit status 0.",
        (
            "Every architecture uses the same "
            f"{len(by_seed)} matched-physical initialization seeds."
        ),
        "",
        "| Architecture | M | design J* | J | R@1 | R@10 | R@100 | Overlap@1k | Train(s) | QPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for architecture in ORDER:
        arch_rows = sorted(by_arch[architecture], key=lambda row: int(row["init_seed"]))
        aggregate = {}
        for metric in (*METRICS, "training_seconds", "qps"):
            values = [float(row[metric]) for row in arch_rows]
            aggregate[metric] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values),
                "min": min(values),
                "max": max(values),
            }
        summary["architectures"][architecture] = aggregate
        row0 = arch_rows[0]
        lines.append(
            f"| {architecture} | {row0['group_count']} | {row0['design_proxy_j']:.1f} | "
            f"{aggregate['reconstruction_error']['mean']:.1f}±{aggregate['reconstruction_error']['std']:.1f} | "
            f"{aggregate['recall1']['mean']:.4f}±{aggregate['recall1']['std']:.4f} | "
            f"{aggregate['recall10']['mean']:.4f}±{aggregate['recall10']['std']:.4f} | "
            f"{aggregate['recall100']['mean']:.4f}±{aggregate['recall100']['std']:.4f} | "
            f"{aggregate['overlap1000']['mean']:.4f}±{aggregate['overlap1000']['std']:.4f} | "
            f"{aggregate['training_seconds']['mean']:.1f} | {aggregate['qps']['mean']:.1f} |"
        )

    lines.extend(["", "## Paired factorial and grid contrasts", ""])
    lines.append(
        "Positive values favor the named preferred variant; J is reference minus preferred."
    )
    lines.extend(
        [
            "",
            "| Contrast | Metric | Mean advantage | Bootstrap 95% CI | Exact sign-flip p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for contrast, preferred, reference in CONTRASTS:
        contrast_summary = {
            "preferred": preferred,
            "reference": reference,
        }
        for metric in METRICS:
            diffs = [
                preferred_advantage(
                    metric,
                    float(by_seed[seed][preferred][metric]),
                    float(by_seed[seed][reference][metric]),
                )
                for seed in sorted(by_seed)
            ]
            ci = mean_ci(diffs)
            result = {
                "differences": diffs,
                "mean": statistics.mean(diffs),
                "ci95": list(ci),
                "sign_flip_p": sign_flip_p(diffs),
            }
            contrast_summary[metric] = result
            lines.append(
                f"| {contrast} | {metric} | {result['mean']:.6f} | "
                f"[{ci[0]:.6f}, {ci[1]:.6f}] | {result['sign_flip_p']:.4f} |"
            )
        summary["paired"][contrast] = contrast_summary
    return summary, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_dir", type=Path)
    args = parser.parse_args()
    matrix_dir = args.matrix_dir.resolve()
    logs = sorted((matrix_dir / "logs").glob("*.log"))
    if not logs:
        raise RuntimeError("no logs found")
    rows = [parse_log(path) for path in logs]
    rows.sort(key=lambda row: (int(row["init_seed"]), str(row["architecture"])))
    fieldnames = list(rows[0])
    with (matrix_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary, markdown = summarize(rows)
    (matrix_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (matrix_dir / "summary.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
