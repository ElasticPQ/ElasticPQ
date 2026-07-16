#!/usr/bin/env python3
"""Parse the paired SIFT1M/128b descriptor/start interaction matrix."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


FLOAT = r"[-+0-9.eE]+"
CELLS = (
    "searched_searched",
    "searched_neutral",
    "balanced_searched",
    "balanced_neutral",
)
METRICS = ("reconstruction_error", "recall1", "recall10", "overlap1000")


def header(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.*)$", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing header {key}")
    return match.group(1).strip()


def value(text: str, pattern: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing pattern {pattern}")
    return float(match.group(1))


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text()
    if not re.search(r"^\s*Exit status: 0$", text, re.MULTILINE):
        raise RuntimeError(f"non-zero or missing exit status: {path}")
    recall = re.search(
        rf"recall@1: ({FLOAT}) recall@10: ({FLOAT}) "
        rf"recall@100: ({FLOAT}) recall@1000: ({FLOAT})",
        text,
    )
    profile = re.search(
        r"\[profile\] transform .*?proxy_iters=(\d+).*?exact_iters=(\d+) "
        r"iterations=(\d+)",
        text,
    )
    method_line = re.search(r"^meta\.method (\{.*\})$", text, re.MULTILINE)
    dataset_line = re.search(r"^meta\.dataset (\{.*\})$", text, re.MULTILINE)
    if not recall or not profile or not method_line or not dataset_line:
        raise RuntimeError(f"missing metrics or metadata: {path}")
    method = json.loads(method_line.group(1))
    dataset = json.loads(dataset_line.group(1))
    return {
        "case_id": header(text, "case_id"),
        "cell": header(text, "cell"),
        "transform_seed": int(header(text, "transform_seed")),
        "proxy_iterations": int(profile.group(1)),
        "exact_iterations": int(profile.group(2)),
        "training_seconds": value(text, rf"training total:\s+({FLOAT}) s"),
        "add_seconds": value(text, rf"add/encode time:\s+({FLOAT}) s"),
        "search_seconds": value(text, rf"search time:\s+({FLOAT}) s"),
        "reconstruction_error": value(
            text, rf"reconstruction error \(sample\):\s+({FLOAT})"
        ),
        "recall1": float(recall.group(1)),
        "recall10": float(recall.group(2)),
        "recall100": float(recall.group(3)),
        "recall1000": float(recall.group(4)),
        "overlap1000": value(text, rf"overlap@1000\(gt=1000\):\s+({FLOAT})"),
        "qps": value(text, rf"QPS:\s+({FLOAT})"),
        "max_rss_kib": int(value(text, r"Maximum resident set size \(kbytes\):\s+(\d+)")),
        "train_rows": int(dataset["train_rows"]),
        "main_bits": int(method["total_bits"]),
        "transform_seed_meta": int(method["transform_seed"]),
        "transform_init_mode": method["transform_init_mode"],
        "exit_status": 0,
        "log_path": str(path),
    }


def mean_ci(values: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    samples = sorted(
        statistics.mean(rng.choice(values) for _ in values) for _ in range(20000)
    )
    return samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]


def sign_flip_p(values: list[float]) -> float:
    observed = abs(statistics.mean(values))
    total = 1 << len(values)
    extreme = 0
    for mask in range(total):
        candidate = statistics.mean(
            value if (mask >> index) & 1 else -value
            for index, value in enumerate(values)
        )
        if abs(candidate) >= observed - 1e-15:
            extreme += 1
    return extreme / total


def start_effect(metric: str, searched_start: float, neutral_start: float) -> float:
    if metric == "reconstruction_error":
        return neutral_start - searched_start
    return searched_start - neutral_start


def summarize(rows: list[dict[str, object]]) -> tuple[dict, str]:
    by_seed: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    by_cell: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["transform_seed"])][str(row["cell"])] = row
        by_cell[str(row["cell"])].append(row)
    if set(by_cell) != set(CELLS):
        raise RuntimeError(f"cell set mismatch: {set(by_cell)}")
    for seed, cells in by_seed.items():
        if set(cells) != set(CELLS):
            raise RuntimeError(f"incomplete seed {seed}: {set(cells)}")

    summary: dict[str, object] = {
        "case_count": len(rows),
        "seeds": sorted(by_seed),
        "cells": {},
        "effects": {},
    }
    lines = [
        "# SIFT1M/128b paired descriptor/start interaction",
        "",
        f"All {len(rows)} cases completed with exit status 0 over {len(by_seed)} shared seeds.",
        "Positive effects favor the searched physical start; positive interactions",
        "mean that the searched-start benefit is larger with the searched descriptor.",
        "",
        "| Descriptor | Physical start | J | R@1 | R@10 | Overlap@1k | Train(s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "searched_searched": ("Searched", "Searched"),
        "searched_neutral": ("Searched", "Neutral"),
        "balanced_searched": ("Balanced/equal", "Searched"),
        "balanced_neutral": ("Balanced/equal", "Neutral"),
    }
    for cell in CELLS:
        cell_rows = sorted(by_cell[cell], key=lambda row: int(row["transform_seed"]))
        aggregate = {}
        for metric in (*METRICS, "training_seconds"):
            values = [float(row[metric]) for row in cell_rows]
            aggregate[metric] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values),
                "min": min(values),
                "max": max(values),
            }
        summary["cells"][cell] = aggregate
        descriptor_label, start_label = labels[cell]
        lines.append(
            f"| {descriptor_label} | {start_label} | "
            f"{aggregate['reconstruction_error']['mean']:.1f}±{aggregate['reconstruction_error']['std']:.1f} | "
            f"{aggregate['recall1']['mean']:.4f}±{aggregate['recall1']['std']:.4f} | "
            f"{aggregate['recall10']['mean']:.4f}±{aggregate['recall10']['std']:.4f} | "
            f"{aggregate['overlap1000']['mean']:.4f}±{aggregate['overlap1000']['std']:.4f} | "
            f"{aggregate['training_seconds']['mean']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Paired start effects and descriptor-conditioned interaction",
            "",
            "| Effect | Metric | Mean | Bootstrap 95% CI | Exact sign-flip p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for metric_index, metric in enumerate(METRICS):
        searched_effects = []
        balanced_effects = []
        for seed in sorted(by_seed):
            cells = by_seed[seed]
            searched_effects.append(
                start_effect(
                    metric,
                    float(cells["searched_searched"][metric]),
                    float(cells["searched_neutral"][metric]),
                )
            )
            balanced_effects.append(
                start_effect(
                    metric,
                    float(cells["balanced_searched"][metric]),
                    float(cells["balanced_neutral"][metric]),
                )
            )
        interaction = [
            searched - balanced
            for searched, balanced in zip(searched_effects, balanced_effects)
        ]
        for effect_index, (name, values) in enumerate(
            (
                ("searched_descriptor_start_effect", searched_effects),
                ("balanced_descriptor_start_effect", balanced_effects),
                ("descriptor_conditioned_interaction", interaction),
            )
        ):
            ci = mean_ci(values, 20260716 + 100 * metric_index + effect_index)
            result = {
                "values": values,
                "mean": statistics.mean(values),
                "ci95": list(ci),
                "sign_flip_p": sign_flip_p(values),
            }
            summary["effects"].setdefault(name, {})[metric] = result
            lines.append(
                f"| {name} | {metric} | {result['mean']:.6f} | "
                f"[{ci[0]:.6f}, {ci[1]:.6f}] | {result['sign_flip_p']:.4f} |"
            )
    return summary, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.matrix_dir / "manifest.json").read_text())
    expected = {case["case_id"] for case in manifest["cases"]}
    logs = {path.stem: path for path in (args.matrix_dir / "logs").glob("*.log")}
    if set(logs) != expected:
        raise RuntimeError(
            f"log set mismatch: missing={sorted(expected - set(logs))}, "
            f"extra={sorted(set(logs) - expected)}"
        )
    rows = [parse_log(logs[case_id]) for case_id in sorted(expected)]
    rows.sort(key=lambda row: (int(row["transform_seed"]), str(row["cell"])))
    if any(row["train_rows"] != 100000 for row in rows):
        raise RuntimeError("not all cases used the full 100K training split")
    if any(row["main_bits"] != 128 for row in rows):
        raise RuntimeError("not all cases used a 128-bit product payload")
    if any(row["proxy_iterations"] != 128 or row["exact_iterations"] != 1 for row in rows):
        raise RuntimeError("fixed schedule mismatch")
    if any(row["transform_init_mode"] != "identity" for row in rows):
        raise RuntimeError("unexpected transform initialization mode")
    if any(row["transform_seed"] != row["transform_seed_meta"] for row in rows):
        raise RuntimeError("transform seed metadata mismatch")

    with (args.matrix_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary, markdown = summarize(rows)
    (args.matrix_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.matrix_dir / "summary.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
