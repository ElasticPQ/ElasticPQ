#!/usr/bin/env python3
"""Parse the SIFT1M/128b coupled descriptor/orientation path experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


FLOAT = r"[-+0-9.eE]+"
PATHS = (
    "searched_pair",
    "searched_descriptor_neutral_start",
    "balanced_equal_pair",
)


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


def parse_log(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
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
    row: dict[str, object] = {
        "case_id": header(text, "case_id"),
        "protocol": header(text, "protocol"),
        "path_name": header(text, "path_name"),
        "transform_niter": int(header(text, "transform_niter")),
        "proxy_iterations": int(profile.group(1)),
        "exact_iterations": int(profile.group(2)),
        "total_rotation_iterations": int(profile.group(3)),
        "training_seconds": value(text, rf"training total:\s+({FLOAT}) s"),
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
        "transform_init_mode": method["transform_init_mode"],
        "exit_status": 0,
        "log_path": str(path),
    }
    trajectories: list[dict[str, object]] = []
    pattern = re.compile(
        rf"\[profile\] transform\.iter=(\d+) stage=(proxy|exact).*?"
        rf"objective=({FLOAT}) (eval_mse|train_mse)"
    )
    for match in pattern.finditer(text):
        trajectories.append(
            {
                "case_id": row["case_id"],
                "protocol": row["protocol"],
                "path_name": row["path_name"],
                "iteration": int(match.group(1)),
                "stage": match.group(2),
                "objective": float(match.group(3)),
                "objective_kind": match.group(4),
            }
        )
    if len(trajectories) != row["total_rotation_iterations"]:
        raise RuntimeError(f"trajectory length mismatch: {path}")
    return row, trajectories


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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
    rows: list[dict[str, object]] = []
    trajectories: list[dict[str, object]] = []
    for case_id in sorted(expected):
        row, trajectory = parse_log(logs[case_id])
        rows.append(row)
        trajectories.extend(trajectory)
    if any(row["train_rows"] != 100000 for row in rows):
        raise RuntimeError("not all cases used the full 100K training split")
    if any(row["main_bits"] != 128 for row in rows):
        raise RuntimeError("not all cases used a 128-bit product payload")
    if any(row["transform_init_mode"] != "identity" for row in rows):
        raise RuntimeError("unexpected transform initialization mode")
    for row in rows:
        if str(row["protocol"]).startswith("fixed_"):
            expected_iters = int(str(row["protocol"]).split("_")[1])
            if row["proxy_iterations"] != expected_iters or row["exact_iterations"] != 1:
                raise RuntimeError(f"fixed schedule mismatch: {row['case_id']}")

    write_csv(args.matrix_dir / "results.csv", rows)
    write_csv(args.matrix_dir / "trajectories.csv", trajectories)
    by_protocol: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_protocol[str(row["protocol"])][str(row["path_name"])] = row
    plateau: dict[str, float] = {}
    for row in rows:
        proxy = [
            float(item["objective"])
            for item in trajectories
            if item["case_id"] == row["case_id"] and item["stage"] == "proxy"
        ]
        tail = proxy[-10:]
        plateau[str(row["case_id"])] = (max(tail) - min(tail)) / abs(min(tail))

    protocols = [
        "auto_stop",
        f"fixed_{manifest['extended_iterations']}",
        f"fixed_{manifest['deep_iterations']}",
    ]
    terminal_iterations = int(manifest.get("terminal_iterations", 0))
    if terminal_iterations:
        protocols.append(f"fixed_{terminal_iterations}")
    lines = [
        "# SIFT1M/128b coupled descriptor/orientation paths",
        "",
        "All paths use the product-only codec, full 100K training split, identity",
        "transform initialization after each structure's physical permutation, and",
        "one final exact-polish update.",
        "",
        "| Schedule | Path | Proxy iters | J | R@1 | R@10 | Overlap@1k | Last-10 span | Train(s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    comparisons: dict[str, object] = {}
    for protocol in protocols:
        named = by_protocol[protocol]
        expected_paths = (
            {"searched_pair", "balanced_equal_pair"}
            if terminal_iterations and protocol == f"fixed_{terminal_iterations}"
            else set(PATHS)
        )
        if set(named) != expected_paths:
            raise RuntimeError(f"incomplete protocol {protocol}: {sorted(named)}")
        for path_name in PATHS:
            if path_name not in named:
                continue
            row = named[path_name]
            lines.append(
                f"| {protocol} | {path_name} | {row['proxy_iterations']} | "
                f"{float(row['reconstruction_error']):.2f} | {float(row['recall1']):.4f} | "
                f"{float(row['recall10']):.4f} | {float(row['overlap1000']):.4f} | "
                f"{100.0 * plateau[str(row['case_id'])]:.4f}% | "
                f"{float(row['training_seconds']):.1f} |"
            )
        searched = named["searched_pair"]
        deltas = {}
        for reference in PATHS[1:]:
            if reference not in named:
                continue
            control = named[reference]
            deltas[reference] = {
                "control_minus_searched_J": float(control["reconstruction_error"])
                - float(searched["reconstruction_error"]),
                "searched_minus_control_recall1": float(searched["recall1"])
                - float(control["recall1"]),
                "searched_minus_control_recall10": float(searched["recall10"])
                - float(control["recall10"]),
                "searched_minus_control_overlap1000": float(searched["overlap1000"])
                - float(control["overlap1000"]),
            }
        comparisons[protocol] = deltas
    summary = {
        "case_count": len(rows),
        "paths": list(PATHS),
        "plateau_last10_relative_span": plateau,
        "comparisons": comparisons,
    }
    (args.matrix_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.matrix_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows": len(rows), "trajectory_rows": len(trajectories)}))


if __name__ == "__main__":
    main()
