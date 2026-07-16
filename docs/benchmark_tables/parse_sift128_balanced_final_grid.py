#!/usr/bin/env python3
"""Validate and summarize the balanced final-codec M-grid."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def require(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"missing {label}")
    return match.group(1)


def parse_log(path: Path, case: dict[str, str]) -> dict[str, object]:
    text = path.read_text()
    if int(require(r"^\s*Exit status:\s+(\d+)\s*$", text, "exit status")) != 0:
        raise RuntimeError(f"nonzero exit: {path}")
    dataset = json.loads(require(r"^meta\.dataset\s+(\{.*\})$", text, "dataset metadata"))
    config = json.loads(require(r"^meta\.config\s+(\{.*\})$", text, "config metadata"))
    run = json.loads(require(r"^meta\.run\s+(\{.*\})$", text, "run metadata"))
    transform = config["index"]["transform"]
    if dataset["train_rows"] != 100000 or dataset["train_rows_full"] != 100000:
        raise RuntimeError(f"unexpected train rows: {path}")
    expected_niter = int(case["transform_niter"])
    if run["maxtrain"] != 0 or transform["transform_niter"] != expected_niter:
        raise RuntimeError(f"unexpected training protocol: {path}")
    if transform["transform_init_mode"] != "identity":
        raise RuntimeError(f"unexpected init mode: {path}")
    proxy_iters = int(require(r"proxy_iters=(\d+)", text, "proxy iterations"))
    exact_iters = int(require(r"exact_iters=(\d+)", text, "exact iterations"))
    if (proxy_iters, exact_iters) == (expected_niter, 1):
        trajectory_stage = "proxy"
        terminal_iteration = expected_niter + 1
    elif (proxy_iters, exact_iters) == (0, expected_niter):
        # When every block has at most transform_proxy_max_bits, the complete
        # codec is already the optimization codec and no separate polish runs.
        trajectory_stage = "exact"
        terminal_iteration = expected_niter
    else:
        raise RuntimeError(
            f"unexpected rotation stages proxy={proxy_iters} exact={exact_iters}: {path}"
        )

    trajectory = [
        float(value)
        for value in re.findall(
            rf"transform\.iter=\d+ stage={trajectory_stage} .*? objective=({FLOAT}) train_mse",
            text,
        )
    ]
    if len(trajectory) != expected_niter:
        raise RuntimeError(f"unexpected trajectory length {len(trajectory)}: {path}")
    last10 = trajectory[-10:]
    return {
        "case_id": case["case_id"],
        "m": int(case["m"]),
        "allocation": case["allocation"],
        "holdout_mse": float(
            require(
                rf"transform\.final_holdout .*? objective=({FLOAT}) exact_mse",
                text,
                "final holdout MSE",
            )
        ),
        "reconstruction_error": float(
            require(rf"reconstruction error \(sample\):\s+({FLOAT})", text, "J")
        ),
        "recall1": float(require(rf"recall@1:\s+({FLOAT})", text, "Recall@1")),
        "recall10": float(require(rf"recall@10:\s+({FLOAT})", text, "Recall@10")),
        "recall100": float(require(rf"recall@100:\s+({FLOAT})", text, "Recall@100")),
        "overlap1000": float(
            require(rf"overlap@1000\(gt=1000\):\s+({FLOAT})", text, "overlap")
        ),
        "exact_polish_mse": float(
            require(
                rf"transform\.iter={terminal_iteration} stage=exact .*? objective=({FLOAT}) train_mse",
                text,
                "exact polish MSE",
            )
        ),
        "terminal_proxy_mse": trajectory[-1],
        "last10_relative_span": (max(last10) - min(last10)) / trajectory[-1],
        "training_seconds": float(
            require(rf"^\s*training total:\s+({FLOAT}) s$", text, "training seconds")
        ),
        "qps": float(require(rf"^\s*QPS:\s+({FLOAT})$", text, "QPS")),
        "log_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, required=True)
    args = parser.parse_args()

    cases: list[dict[str, str]] = []
    with (args.matrix_dir / "cases.tsv").open(newline="") as handle:
        cases.extend(csv.DictReader(handle, delimiter="\t"))
    rows = [
        parse_log(args.matrix_dir / "logs" / f"{case['case_id']}.log", case)
        for case in cases
    ]
    winner = min(rows, key=lambda row: float(row["holdout_mse"]))

    with (args.matrix_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "case_count": len(rows),
        "selection_rule": "minimum disjoint final exact-codebook holdout MSE",
        "winner": winner,
    }
    (args.matrix_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# SIFT1M/128b balanced final-codec M-grid",
        "",
        "Winner selection uses only the disjoint final exact-codebook holdout MSE.",
        "All test metrics are reported after applying that fixed selection rule.",
        "",
        "| M | Allocation | Holdout MSE | J | R@1 | R@10 | Exact polish MSE | Last-10 span |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        mark = " (winner)" if row is winner else ""
        lines.append(
            f"| {row['m']} | {row['allocation']}{mark} | "
            f"{row['holdout_mse']:.6f} | {row['reconstruction_error']:.3f} | "
            f"{row['recall1']:.4f} | {row['recall10']:.4f} | "
            f"{row['exact_polish_mse']:.6f} | "
            f"{100.0 * row['last10_relative_span']:.4f}% |"
        )
    (args.matrix_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
