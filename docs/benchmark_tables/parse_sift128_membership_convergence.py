#!/usr/bin/env python3
"""Parse the SIFT1M fixed-architecture membership convergence control."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


FLOAT = r"[-+0-9.eE]+"
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
    if not recall or not profile:
        raise RuntimeError(f"missing recall or transform profile: {path}")

    row: dict[str, object] = {
        "case_id": header(text, "case_id"),
        "protocol": header(text, "protocol"),
        "membership": header(text, "membership"),
        "membership_seed": int(header(text, "membership_seed")),
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
        "overlap1000": value(
            text, rf"overlap@1000\(gt=1000\):\s+({FLOAT})"
        ),
        "qps": value(text, rf"QPS:\s+({FLOAT})"),
        "max_rss_kib": int(
            value(text, r"Maximum resident set size \(kbytes\):\s+(\d+)")
        ),
        "exit_status": 0,
        "log_path": str(path),
    }

    trajectories = []
    iteration_pattern = re.compile(
        rf"\[profile\] transform\.iter=(\d+) stage=(proxy|exact).*?"
        rf"objective=({FLOAT}) (eval_mse|train_mse)"
    )
    for match in iteration_pattern.finditer(text):
        trajectories.append(
            {
                "case_id": row["case_id"],
                "protocol": row["protocol"],
                "membership": row["membership"],
                "iteration": int(match.group(1)),
                "stage": match.group(2),
                "objective": float(match.group(3)),
                "objective_kind": match.group(4),
            }
        )
    if len(trajectories) != row["total_rotation_iterations"]:
        raise RuntimeError(
            f"trajectory length mismatch for {path}: "
            f"{len(trajectories)} != {row['total_rotation_iterations']}"
        )
    return row, trajectories


def mean_sd(rows: list[dict[str, object]], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def render_summary(
    rows: list[dict[str, object]],
    trajectories: list[dict[str, object]],
    extended: int,
    deep: int,
) -> tuple[dict[str, object], str]:
    by_protocol: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_protocol[str(row["protocol"])].append(row)
    fixed_name = f"fixed_{extended}"
    deep_name = f"fixed_{deep}"
    if set(by_protocol) != {"auto_stop", fixed_name, deep_name}:
        raise RuntimeError(f"unexpected protocols: {sorted(by_protocol)}")

    summary: dict[str, object] = {
        "case_count": len(rows),
        "extended_iterations": extended,
        "deep_iterations": deep,
    }
    lines = [
        "# SIFT1M/128b membership convergence control",
        "",
        "The experiment fixes the searched 120-bit product architecture used inside the",
        "complete 128-bit codec and disables the residual tail to isolate UnevenOPQ.",
        "",
        "| Schedule | Membership | Proxy iters | J | R@1 | R@10 | Overlap@1k | Train(s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    aggregates: dict[str, object] = {}
    for protocol in ("auto_stop", fixed_name, deep_name):
        protocol_rows = by_protocol[protocol]
        named = {str(row["membership"]): row for row in protocol_rows}
        random_rows = [row for row in protocol_rows if str(row["membership"]).startswith("random_")]
        protocol_summary: dict[str, object] = {}
        for membership in ("learned", "contiguous"):
            row = named[membership]
            lines.append(
                f"| {protocol} | {membership} | {row['proxy_iterations']} | "
                f"{float(row['reconstruction_error']):.2f} | {float(row['recall1']):.4f} | "
                f"{float(row['recall10']):.4f} | {float(row['overlap1000']):.4f} | "
                f"{float(row['training_seconds']):.1f} |"
            )
            protocol_summary[membership] = row
        if random_rows:
            random_summary = {}
            for metric in ("proxy_iterations", *METRICS, "training_seconds"):
                mean, sd = mean_sd(random_rows, metric)
                random_summary[metric] = {"mean": mean, "sd": sd}
            lines.append(
                f"| {protocol} | random (5) | {random_summary['proxy_iterations']['mean']:.1f}"
                f"±{random_summary['proxy_iterations']['sd']:.1f} | "
                f"{random_summary['reconstruction_error']['mean']:.2f}±"
                f"{random_summary['reconstruction_error']['sd']:.2f} | "
                f"{random_summary['recall1']['mean']:.4f}±{random_summary['recall1']['sd']:.4f} | "
                f"{random_summary['recall10']['mean']:.4f}±{random_summary['recall10']['sd']:.4f} | "
                f"{random_summary['overlap1000']['mean']:.4f}±"
                f"{random_summary['overlap1000']['sd']:.4f} | "
                f"{random_summary['training_seconds']['mean']:.1f}±"
                f"{random_summary['training_seconds']['sd']:.1f} |"
            )
            protocol_summary["random"] = random_summary
        aggregates[protocol] = protocol_summary

    fixed = aggregates[fixed_name]
    learned = fixed["learned"]
    contiguous = fixed["contiguous"]
    advantages = {
        "contiguous_minus_learned_J": float(contiguous["reconstruction_error"])
        - float(learned["reconstruction_error"]),
        "learned_minus_contiguous_recall1": float(learned["recall1"])
        - float(contiguous["recall1"]),
        "learned_minus_contiguous_recall10": float(learned["recall10"])
        - float(contiguous["recall10"]),
        "random_mean_minus_learned_J": float(fixed["random"]["reconstruction_error"]["mean"])
        - float(learned["reconstruction_error"]),
        "learned_minus_random_mean_recall1": float(learned["recall1"])
        - float(fixed["random"]["recall1"]["mean"]),
        "learned_minus_random_mean_recall10": float(learned["recall10"])
        - float(fixed["random"]["recall10"]["mean"]),
    }
    deep_fixed = aggregates[deep_name]
    deep_learned = deep_fixed["learned"]
    deep_contiguous = deep_fixed["contiguous"]
    deep_advantages = {
        "contiguous_minus_learned_J": float(deep_contiguous["reconstruction_error"])
        - float(deep_learned["reconstruction_error"]),
        "learned_minus_contiguous_recall1": float(deep_learned["recall1"])
        - float(deep_contiguous["recall1"]),
        "learned_minus_contiguous_recall10": float(deep_learned["recall10"])
        - float(deep_contiguous["recall10"]),
        "learned_minus_contiguous_overlap1000": float(deep_learned["overlap1000"])
        - float(deep_contiguous["overlap1000"]),
    }

    plateau: dict[str, object] = {}
    for protocol in (fixed_name, deep_name):
        for row in by_protocol[protocol]:
            case = str(row["case_id"])
            proxy = [
                float(item["objective"])
                for item in trajectories
                if item["case_id"] == case and item["stage"] == "proxy"
            ]
            tail = proxy[-10:]
            plateau[case] = {
                "last_objective": proxy[-1],
                "last10_relative_span": (max(tail) - min(tail)) / abs(min(tail)),
            }

    summary["aggregates"] = aggregates
    summary["fixed_iteration_advantages"] = advantages
    summary["deep_iteration_advantages"] = deep_advantages
    summary["fixed_iteration_plateau"] = plateau
    lines.extend(
        [
            "",
            f"At {extended} forced proxy iterations, searched-membership advantages are:",
            "",
            f"- contiguous minus searched J: {advantages['contiguous_minus_learned_J']:+.2f}",
            f"- searched minus contiguous R@1/R@10: "
            f"{advantages['learned_minus_contiguous_recall1']:+.4f}/"
            f"{advantages['learned_minus_contiguous_recall10']:+.4f}",
            f"- random-mean minus searched J: {advantages['random_mean_minus_learned_J']:+.2f}",
            f"- searched minus random-mean R@1/R@10: "
            f"{advantages['learned_minus_random_mean_recall1']:+.4f}/"
            f"{advantages['learned_minus_random_mean_recall10']:+.4f}",
            "",
            f"The {deep}-iteration searched/contiguous endpoints are reported above as a",
            "deeper terminal check after the common 128-iteration production cap.",
            f"Their contiguous-minus-searched J gap is "
            f"{deep_advantages['contiguous_minus_learned_J']:+.2f}; searched-minus-contiguous",
            f"R@1/R@10/Overlap gaps are "
            f"{deep_advantages['learned_minus_contiguous_recall1']:+.4f}/"
            f"{deep_advantages['learned_minus_contiguous_recall10']:+.4f}/"
            f"{deep_advantages['learned_minus_contiguous_overlap1000']:+.4f}.",
            "",
            "The fixed schedule disables early stopping; its per-iteration profile is in",
            "`trajectories.csv`. The final exact-polish iteration is excluded from the",
            "proxy-stage plateau diagnostic because it uses a different codebook capacity.",
        ]
    )
    return summary, "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_dir", type=Path)
    parser.add_argument("--stable-results", type=Path)
    parser.add_argument("--stable-trajectories", type=Path)
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

    fixed_name = f"fixed_{manifest['extended_iterations']}"
    deep_name = f"fixed_{manifest['deep_iterations']}"
    for row in rows:
        if row["protocol"] in {fixed_name, deep_name}:
            expected_iterations = (
                manifest["extended_iterations"]
                if row["protocol"] == fixed_name
                else manifest["deep_iterations"]
            )
            if row["proxy_iterations"] != expected_iterations:
                raise RuntimeError(f"fixed schedule did not run to cap: {row['case_id']}")
            if row["exact_iterations"] != 1:
                raise RuntimeError(f"unexpected exact-polish count: {row['case_id']}")

    results_path = args.matrix_dir / "results.csv"
    trajectories_path = args.matrix_dir / "trajectories.csv"
    write_csv(results_path, rows)
    write_csv(trajectories_path, trajectories)
    if args.stable_results:
        write_csv(args.stable_results, rows)
    if args.stable_trajectories:
        write_csv(args.stable_trajectories, trajectories)

    summary, markdown = render_summary(
        rows,
        trajectories,
        int(manifest["extended_iterations"]),
        int(manifest["deep_iterations"]),
    )
    (args.matrix_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.matrix_dir / "summary.md").write_text(markdown)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "trajectory_rows": len(trajectories),
                "summary": str(args.matrix_dir / "summary.md"),
            }
        )
    )


if __name__ == "__main__":
    main()
