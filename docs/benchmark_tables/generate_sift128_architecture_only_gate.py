#!/usr/bin/env python3
"""Generate configs and a case matrix for the SIFT1M/128b architecture-only gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ARCHITECTURES = (
    "learned_architecture",
    "learned_dims_balanced_bits",
    "balanced_dims_dp_bits_matched_m",
    "balanced_dims_balanced_bits_matched_m",
    "grid_best_dp_bits",
)
SEEDS = tuple(range(1001, 1011))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/epq_train_standard.json"),
    )
    parser.add_argument(
        "--structure-dir",
        type=Path,
        default=Path("docs/benchmark_tables/sift128_architecture_only_structures"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.base_config.resolve().read_text())

    configs: dict[int, Path] = {}
    for seed in SEEDS:
        config = json.loads(json.dumps(base))
        transform = config.setdefault("index", {}).setdefault("transform", {})
        transform["transform_init_mode"] = "matched_physical"
        transform["transform_init_seed"] = seed
        config.setdefault("builder", {})["auto_reuse_structure"] = False
        path = config_dir / f"matched_physical_{seed}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        configs[seed] = path

    cases_path = output_dir / "cases.tsv"
    with cases_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["case_id", "architecture", "init_seed", "structure_path", "config_path"]
        )
        for seed in SEEDS:
            for architecture in ARCHITECTURES:
                structure_path = (args.structure_dir.resolve() / f"{architecture}.json")
                if not structure_path.exists():
                    raise FileNotFoundError(structure_path)
                writer.writerow(
                    [
                        f"{architecture}__w{seed}",
                        architecture,
                        seed,
                        structure_path,
                        configs[seed],
                    ]
                )
    manifest = {
        "architectures": list(ARCHITECTURES),
        "seeds": list(SEEDS),
        "case_count": len(ARCHITECTURES) * len(SEEDS),
        "initialization": "matched_physical",
        "codec": "product-only EPQ",
        "dataset": "SIFT1M",
        "bits": 128,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {cases_path} with {manifest['case_count']} cases")


if __name__ == "__main__":
    main()
