#!/usr/bin/env python3
"""Generate the SIFT1M/128b fixed-architecture convergence-control matrix."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from generate_sift128_membership_rotation_matrix import make_structure, write_json


DEFAULT_MEMBERSHIP_SEEDS = (101, 202, 303, 404, 505)


def parse_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def make_config(base: dict, transform_niter: int) -> dict:
    config = copy.deepcopy(base)
    config.setdefault("builder", {})["auto_reuse_structure"] = False
    transform = config.setdefault("index", {}).setdefault("transform", {})
    transform["transform_niter"] = transform_niter
    transform["transform_seed"] = 123
    transform["transform_init_mode"] = "identity"
    transform["transform_init_seed"] = 0
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-structure", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--membership-seeds",
        default=",".join(map(str, DEFAULT_MEMBERSHIP_SEEDS)),
    )
    parser.add_argument("--extended-iters", type=int, default=128)
    parser.add_argument("--deep-iters", type=int, default=512)
    args = parser.parse_args()

    if args.extended_iters <= 0:
        raise ValueError("extended-iters must be positive")
    if args.deep_iters <= args.extended_iters:
        raise ValueError("deep-iters must exceed extended-iters")
    membership_seeds = parse_ints(args.membership_seeds)
    if not membership_seeds:
        raise ValueError("membership-seeds must be non-empty")

    reference = json.loads(args.base_structure.read_text())
    base_config = json.loads(args.base_config.read_text())
    structures_dir = args.out_dir / "structures"
    configs_dir = args.out_dir / "configs"

    structures: list[tuple[str, int, Path]] = []
    for variant, seed in (("learned", -1), ("contiguous", -1)):
        path = structures_dir / f"{variant}.json"
        write_json(path, make_structure(reference, variant, None))
        structures.append((variant, seed, path.resolve()))
    for seed in membership_seeds:
        name = f"random_{seed}"
        path = structures_dir / f"{name}.json"
        write_json(path, make_structure(reference, "random", seed))
        structures.append((name, seed, path.resolve()))

    protocols = (
        ("auto_stop", 0),
        (f"fixed_{args.extended_iters}", args.extended_iters),
        (f"fixed_{args.deep_iters}", args.deep_iters),
    )
    config_paths: dict[str, Path] = {}
    for protocol, transform_niter in protocols:
        path = configs_dir / f"{protocol}.json"
        write_json(path, make_config(base_config, transform_niter))
        config_paths[protocol] = path.resolve()

    cases: list[dict[str, object]] = []
    # Protocol-major ordering balances fixed-iteration cases across two shards.
    for protocol, transform_niter in protocols[:2]:
        for membership, membership_seed, structure_path in structures:
            cases.append(
                {
                    "case_id": f"{protocol}__{membership}",
                    "protocol": protocol,
                    "membership": membership,
                    "membership_seed": membership_seed,
                    "transform_niter": transform_niter,
                    "structure_path": str(structure_path),
                    "config_path": str(config_paths[protocol]),
                }
            )
    deep_protocol, deep_niter = protocols[2]
    for membership, membership_seed, structure_path in structures[:2]:
        cases.append(
            {
                "case_id": f"{deep_protocol}__{membership}",
                "protocol": deep_protocol,
                "membership": membership,
                "membership_seed": membership_seed,
                "transform_niter": deep_niter,
                "structure_path": str(structure_path),
                "config_path": str(config_paths[deep_protocol]),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "sift1M",
        "complete_payload_bits": 128,
        "tested_product_bits": reference["total_bits"],
        "tail_enabled": False,
        "target": "epq",
        "base_structure": str(args.base_structure.resolve()),
        "base_config": str(args.base_config.resolve()),
        "membership_seeds": membership_seeds,
        "extended_iterations": args.extended_iters,
        "deep_iterations": args.deep_iters,
        "case_count": len(cases),
        "cases": cases,
    }
    write_json(args.out_dir / "manifest.json", manifest)
    with (args.out_dir / "cases.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(cases[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(cases)

    print(json.dumps({"out_dir": str(args.out_dir), "case_count": len(cases)}))


if __name__ == "__main__":
    main()
