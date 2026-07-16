#!/usr/bin/env python3
"""Generate the paired SIFT1M/128b descriptor/start interaction matrix."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path


CELLS = (
    "searched_searched",
    "searched_neutral",
    "balanced_searched",
    "balanced_neutral",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def descriptor(structure: dict) -> list[tuple[int, int]]:
    return sorted(
        (len(group["dims"]), int(group["nbits"]))
        for group in structure["groups"]
    )


def physical_permutation(structure: dict) -> list[int]:
    return [dim for group in structure["groups"] for dim in group["dims"]]


def make_config(base: dict, seed: int) -> dict:
    config = copy.deepcopy(base)
    config.setdefault("builder", {})["auto_reuse_structure"] = False
    transform = config.setdefault("index", {}).setdefault("transform", {})
    transform["transform_niter"] = 128
    transform["transform_seed"] = seed
    transform["transform_init_mode"] = "identity"
    transform["transform_init_seed"] = 0
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--searched-searched", type=Path, required=True)
    parser.add_argument("--searched-neutral", type=Path, required=True)
    parser.add_argument("--balanced-searched", type=Path, required=True)
    parser.add_argument("--balanced-neutral", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1001)
    parser.add_argument("--seed-count", type=int, default=10)
    args = parser.parse_args()

    if args.seed_count < 2:
        raise ValueError("seed-count must be at least two")
    structures = {
        "searched_searched": args.searched_searched.resolve(),
        "searched_neutral": args.searched_neutral.resolve(),
        "balanced_searched": args.balanced_searched.resolve(),
        "balanced_neutral": args.balanced_neutral.resolve(),
    }
    loaded = {name: json.loads(path.read_text()) for name, path in structures.items()}
    if {int(value["total_bits"]) for value in loaded.values()} != {128}:
        raise RuntimeError("all cells must use a 128-bit product payload")
    if descriptor(loaded["searched_searched"]) != descriptor(loaded["searched_neutral"]):
        raise RuntimeError("searched cells do not share a descriptor")
    if descriptor(loaded["balanced_searched"]) != descriptor(loaded["balanced_neutral"]):
        raise RuntimeError("balanced cells do not share a descriptor")
    searched_physical = physical_permutation(loaded["searched_searched"])
    if physical_permutation(loaded["balanced_searched"]) != searched_physical:
        raise RuntimeError("searched-start cells do not share physical W0")
    identity = list(range(128))
    if physical_permutation(loaded["searched_neutral"]) != identity:
        raise RuntimeError("searched-neutral cell is not identity ordered")
    if physical_permutation(loaded["balanced_neutral"]) != identity:
        raise RuntimeError("balanced-neutral cell is not identity ordered")
    if searched_physical == identity:
        raise RuntimeError("searched physical start unexpectedly equals identity")

    base = json.loads(args.base_config.read_text())
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    config_paths: dict[int, Path] = {}
    for seed in seeds:
        path = (args.out_dir / "configs" / f"fixed_128_seed_{seed}.json").resolve()
        write_json(path, make_config(base, seed))
        config_paths[seed] = path

    cases: list[dict[str, object]] = []
    for seed in seeds:
        for cell in CELLS:
            cases.append(
                {
                    "case_id": f"seed_{seed}__{cell}",
                    "cell": cell,
                    "transform_seed": seed,
                    "structure_path": str(structures[cell]),
                    "config_path": str(config_paths[seed]),
                }
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "sift1M",
        "bits": 128,
        "target": "epq",
        "tail_enabled": False,
        "protocol": "128 capped-proxy updates plus one exact-polish update",
        "seeds": seeds,
        "cells": list(CELLS),
        "structures": {name: str(path) for name, path in structures.items()},
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
