#!/usr/bin/env python3
"""Generate fixed-architecture membership/rotation controls for SIFT1M 128b."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path


DEFAULT_MEMBERSHIP_SEEDS = (101, 202, 303, 404, 505)
DEFAULT_HAAR_SEEDS = (1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010)


def parse_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate_architecture(reference: dict, candidate: dict) -> None:
    assert candidate["d"] == reference["d"]
    assert candidate["total_bits"] == reference["total_bits"]
    assert len(candidate["groups"]) == len(reference["groups"])
    seen: list[int] = []
    for ref_group, group in zip(reference["groups"], candidate["groups"]):
        assert len(group["dims"]) == len(ref_group["dims"])
        assert group["nbits"] == ref_group["nbits"]
        seen.extend(group["dims"])
    assert sorted(seen) == list(range(reference["d"]))


def make_structure(reference: dict, variant: str, seed: int | None) -> dict:
    structure = copy.deepcopy(reference)
    sizes = [len(group["dims"]) for group in reference["groups"]]
    if variant == "learned":
        flat_dims = [dim for group in reference["groups"] for dim in group["dims"]]
    elif variant == "contiguous":
        flat_dims = list(range(reference["d"]))
    elif variant == "random":
        if seed is None:
            raise ValueError("random membership requires a seed")
        flat_dims = list(range(reference["d"]))
        random.Random(seed).shuffle(flat_dims)
    else:
        raise ValueError(f"unknown membership variant: {variant}")

    offset = 0
    for group, size in zip(structure["groups"], sizes):
        group["dims"] = flat_dims[offset : offset + size]
        offset += size
    structure["meta"] = {
        "builder": "FixedStructureBuilder",
        "membership_control": {
            "variant": variant,
            "seed": seed,
            "parent_total_bits": reference["total_bits"],
            "group_sizes": sizes,
            "group_bits": [group["nbits"] for group in reference["groups"]],
        },
    }
    validate_architecture(reference, structure)
    return structure


def make_config(base: dict, init_mode: str, init_seed: int) -> dict:
    config = copy.deepcopy(base)
    config.setdefault("builder", {})["auto_reuse_structure"] = False
    transform = config.setdefault("index", {}).setdefault("transform", {})
    transform["transform_seed"] = 123
    transform["transform_init_mode"] = init_mode
    transform["transform_init_seed"] = init_seed
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
    parser.add_argument(
        "--haar-seeds",
        default=",".join(map(str, DEFAULT_HAAR_SEEDS)),
    )
    args = parser.parse_args()

    reference = json.loads(args.base_structure.read_text())
    base_config = json.loads(args.base_config.read_text())
    membership_seeds = parse_ints(args.membership_seeds)
    haar_seeds = parse_ints(args.haar_seeds)
    if not membership_seeds or not haar_seeds:
        raise ValueError("membership and Haar seed lists must be non-empty")

    structures_dir = args.out_dir / "structures"
    configs_dir = args.out_dir / "configs"
    structures: list[tuple[str, int | None, Path]] = []
    for variant, seed in [("learned", None), ("contiguous", None)]:
        path = structures_dir / f"{variant}.json"
        write_json(path, make_structure(reference, variant, seed))
        structures.append((variant, seed, path.resolve()))
    for seed in membership_seeds:
        name = f"random_{seed}"
        path = structures_dir / f"{name}.json"
        write_json(path, make_structure(reference, "random", seed))
        structures.append((name, seed, path.resolve()))

    config_paths: dict[tuple[str, int], Path] = {}
    for mode, seeds in [
        ("identity", [0]),
        ("haar_r", haar_seeds),
        ("matched_physical", haar_seeds[:3]),
    ]:
        for seed in seeds:
            path = configs_dir / f"{mode}_{seed}.json"
            write_json(path, make_config(base_config, mode, seed))
            config_paths[(mode, seed)] = path.resolve()

    cases: list[dict[str, object]] = []
    for membership, membership_seed, structure_path in structures:
        cases.append(
            {
                "case_id": f"identity__{membership}",
                "protocol": "production_identity",
                "membership": membership,
                "membership_seed": membership_seed if membership_seed is not None else -1,
                "init_mode": "identity",
                "init_seed": 0,
                "structure_path": str(structure_path),
                "config_path": str(config_paths[("identity", 0)]),
            }
        )
        for seed in haar_seeds:
            cases.append(
                {
                    "case_id": f"haar_r_{seed}__{membership}",
                    "protocol": "literal_haar_r",
                    "membership": membership,
                    "membership_seed": membership_seed if membership_seed is not None else -1,
                    "init_mode": "haar_r",
                    "init_seed": seed,
                    "structure_path": str(structure_path),
                    "config_path": str(config_paths[("haar_r", seed)]),
                }
            )

    matched_memberships = {"learned", "contiguous", f"random_{membership_seeds[0]}"}
    for membership, membership_seed, structure_path in structures:
        if membership not in matched_memberships:
            continue
        for seed in haar_seeds[:3]:
            cases.append(
                {
                    "case_id": f"matched_{seed}__{membership}",
                    "protocol": "matched_physical",
                    "membership": membership,
                    "membership_seed": membership_seed if membership_seed is not None else -1,
                    "init_mode": "matched_physical",
                    "init_seed": seed,
                    "structure_path": str(structure_path),
                    "config_path": str(config_paths[("matched_physical", seed)]),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "sift1M",
        "total_bits": 128,
        "main_bits": reference["total_bits"],
        "base_structure": str(args.base_structure.resolve()),
        "base_config": str(args.base_config.resolve()),
        "membership_seeds": membership_seeds,
        "haar_seeds": haar_seeds,
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
