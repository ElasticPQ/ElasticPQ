#!/usr/bin/env python3
"""Generate the SIFT1M/128b coupled descriptor/orientation path matrix."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path


PATHS = (
    "searched_pair",
    "searched_descriptor_neutral_start",
    "balanced_equal_pair",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def descriptor_pairs(structure: dict) -> list[tuple[int, int]]:
    return sorted(
        (len(group["dims"]), int(group["nbits"]))
        for group in structure["groups"]
    )


def physical_permutation(structure: dict) -> list[int]:
    return [dim for group in structure["groups"] for dim in group["dims"]]


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
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--searched-structure", type=Path, required=True)
    parser.add_argument("--neutral-searched-descriptor", type=Path, required=True)
    parser.add_argument("--balanced-equal-structure", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--extended-iters", type=int, default=128)
    parser.add_argument("--deep-iters", type=int, default=512)
    parser.add_argument("--terminal-iters", type=int, default=0)
    args = parser.parse_args()

    if args.terminal_iters and args.terminal_iters <= args.deep_iters:
        raise ValueError("terminal-iters must exceed deep-iters")

    structures = {
        "searched_pair": args.searched_structure.resolve(),
        "searched_descriptor_neutral_start": args.neutral_searched_descriptor.resolve(),
        "balanced_equal_pair": args.balanced_equal_structure.resolve(),
    }
    loaded = {name: json.loads(path.read_text()) for name, path in structures.items()}
    if {int(value["total_bits"]) for value in loaded.values()} != {128}:
        raise RuntimeError("all structures must use a 128-bit product payload")
    if descriptor_pairs(loaded["searched_pair"]) != descriptor_pairs(
        loaded["searched_descriptor_neutral_start"]
    ):
        raise RuntimeError("searched and neutral-start structures do not share a descriptor")
    identity = list(range(128))
    if physical_permutation(loaded["searched_pair"]) == identity:
        raise RuntimeError("searched pair unexpectedly has identity membership")
    if physical_permutation(loaded["searched_descriptor_neutral_start"]) != identity:
        raise RuntimeError("neutral searched descriptor must have identity membership")
    if physical_permutation(loaded["balanced_equal_pair"]) != identity:
        raise RuntimeError("balanced/equal pair must have identity membership")

    base_config = json.loads(args.base_config.read_text())
    protocols = [
        ("auto_stop", 0),
        (f"fixed_{args.extended_iters}", args.extended_iters),
        (f"fixed_{args.deep_iters}", args.deep_iters),
    ]
    if args.terminal_iters:
        protocols.append((f"fixed_{args.terminal_iters}", args.terminal_iters))
    config_paths: dict[str, Path] = {}
    for protocol, niter in protocols:
        path = (args.out_dir / "configs" / f"{protocol}.json").resolve()
        write_json(path, make_config(base_config, niter))
        config_paths[protocol] = path

    cases: list[dict[str, object]] = []
    # Protocol-major and path-minor ordering makes modulo-three shards path-stable.
    for protocol, niter in protocols[:3]:
        for path_name in PATHS:
            cases.append(
                {
                    "case_id": f"{protocol}__{path_name}",
                    "protocol": protocol,
                    "path_name": path_name,
                    "transform_niter": niter,
                    "structure_path": str(structures[path_name]),
                    "config_path": str(config_paths[protocol]),
                }
            )
    if args.terminal_iters:
        protocol, niter = protocols[-1]
        for path_name in ("searched_pair", "balanced_equal_pair"):
            cases.append(
                {
                    "case_id": f"{protocol}__{path_name}",
                    "protocol": protocol,
                    "path_name": path_name,
                    "transform_niter": niter,
                    "structure_path": str(structures[path_name]),
                    "config_path": str(config_paths[protocol]),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "sift1M",
        "bits": 128,
        "target": "epq",
        "tail_enabled": False,
        "base_config": str(args.base_config.resolve()),
        "extended_iterations": args.extended_iters,
        "deep_iterations": args.deep_iters,
        "terminal_iterations": args.terminal_iters,
        "paths": list(PATHS),
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
