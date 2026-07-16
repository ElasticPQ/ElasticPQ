#!/usr/bin/env python3
"""Generate the long-horizon balanced final-codec M-grid matrix."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path


M_VALUES = tuple(range(11, 21))
ALLOCATIONS = ("dp", "equal")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


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
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--transform-niter", type=int, default=1024)
    args = parser.parse_args()

    base = json.loads(args.base_config.resolve().read_text())
    full_config = (
        args.out_dir / "configs" / f"fixed_{args.transform_niter}.json"
    ).resolve()
    smoke_config = (args.out_dir / "configs" / "smoke_2.json").resolve()
    write_json(full_config, make_config(base, args.transform_niter))
    smoke = make_config(base, 2)
    smoke_transform = smoke["index"]["transform"]
    smoke_transform["transform_max_train"] = 800
    smoke_transform["transform_max_eval"] = 200
    write_json(smoke_config, smoke)

    cases: list[dict[str, object]] = []
    for m in M_VALUES:
        for allocation in ALLOCATIONS:
            structure = (
                args.structure_dir.resolve()
                / f"grid_m{m}_{allocation}_bits.json"
            )
            if not structure.exists():
                raise FileNotFoundError(structure)
            payload = json.loads(structure.read_text())
            if len(payload["groups"]) != m or int(payload["total_bits"]) != 128:
                raise RuntimeError(f"invalid structure: {structure}")
            cases.append(
                {
                    "case_id": f"m{m}_{allocation}",
                    "m": m,
                    "allocation": allocation,
                    "transform_niter": args.transform_niter,
                    "structure_path": str(structure),
                    "config_path": str(full_config),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "cases.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(cases[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(cases)
    write_json(
        args.out_dir / "manifest.json",
        {
            "dataset": "sift1M",
            "bits": 128,
            "target": "epq",
            "tail_enabled": False,
            "m_values": list(M_VALUES),
            "allocations": list(ALLOCATIONS),
            "transform_niter": args.transform_niter,
            "transform_init_mode": "identity",
            "transform_seed": 123,
            "selection_metric": "disjoint final exact-codebook holdout MSE",
            "case_count": len(cases),
            "cases": cases,
        },
    )
    print(json.dumps({"out_dir": str(args.out_dir), "case_count": len(cases)}))


if __name__ == "__main__":
    main()
