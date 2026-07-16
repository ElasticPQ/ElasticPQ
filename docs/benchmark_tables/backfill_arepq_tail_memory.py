#!/usr/bin/env python3
"""Backfill only AREPQ tail-memory columns from fixed-structure train-only logs.

This intentionally preserves every pre-existing benchmark value, including the
row timestamp, timings, peak RSS, retrieval metrics, and serialized index size.
The fixed-structure reruns are evidence only for exact memory component counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MEMORY_FIELDS = [
    "tail_payload_bits_per_vector",
    "tail_serialized_codebook_bytes",
    "tail_reconstruction_codebook_bytes",
    "tail_transform_copy_bytes",
    "product_tail_table_bytes",
    "tail_pair_table_bytes",
    "tail_norm_table_bytes",
    "tail_query_lut_bytes_per_query",
    "tail_resident_search_model_bytes",
    "tail_resident_auxiliary_table_bytes",
    "tail_resident_model_bytes",
    "memory_backfill_structure",
    "memory_backfill_source",
]
EXPECTED_CASES = {
    ("sift1M", 64),
    ("sift1M", 128),
    ("gist1M", 64),
    ("gist1M", 128),
    ("deep10M", 64),
    ("deep10M", 128),
}


def parse_json_line(text: str, prefix: str) -> dict:
    matches = re.findall(rf"^{re.escape(prefix)} (.+)$", text, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f"expected one {prefix} line, found {len(matches)}")
    value = json.loads(matches[0])
    if not isinstance(value, dict):
        raise RuntimeError(f"{prefix} payload is not an object")
    return value


def parse_log(path: Path) -> tuple[tuple[str, int], dict[str, str]]:
    text = path.read_text(errors="replace")
    if not re.search(r"^\s*Exit status:\s*0\s*$", text, re.MULTILINE):
        raise RuntimeError(f"non-zero or missing exit status: {path}")
    run = parse_json_line(text, "meta.run")
    dataset = parse_json_line(text, "meta.dataset")
    method = parse_json_line(text, "meta.method")
    if run.get("train_only") is not True or int(run.get("maxtrain", -1)) != 0:
        raise RuntimeError(f"memory backfill requires --maxtrain=0 --train-only: {path}")
    if method.get("family") != "arepq":
        raise RuntimeError(f"memory backfill log is not AREPQ: {path}")
    main_builder = method.get("main", {}).get("builder", {})
    if main_builder.get("name") != "FixedStructureBuilder":
        raise RuntimeError(f"memory backfill did not use FixedStructureBuilder: {path}")

    structure_match = re.search(r"^\s*structure time:\s*([0-9.eE+-]+)\s+s\s*$", text, re.MULTILINE)
    if structure_match is None or abs(float(structure_match.group(1))) > 5e-4:
        raise RuntimeError(f"memory backfill has non-zero structure time: {path}")

    memory = method.get("tail_memory")
    if not isinstance(memory, dict):
        raise RuntimeError(f"missing method.tail_memory: {path}")
    required = {
        "serialized_codebook_bytes",
        "product_tail_table_bytes",
        "tail_pair_table_bytes",
        "norm_table_bytes",
        "query_lut_bytes_per_query",
        "reconstruction_codebook_bytes",
    }
    missing = sorted(required - memory.keys())
    if missing:
        raise RuntimeError(f"missing tail-memory fields {missing}: {path}")

    serialized_codebook = int(memory["serialized_codebook_bytes"])
    product_tail = int(memory["product_tail_table_bytes"])
    tail_pair = int(memory["tail_pair_table_bytes"])
    norms = int(memory["norm_table_bytes"])

    def memory_value(current_name: str, legacy_name: str) -> int:
        if current_name in memory:
            return int(memory[current_name])
        if legacy_name in memory:
            return int(memory[legacy_name])
        raise RuntimeError(
            f"missing tail-memory field {current_name} (or {legacy_name}): {path}"
        )

    auxiliary = memory_value(
        "resident_auxiliary_table_bytes", "resident_auxiliary_bytes"
    )
    search = memory_value("resident_search_model_bytes", "resident_search_bytes")
    reconstruction = int(memory["reconstruction_codebook_bytes"])
    transform_copy = int(
        memory.get("transform_copy_bytes", int(dataset["dim"]) ** 2 * 4)
    )
    if "resident_model_bytes" in memory:
        total = int(memory["resident_model_bytes"])
    elif "resident_total_bytes" in memory:
        total = int(memory["resident_total_bytes"]) + transform_copy
    else:
        raise RuntimeError(f"missing resident model byte count: {path}")
    if auxiliary != product_tail + tail_pair + norms:
        raise RuntimeError(f"inconsistent resident auxiliary byte count: {path}")
    if search != serialized_codebook + auxiliary:
        raise RuntimeError(f"inconsistent resident search byte count: {path}")
    if total != search + reconstruction + transform_copy:
        raise RuntimeError(f"inconsistent resident total byte count: {path}")

    dataset_name = str(dataset["name"])
    bits = int(run["bits"])
    key = (dataset_name, bits)
    values = {
        "tail_payload_bits_per_vector": str(
            int(method["tail_bits"]) * int(method["tail_stages"])
        ),
        "tail_serialized_codebook_bytes": str(serialized_codebook),
        "tail_reconstruction_codebook_bytes": str(reconstruction),
        "tail_transform_copy_bytes": str(transform_copy),
        "product_tail_table_bytes": str(product_tail),
        "tail_pair_table_bytes": str(tail_pair),
        "tail_norm_table_bytes": str(norms),
        "tail_query_lut_bytes_per_query": str(int(memory["query_lut_bytes_per_query"])),
        "tail_resident_search_model_bytes": str(search),
        "tail_resident_auxiliary_table_bytes": str(auxiliary),
        "tail_resident_model_bytes": str(total),
        "memory_backfill_structure": str(run.get("epq_structure", "")),
        "memory_backfill_source": str(path),
    }
    return key, values


def load_memory_logs(log_dir: Path) -> dict[tuple[str, int], dict[str, str]]:
    selected: dict[tuple[str, int], dict[str, str]] = {}
    for path in sorted(log_dir.glob("*.log")):
        key, values = parse_log(path)
        if key in selected:
            raise RuntimeError(f"duplicate memory backfill case {key} in {log_dir}")
        selected[key] = values
    missing = sorted(EXPECTED_CASES - selected.keys())
    extra = sorted(selected.keys() - EXPECTED_CASES)
    if missing or extra:
        raise RuntimeError(f"memory log coverage mismatch: missing={missing}, extra={extra}")
    return selected


def update_csv(
    path: Path,
    memory_by_case: dict[tuple[str, int], dict[str, str]],
    dry_run: bool,
) -> int:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        original_fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")

    non_memory_fields = [name for name in original_fields if name not in MEMORY_FIELDS]
    before = [{name: row.get(name, "") for name in non_memory_fields} for row in rows]
    touched_cases: set[tuple[str, int]] = set()
    updated_rows = 0
    for row in rows:
        key = (row["dataset"], int(float(row["budget_b"])))
        values = memory_by_case.get(key)
        if values is None:
            continue
        row.update(values)
        touched_cases.add(key)
        updated_rows += 1

    if touched_cases != EXPECTED_CASES:
        raise RuntimeError(
            f"CSV coverage mismatch for {path}: touched={sorted(touched_cases)}"
        )
    after = [{name: row.get(name, "") for name in non_memory_fields} for row in rows]
    if before != after:
        raise RuntimeError(f"non-memory CSV fields changed unexpectedly: {path}")

    fieldnames = original_fields + [name for name in MEMORY_FIELDS if name not in original_fields]
    if not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    return updated_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-log-dir", type=Path, default=None)
    parser.add_argument("--ivf-log-dir", type=Path, required=True)
    parser.add_argument("--flat-csv", type=Path, default=ROOT / "AREPQ.csv")
    parser.add_argument("--ivf-csv", type=Path, default=ROOT / "IVF-AREPQ.csv")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ivf_memory = load_memory_logs(args.ivf_log_dir)
    ivf_rows = update_csv(args.ivf_csv, ivf_memory, args.dry_run)
    action = "validated" if args.dry_run else "updated"
    if args.flat_log_dir is not None:
        flat_memory = load_memory_logs(args.flat_log_dir)
        flat_rows = update_csv(args.flat_csv, flat_memory, args.dry_run)
        print(f"{action} memory fields only: {args.flat_csv} rows={flat_rows}")
    print(f"{action} memory fields only: {args.ivf_csv} rows={ivf_rows}")


if __name__ == "__main__":
    main()
