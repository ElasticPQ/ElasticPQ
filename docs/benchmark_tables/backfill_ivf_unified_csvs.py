#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


CSV_DIR = Path(__file__).resolve().parent
LOG_ROOT = CSV_DIR / "logs"
LEGACY_SERIAL_LOG_DIR = CSV_DIR / "logs" / "unified_topk1000_fullsplits_serial_20260618"


def discover_latest_dir(patterns, fallback):
    matches = []
    for pattern in patterns:
        for path in LOG_ROOT.glob(pattern):
            if path.is_dir():
                matches.append(path)
    if not matches:
        return fallback
    return sorted(matches, key=lambda path: (path.stat().st_mtime, path.name))[-1]


DEFAULT_SERIAL_LOG_DIR = discover_latest_dir(
    ["joint_topk1000_fullsplits_serial*", "unified_topk1000_fullsplits_serial*"],
    fallback=LEGACY_SERIAL_LOG_DIR,
)

CSV_TARGETS = {
    "IVF+PQ": {
        "path": CSV_DIR / "IVF-PQ.csv",
        "impl_backend": lambda bits: f"faiss:IVF+PQ M={bits // 8} nbits=8",
    },
    "IVF+OPQ": {
        "path": CSV_DIR / "IVF-OPQ.csv",
        "impl_backend": lambda bits: f"faiss:IVF+OPQ M={bits // 8} nbits=8",
    },
    "IVF+DPOPQ": {
        "path": CSV_DIR / "IVF-DP-OPQ.csv",
        "impl_backend": lambda bits: f"local:IVF+DP-OPQ PCA+DP M={bits // 8} nbits=8",
    },
    "IVF+BAPQ": {
        "path": CSV_DIR / "IVF-BAPQ.csv",
        "impl_backend": lambda bits: "cpp:IVF+BAPQ payload=IndexBAPQ q=4 bmax=12 km=50x3",
    },
    "IVF+VAQ": {
        "path": CSV_DIR / "IVF-VAQ.csv",
        "impl_backend": lambda bits: "TheDatumOrg/VAQ IVF residual ADC",
    },
    "IVF+EPQ": {
        "path": CSV_DIR / "IVF-EPQ.csv",
        "impl_backend": lambda bits: "cpp:IVF+EPQ payload=IndexEPQ",
    },
    "IVF+AREPQ": {
        "path": CSV_DIR / "IVF-AREPQ.csv",
        "impl_backend": lambda bits: "cpp:IVF+AREPQ payload=IndexAREPQ main=IndexEPQ tail=8b",
    },
    "IVF+RaBitQ": {
        "path": CSV_DIR / "IVF-RaBitQ.csv",
        "impl_backend": lambda bits: "faiss:IVF+RaBitQ payload=IndexRaBitQ nb_bits=1",
    },
    "IVF+RQ": {
        "path": CSV_DIR / "IVF-RQ.csv",
        "impl_backend": lambda bits: f"faiss:IVF+RQ M={bits // 8 - 1} nbits=8",
    },
    "IVF+LSQ": {
        "path": CSV_DIR / "IVF-LSQ.csv",
        "impl_backend": lambda bits: f"faiss:IVF+LSQ M={bits // 8 - 1} nbits=8",
    },
}

DATASET_ORDER = {"sift1M": 0, "gist1M": 1, "deep10M": 2}

FIELDNAMES = [
    "timestamp",
    "dataset",
    "d",
    "nb",
    "nq",
    "budget_b",
    "nlist",
    "nprobe",
    "topk",
    "rerank_depth",
    "train_rows",
    "threads",
    "impl_backend",
    "coarse_train_s",
    "coarse_add_s",
    "coarse_assign_s",
    "avg_candidates_per_q",
    "candidate_hit_rate",
    "structure_s",
    "prep_s",
    "codebook_s",
    "train_total_s",
    "peak_ram_gb",
    "add_encode_s",
    "encode_us_per_vec",
    "rerank_total_s",
    "search_total_s",
    "search_ms_per_q",
    "qps",
    "recall_1",
    "recall_10",
    "recall_100",
    "recall_1000",
    "overlap_1000",
    "J",
    "index_size_mb",
    "notes",
]


def fmt_float(value: float, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "N/A"
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def fmt_int(value: int) -> str:
    return str(int(value))


def parse_peak_ram_gb(log_text: str) -> float:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", log_text)
    if not match:
        raise RuntimeError("failed to find Maximum resident set size in log")
    kb = int(match.group(1))
    return kb * 1024.0 / 1e9


def parse_kv_line(line: str) -> dict:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", line))


def parse_header(log_text: str) -> dict:
    match = re.search(r"^dataset=.*$", log_text, re.MULTILINE)
    if not match:
        return {}
    fields = parse_kv_line(match.group(0))
    out = {}
    for key in (
        "dataset",
        "target",
        "refine",
        "rerank_depth",
    ):
        if key in fields:
            out[key] = fields[key]
    for key in (
        "d",
        "nb",
        "nq",
        "nt",
        "bits",
        "nlist",
        "nprobe",
        "topk",
        "metric_topk",
        "recon_sample",
        "threads",
        "train_limit",
        "base_limit",
        "query_limit",
        "base_batch_size",
        "coarse_kmeans_niter",
        "coarse_kmeans_nredo",
    ):
        if key in fields:
            out[key] = int(fields[key])
    if "refine_k_factor" in fields:
        out["refine_k_factor"] = float(fields["refine_k_factor"])
    return out


def parse_script_metadata(log_text: str) -> dict:
    values = {}
    for line in log_text.splitlines():
        if not line.startswith("script."):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        values[key[7:]] = value.strip()
    return values


def compact_text(value):
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def compact_path(value):
    text = compact_text(value)
    return Path(text).name if text else None


def canonical_target_name(name: str) -> str:
    if name.endswith("+RefineFlat"):
        return name[: -len("+RefineFlat")]
    return name


def first_int(*values):
    for value in values:
        if value is None or value == "":
            continue
        return int(value)
    return None


def resolve_timestamp(log_path: Path, metadata: dict, script_meta: dict) -> str:
    hardware_meta = metadata.get("hardware", {})
    for value in (hardware_meta.get("timestamp_utc"), script_meta.get("start_utc")):
        text = compact_text(value)
        if text:
            return text
    return (
        datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def resolve_threads(payload: dict, metadata: dict, header: dict, script_meta: dict) -> str:
    thread_meta = metadata.get("threads", {})
    value = first_int(
        thread_meta.get("requested_threads"),
        thread_meta.get("effective_threads"),
        header.get("threads"),
        script_meta.get("threads"),
        payload.get("threads"),
        12,
    )
    return fmt_int(value)


def summarize_builder(builder: dict) -> str | None:
    if not builder:
        return None
    builder_type = compact_text(builder.get("type") or builder.get("name")) or "unknown"
    flags = []
    for key, label in (
        ("use_grow", "grow"),
        ("use_crystallize", "cryst"),
        ("use_mbeam", "mbeam"),
        ("use_greedy_tail", "greedy"),
        ("use_chain_tail", "chain"),
    ):
        if builder.get(key) is True:
            flags.append(label)
    parts = [builder_type]
    if flags:
        parts.append("+".join(flags))
    proxy_train = builder.get("proxy_max_train_rows")
    proxy_eval = builder.get("proxy_max_eval_rows")
    if proxy_train is not None or proxy_eval is not None:
        parts.append(f"proxy={proxy_train}/{proxy_eval}")
    if builder.get("proxy_kmeans_niter") is not None:
        parts.append(
            f"km={builder.get('proxy_kmeans_niter')}x{builder.get('proxy_kmeans_nredo', 1)}"
        )
    if builder.get("crystallize_proxy_bits") is not None:
        parts.append(f"proxy_bits={builder.get('crystallize_proxy_bits')}")
    return ",".join(parts)


def summarize_method_cfg(method_meta: dict) -> str | None:
    if not method_meta:
        return None
    family = compact_text(method_meta.get("family")) or "unknown"
    parts = [f"family={family}"]

    if family in {"pq", "opq"}:
        for key in ("M", "nbits", "d2", "total_bits"):
            if method_meta.get(key) is not None:
                parts.append(f"{key}={method_meta.get(key)}")
        opq = method_meta.get("opq")
        if isinstance(opq, dict):
            parts.append(
                "opq="
                f"{opq.get('niter')}/{opq.get('niter_pq')}/{opq.get('niter_pq_0')}"
            )
            if opq.get("max_train_points") is not None:
                parts.append(f"opq_max_train={opq.get('max_train_points')}")
    elif family == "rabitq":
        for key in ("nb_bits", "nominal_budget_bits", "effective_budget_bits"):
            if method_meta.get(key) is not None:
                parts.append(f"{key}={method_meta.get(key)}")
    elif family in {"rq", "lsq"}:
        for key in ("M", "nbits", "total_bits", "effective_budget_bits", "search_type"):
            if method_meta.get(key) is not None:
                parts.append(f"{key}={method_meta.get(key)}")
        if family == "rq":
            for key in ("max_beam_size", "train_type", "use_beam_LUT"):
                if method_meta.get(key) is not None:
                    parts.append(f"{key}={method_meta.get(key)}")
        else:
            for key in ("train_iters", "encode_ils_iters", "train_ils_iters", "icm_iters"):
                if method_meta.get(key) is not None:
                    parts.append(f"{key}={method_meta.get(key)}")
    elif family == "dpopq":
        for key in (
            "M",
            "nbits",
            "total_bits",
            "partition_cost",
            "block_alignment",
            "partition_units_exact",
            "partition_units_scale",
            "partition_units_sum",
        ):
            if method_meta.get(key) is not None:
                parts.append(f"{key}={method_meta.get(key)}")
        group_dims = method_meta.get("group_dims")
        if isinstance(group_dims, list):
            parts.append("group_dims=" + "/".join(str(value) for value in group_dims))
        native_index = compact_text(method_meta.get("native_index"))
        if native_index:
            parts.append(f"native={native_index}")
        parts.append("source=no public reference implementation available")
    elif family == "bapq":
        if method_meta.get("subspace_dim") is not None:
            parts.append(f"q={method_meta.get('subspace_dim')}")
        for key in ("bmax", "max_train_rows", "query_batch", "db_chunk"):
            if method_meta.get(key) is not None:
                parts.append(f"{key}={method_meta.get(key)}")
        if method_meta.get("kmeans_niter") is not None:
            parts.append(
                f"km={method_meta.get('kmeans_niter')}x{method_meta.get('kmeans_nredo', 1)}"
            )
    elif family == "vaq":
        for key in (
            "total_bits",
            "subspaces",
            "subspace_dim",
            "min_bits_per_subspace",
            "max_bits_per_subspace",
            "variance_fraction",
            "upstream_commit",
        ):
            if method_meta.get(key) is not None:
                parts.append(f"{key}={method_meta.get(key)}")
        allocation = method_meta.get("bit_allocation")
        if isinstance(allocation, list):
            parts.append("bit_allocation=" + "/".join(str(value) for value in allocation))
    elif family in {"epq", "repq"}:
        if method_meta.get("total_bits") is not None:
            parts.append(f"total_bits={method_meta.get('total_bits')}")
        if method_meta.get("kmeans_niter") is not None:
            parts.append(
                f"km={method_meta.get('kmeans_niter')}x{method_meta.get('kmeans_nredo', 1)}"
            )
        parts.append(
            "transform="
            f"{'on' if method_meta.get('use_uneven_transform') else 'off'}"
        )
        if method_meta.get("use_uneven_transform"):
            parts.append(
                "transform_cfg="
                f"{method_meta.get('transform_niter')}/"
                f"{method_meta.get('transform_kmeans_niter')}x"
                f"{method_meta.get('transform_kmeans_nredo', 1)}"
            )
            if method_meta.get("transform_max_train") is not None:
                parts.append(
                    "transform_rows="
                    f"{method_meta.get('transform_max_train')}/"
                    f"{method_meta.get('transform_max_eval')}"
                )
            if method_meta.get("transform_proxy_max_bits") is not None:
                parts.append(
                    f"transform_proxy_bits={method_meta.get('transform_proxy_max_bits')}"
                )
            if method_meta.get("transform_exact_polish_iters") is not None:
                parts.append(
                    f"transform_exact={method_meta.get('transform_exact_polish_iters')}"
                )
        builder_summary = summarize_builder(method_meta.get("builder", {}))
        if builder_summary:
            parts.append(f"builder={builder_summary}")
    elif family == "avq":
        for key in (
            "effective_bits",
            "dimensions_per_block",
            "default_num_neighbors",
            "training_threads",
            "search_threads",
            "search_batch_size",
        ):
            if method_meta.get(key) is not None:
                parts.append(f"{key}={method_meta.get(key)}")
        if method_meta.get("anisotropic_quantization_threshold") is not None:
            parts.append(
                "aq_threshold="
                f"{fmt_float(method_meta.get('anisotropic_quantization_threshold'), 3)}"
            )
    else:
        for key in ("total_bits", "M", "nbits"):
            if method_meta.get(key) is not None:
                parts.append(f"{key}={method_meta.get(key)}")

    query_cfg = method_meta.get("main")
    if not isinstance(query_cfg, dict):
        query_cfg = method_meta
    if query_cfg.get("ivf_query_weighted_sampling") is not None:
        enabled = bool(query_cfg["ivf_query_weighted_sampling"])
        parts.append(f"ivf_query_weighted_sampling={str(enabled).lower()}")

    return ",".join(parts)


def summarize_env(metadata: dict, script_meta: dict) -> str | None:
    hardware = metadata.get("hardware", {})
    build = metadata.get("build", {})
    parts = []
    cpu = compact_text(hardware.get("cpu_model"))
    if cpu:
        parts.append(f"cpu={cpu}")
    cpuset = compact_text(hardware.get("cpuset") or script_meta.get("cpuset"))
    if cpuset:
        parts.append(f"cpuset={cpuset}")
    mems = compact_text(hardware.get("mems_allowed"))
    if mems:
        parts.append(f"mems={mems}")
    faiss_target = compact_text(build.get("faiss_target"))
    if faiss_target:
        parts.append(f"faiss_target={faiss_target}")
    simd = compact_text(build.get("faiss_simd_hint") or build.get("compile_simd"))
    if simd:
        parts.append(f"simd={simd}")
    compiler_id = compact_text(build.get("compiler_id"))
    compiler_version = compact_text(build.get("compiler_version"))
    if compiler_id or compiler_version:
        parts.append(f"compiler={compiler_id or 'unknown'}-{compiler_version or 'unknown'}")
    return ", ".join(parts) if parts else None


def build_notes(
    stem: str,
    payload: dict,
    target: dict,
    metadata: dict,
    method_meta: dict,
    header: dict,
    script_meta: dict,
) -> str:
    run_meta = metadata.get("run", {})
    config_name = compact_path(
        run_meta.get("config_path") or script_meta.get("config")
    ) or "epq_train_refined.json"
    env_summary = summarize_env(metadata, script_meta)
    method_cfg = summarize_method_cfg(method_meta)
    run_tag = compact_text(script_meta.get("run_tag"))
    rerank_depth = (
        compact_text(run_meta.get("rerank_depth"))
        or compact_text(header.get("rerank_depth"))
        or "all"
    )
    topk = payload.get("topk") or header.get("topk") or run_meta.get("topk") or 1000
    metric_topk = (
        payload.get("metric_topk")
        or header.get("metric_topk")
        or run_meta.get("metric_topk")
        or topk
    )
    recon_sample = (
        payload.get("recon_sample")
        or header.get("recon_sample")
        or run_meta.get("recon_sample")
        or "min(200000,nb)"
    )
    coarse_niter = (
        header.get("coarse_kmeans_niter")
        or run_meta.get("coarse_kmeans_niter")
        or metadata.get("config", {}).get("index", {}).get("kmeans_niter")
    )
    coarse_nredo = (
        header.get("coarse_kmeans_nredo")
        or run_meta.get("coarse_kmeans_nredo")
        or metadata.get("config", {}).get("index", {}).get("kmeans_nredo")
        or 1
    )
    parts = [
        f"config={config_name}",
        "coarse protocol fixed across methods",
        "splits=train/full query/full",
        f"topk={topk}",
        f"metric_topk={metric_topk}",
        f"rerank_depth={rerank_depth}",
        f"recon_sample={recon_sample}",
        "peak_ram_gb=/usr/bin/time -v maxrss",
        "index_size_mb=serialized bytes /(1024^2)",
        f"log_stem={stem}",
    ]
    if coarse_niter is not None:
        parts.append(f"coarse_kmeans={coarse_niter}x{coarse_nredo}")
    if run_tag:
        parts.append(f"run_tag={run_tag}")
    if env_summary:
        parts.append(f"env={env_summary}")
    if method_cfg:
        parts.append(f"method_cfg={method_cfg}")
    if payload.get("refine") or target.get("refine_time", 0.0):
        parts.append(
            f"refine=IndexRefineFlat k_factor={payload.get('refine_k_factor', run_meta.get('refine_k_factor', 1))}"
        )
    if canonical_target_name(target["name"]) == "IVF+RaBitQ":
        parts.append(
            "RaBitQ nominal budget labels follow the shared 64/128-bit benchmark convention; actual footprint is captured by index_size_mb"
        )
    if canonical_target_name(target["name"]) == "IVF+DPOPQ":
        parts.append(
            "in-repository DP-OPQ implementation; no public reference implementation available; IVF residual ADC uses the shared coarse protocol"
        )
    return "; ".join(parts)


def iter_result_files(log_dir: Path):
    for json_path in sorted(log_dir.glob("joint_*.json")):
        stem = json_path.stem
        log_path = log_dir / f"{stem}.log"
        if not log_path.exists():
            raise RuntimeError(f"missing log for {json_path}")
        yield json_path, log_path


def detect_available_targets(log_dir: Path) -> list[str]:
    seen = set()
    for json_path, _ in iter_result_files(log_dir):
        with json_path.open() as f:
            payload = json.load(f)
        for target in payload.get("targets", []):
            name = canonical_target_name(target["name"])
            if name in CSV_TARGETS:
                seen.add(name)
    return [name for name in CSV_TARGETS if name in seen]


def merge_rows(rows_by_target: dict, new_rows_by_target: dict):
    for target, rows in new_rows_by_target.items():
        rows_by_target[target].extend(rows)


def load_rows(log_dir: Path, selected_targets: set[str]):
    rows_by_target = {name: [] for name in CSV_TARGETS}
    for json_path, log_path in iter_result_files(log_dir):
        with json_path.open() as f:
            payload = json.load(f)
        log_text = log_path.read_text()
        peak_ram_gb = parse_peak_ram_gb(log_text)
        header = parse_header(log_text)
        script_meta = parse_script_metadata(log_text)
        metadata = payload.get("metadata", {})
        timestamp = resolve_timestamp(log_path, metadata, script_meta)
        stem = json_path.stem

        coarse = payload["coarse"]
        for target in payload["targets"]:
            name = canonical_target_name(target["name"])
            if name not in CSV_TARGETS:
                continue
            if name not in selected_targets:
                continue
            bits = int(target.get("budget_bits", payload["bits"]))
            refine_time = target.get("refine_time")
            if refine_time is None or (
                isinstance(refine_time, float) and not math.isfinite(refine_time)
            ):
                refine_time = 0.0
            rerank_total = target["rerank_time"] + refine_time
            method_meta = target.get("method") or payload.get("method") or {}
            row = {
                "timestamp": timestamp,
                "dataset": payload["dataset"],
                "d": fmt_int(payload["dim"]),
                "nb": fmt_int(payload["nb"]),
                "nq": fmt_int(payload["nq"]),
                "budget_b": fmt_int(bits),
                "nlist": fmt_int(payload["nlist"]),
                "nprobe": fmt_int(payload["nprobe"]),
                "topk": fmt_int(payload["topk"]),
                "rerank_depth": "all",
                "train_rows": fmt_int(payload["nt"]),
                "threads": resolve_threads(payload, metadata, header, script_meta),
                "impl_backend": CSV_TARGETS[name]["impl_backend"](bits),
                "coarse_train_s": fmt_float(coarse["train_time"]),
                "coarse_add_s": fmt_float(coarse["add_time"]),
                "coarse_assign_s": fmt_float(coarse["assign_time"]),
                "avg_candidates_per_q": fmt_float(coarse["avg_candidates"], 4),
                "candidate_hit_rate": fmt_float(coarse["candidate_hit_rate"], 4),
                "structure_s": fmt_float(target["structure_time"]),
                "prep_s": fmt_float(target["preparation_time"]),
                "codebook_s": fmt_float(target["codebook_time"]),
                "train_total_s": fmt_float(target["train_total"]),
                "peak_ram_gb": fmt_float(peak_ram_gb, 3),
                "add_encode_s": fmt_float(target["add_time"]),
                "encode_us_per_vec": fmt_float(target["encode_per_vector"] * 1e6, 3),
                "rerank_total_s": fmt_float(rerank_total),
                "search_total_s": fmt_float(target["total_query_time"]),
                "search_ms_per_q": fmt_float(target["search_per_query"] * 1e3, 3),
                "qps": fmt_float(target["qps"], 3),
                "recall_1": fmt_float(target["recall1"], 4),
                "recall_10": fmt_float(target["recall10"], 4),
                "recall_100": fmt_float(target["recall100"], 4),
                "recall_1000": fmt_float(target["recall1000"], 4),
                "overlap_1000": fmt_float(target["overlap1000"], 4),
                "J": fmt_float(target["reconstruction_error"], 4),
                "index_size_mb": fmt_float(target.get("index_size_mib"), 3),
                "notes": build_notes(
                    stem,
                    payload,
                    target,
                    metadata,
                    method_meta,
                    header,
                    script_meta,
                ),
            }
            rows_by_target[name].append(row)
    return rows_by_target


def load_rows_from_dirs(log_dirs: list[Path], selected_targets: set[str]):
    rows_by_target = {name: [] for name in CSV_TARGETS}
    for log_dir in log_dirs:
        merge_rows(rows_by_target, load_rows(log_dir, selected_targets))
    return rows_by_target


def sort_key(row):
    return (
        DATASET_ORDER[row["dataset"]],
        int(row["budget_b"]),
        int(row["nprobe"]),
    )


def write_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=sort_key):
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        action="append",
        default=None,
        help="directory containing per-run JSON/log pairs",
    )
    parser.add_argument(
        "--targets",
        default=None,
        help="comma-separated canonical target names to backfill; default is auto-detected from available JSON payloads",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=24,
        help="expected row count per selected target",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log_dirs = args.log_dir or [DEFAULT_SERIAL_LOG_DIR]
    for log_dir in log_dirs:
        if not log_dir.exists():
            raise RuntimeError(f"serial log dir does not exist: {log_dir}")
    if args.targets is None:
        selected_targets = set()
        for log_dir in log_dirs:
            selected_targets.update(detect_available_targets(log_dir))
    else:
        selected_targets = {t.strip() for t in args.targets.split(",") if t.strip()}
    if not selected_targets:
        raise RuntimeError(f"no supported targets found in {log_dirs}")
    unknown = selected_targets - set(CSV_TARGETS)
    if unknown:
        raise RuntimeError(f"unknown targets: {sorted(unknown)}")
    rows_by_target = load_rows_from_dirs(log_dirs, selected_targets)
    for target in selected_targets:
        meta = CSV_TARGETS[target]
        rows = rows_by_target[target]
        if len(rows) != args.expected_rows:
            raise RuntimeError(
                f"{target} expected {args.expected_rows} rows from unified runs, got {len(rows)}"
            )
        write_csv(meta["path"], rows)
        print(f"wrote {len(rows)} rows to {meta['path']}")


if __name__ == "__main__":
    main()
