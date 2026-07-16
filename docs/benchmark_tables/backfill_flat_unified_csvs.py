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

LEGACY_FULL_LOG_DIR = LOG_ROOT / "flat_topk1000_fullsplits_serial_20260619"
LEGACY_SIZE_LOG_DIR = LOG_ROOT / "flat_topk1000_fullsplits_size_serial_20260619"
LEGACY_AVQ_LOG_DIR = LOG_ROOT / "flat_topk1000_fullsplits_serial_20260621_avq"


def discover_latest_dir(patterns, fallback, exclude_substrings=()):
    matches = []
    for pattern in patterns:
        for path in LOG_ROOT.glob(pattern):
            if not path.is_dir():
                continue
            if any(token in path.name for token in exclude_substrings):
                continue
            matches.append(path)
    if not matches:
        return fallback
    return sorted(matches, key=lambda path: (path.stat().st_mtime, path.name))[-1]


DEFAULT_FULL_LOG_DIR = discover_latest_dir(
    ["flat_topk1000_fullsplits_serial*"],
    fallback=LEGACY_FULL_LOG_DIR,
    exclude_substrings=("avq", "size"),
)
DEFAULT_SIZE_LOG_DIR = discover_latest_dir(
    ["flat_topk1000_fullsplits_size_serial*"],
    fallback=LEGACY_SIZE_LOG_DIR,
)
DEFAULT_AVQ_LOG_DIR = discover_latest_dir(
    ["flat_topk1000_fullsplits_serial*avq*"],
    fallback=LEGACY_AVQ_LOG_DIR,
)
DEFAULT_DPOPQ_LOG_DIR = discover_latest_dir(
    ["flat_dpopq_full*"],
    fallback=DEFAULT_FULL_LOG_DIR,
)

CSV_TARGETS = {
    "pq": {
        "csv_name": "PQ",
        "path": CSV_DIR / "PQ.csv",
        "impl_backend": lambda bits: f"faiss:IndexPQ M={bits // 8} nbits=8",
        "log_dir_key": "full",
        "needs_index_size": True,
    },
    "opq": {
        "csv_name": "OPQ",
        "path": CSV_DIR / "OPQ.csv",
        "impl_backend": lambda bits: f"faiss:IndexPreTransform(OPQ+PQ) M={bits // 8} nbits=8",
        "log_dir_key": "full",
        "needs_index_size": True,
    },
    "dpopq": {
        "csv_name": "DP-OPQ",
        "path": CSV_DIR / "DP-OPQ.csv",
        "impl_backend": lambda bits: f"local:DP-OPQ PCA+DP M={bits // 8} nbits=8",
        "log_dir_key": "dpopq",
        "needs_index_size": True,
    },
    "epq": {
        "csv_name": "EPQ",
        "path": CSV_DIR / "EPQ.csv",
        "impl_backend": lambda bits: "cpp:IndexEPQ uneven_transform=on",
        "log_dir_key": "full",
        "needs_index_size": True,
    },
    "arepq": {
        "csv_name": "AREPQ",
        "path": CSV_DIR / "AREPQ.csv",
        "impl_backend": lambda bits: (
            f"cpp:IndexAREPQ main_bits={bits - 8} tail_bits=8 beam=6"
        ),
        "log_dir_key": "full",
        "needs_index_size": True,
    },
    "bapq": {
        "csv_name": "BAPQ",
        "path": CSV_DIR / "BAPQ.csv",
        "impl_backend": lambda bits: "cpp:IndexBAPQ q=4 bmax=12 km=50x3",
        "log_dir_key": "full",
        "needs_index_size": True,
    },
    "rabitq": {
        "csv_name": "RaBitQ",
        "path": CSV_DIR / "RaBitQ.csv",
        "impl_backend": lambda bits: "faiss:IndexRaBitQ nb_bits=1",
        "log_dir_key": "full",
        "needs_index_size": True,
    },
    "rq": {
        "csv_name": "RQ",
        "path": CSV_DIR / "RQ.csv",
        "impl_backend": lambda bits: f"faiss:IndexResidualQuantizer M={bits // 8 - 1} nbits=8",
        "log_dir_key": "full",
        "needs_index_size": True,
    },
    "lsq": {
        "csv_name": "LSQ",
        "path": CSV_DIR / "LSQ.csv",
        "impl_backend": lambda bits: f"faiss:IndexLocalSearchQuantizer M={bits // 8 - 1} nbits=8",
        "log_dir_key": "full",
        "needs_index_size": True,
    },
    "avq": {
        "csv_name": "AVQ",
        "path": CSV_DIR / "AVQ.csv",
        "impl_backend": lambda bits: "scann:score_ah(lut16) via python-embedded adapter",
        "log_dir_key": "avq",
        "needs_index_size": False,
    },
    "vaq": {
        "csv_name": "VAQ",
        "path": CSV_DIR / "VAQ.csv",
        "impl_backend": lambda bits: "TheDatumOrg/VAQ PCA+variance-aware allocation",
        "log_dir_key": "full",
        "needs_index_size": True,
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
    "train_rows",
    "threads",
    "impl_backend",
    "structure_s",
    "prep_s",
    "codebook_s",
    "train_total_s",
    "peak_ram_gb",
    "add_encode_s",
    "encode_us_per_vec",
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


def rabitq_min_budget_bits(d: int) -> int:
    return (((d + 7) // 8) + 8) * 8


def fmt_float(value: float, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "N/A"
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def fmt_int(value: int) -> str:
    return str(int(value))


def parse_required(pattern: str, text: str, label: str):
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"failed to parse {label}")
    return match


def parse_peak_ram_gb(log_text: str) -> float:
    match = parse_required(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        log_text,
        "Maximum resident set size",
    )
    kb = int(match.group(1))
    return kb * 1024.0 / 1e9


def parse_kv_line(line: str) -> dict:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", line))


def parse_header(log_text: str):
    header_match = parse_required(r"^dataset=.*$", log_text, "benchmark header line")
    fields = parse_kv_line(header_match.group(0))
    required = ["dataset", "d", "nb", "nq", "nt", "gt_k", "bits", "mode"]
    missing = [key for key in required if key not in fields]
    if missing:
        raise RuntimeError(f"failed to parse benchmark header keys: {missing}")
    return {
        "dataset": fields["dataset"],
        "d": int(fields["d"]),
        "nb": int(fields["nb"]),
        "nq": int(fields["nq"]),
        "nt": int(fields["nt"]),
        "gt_k": int(fields["gt_k"]),
        "bits": int(fields["bits"]),
        "mode": fields["mode"],
        "train_only": fields.get("train_only", "false"),
        "skip_search": fields.get("skip_search", "false"),
        "topk": int(fields["topk"]) if "topk" in fields else 1000,
        "metric_topk": int(fields["metric_topk"]) if "metric_topk" in fields else 1000,
        "recon_sample": (
            int(fields["recon_sample"]) if "recon_sample" in fields else None
        ),
        "threads": int(fields["threads"]) if "threads" in fields else None,
        "maxtrain": int(fields["maxtrain"]) if "maxtrain" in fields else None,
    }


def parse_summary(log_text: str):
    def grab(label: str, pattern: str) -> float:
        match = parse_required(pattern, log_text, label)
        return float(match.group(1))

    recalls = parse_required(
        r"recall@1:\s*([0-9.]+)\s+recall@10:\s*([0-9.]+)\s+recall@100:\s*([0-9.]+)\s+recall@1000:\s*([0-9.]+)",
        log_text,
        "recall summary",
    )
    return {
        "structure_s": grab("structure time", r"structure time:\s*([0-9.]+)\s+s"),
        "prep_s": grab("preparation time", r"preparation time:\s*([0-9.]+)\s+s"),
        "codebook_s": grab("codebook time", r"codebook time:\s*([0-9.]+)\s+s"),
        "train_total_s": grab("training total", r"training total:\s*([0-9.]+)\s+s"),
        "add_encode_s": grab("add/encode time", r"add/encode time:\s*([0-9.]+)\s+s"),
        "encode_s_per_vec": grab(
            "encode per vector", r"encode per vector:\s*([0-9.]+)\s+s/vector"
        ),
        "search_total_s": grab("search time", r"search time:\s*([0-9.]+)\s+s"),
        "search_s_per_q": grab(
            "search per query", r"search per query:\s*([0-9.]+)\s+s/query"
        ),
        "qps": grab("QPS", r"QPS:\s*([0-9.]+)"),
        "recall_1": float(recalls.group(1)),
        "recall_10": float(recalls.group(2)),
        "recall_100": float(recalls.group(3)),
        "recall_1000": float(recalls.group(4)),
        "overlap_1000": grab("overlap@1000", r"overlap@1000\(gt=1000\):\s*([0-9.]+)"),
        "J": parse_optional_float(
            log_text,
            r"reconstruction error \(sample\):\s*([0-9.]+)",
        ),
    }


def parse_index_size_mib(log_text: str) -> float:
    match = re.search(r"serialized index size:\s*([0-9.]+)\s+MiB", log_text)
    if not match:
        raise RuntimeError("failed to parse serialized index size")
    return float(match.group(1))


def parse_optional_float(text: str, pattern: str):
    match = re.search(pattern, text)
    if not match:
        return math.nan
    return float(match.group(1))


def parse_meta_json(log_text: str, key: str) -> dict:
    match = re.search(rf"^{re.escape(key)}\s+(.+)$", log_text, re.MULTILINE)
    if not match:
        return {}
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to parse {key} JSON: {exc}") from exc
    return value if isinstance(value, dict) else {}


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


def resolve_timestamp(log_path: Path, hardware_meta: dict, script_meta: dict) -> str:
    for value in (
        hardware_meta.get("timestamp_utc"),
        script_meta.get("start_utc"),
    ):
        text = compact_text(value)
        if text:
            return text
    return (
        datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def first_int(*values):
    for value in values:
        if value is None or value == "":
            continue
        return int(value)
    return None


def resolve_threads(header: dict, common_meta: dict, script_meta: dict) -> str:
    thread_meta = common_meta.get("threads", {})
    value = first_int(
        thread_meta.get("requested_threads"),
        thread_meta.get("effective_threads"),
        header.get("threads"),
        script_meta.get("threads"),
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
        if method_meta.get("native_index") is not None:
            parts.append(f"native={method_meta.get('native_index')}")
        if method_meta.get("source_status") is not None:
            parts.append(f"source={method_meta.get('source_status')}")
        group_dims = method_meta.get("group_dims")
        if isinstance(group_dims, list):
            parts.append("group_dims=" + "/".join(str(value) for value in group_dims))
    elif family == "rabitq":
        for key in ("nb_bits", "nominal_budget_bits", "effective_budget_bits"):
            if method_meta.get(key) is not None:
                parts.append(f"{key}={method_meta.get(key)}")
    elif family in {"rq", "lsq"}:
        for key in (
            "M",
            "nbits",
            "total_bits",
            "effective_budget_bits",
            "search_type",
        ):
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
    elif family == "arepq":
        for key in (
            "total_bits",
            "main_bits",
            "tail_bits",
            "tail_stages",
            "tail_ksub",
            "tail_beam_candidates",
            "icm_iters",
        ):
            if method_meta.get(key) is not None:
                parts.append(f"{key}={method_meta.get(key)}")
        main = method_meta.get("main")
        if isinstance(main, dict):
            if main.get("total_bits") is not None:
                parts.append(f"main_total_bits={main.get('total_bits')}")
            builder_summary = summarize_builder(main.get("builder", {}))
            if builder_summary:
                parts.append(f"main_builder={builder_summary}")
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
    else:
        for key in ("total_bits", "M", "nbits"):
            if method_meta.get(key) is not None:
                parts.append(f"{key}={method_meta.get(key)}")

    return ",".join(parts)


def summarize_env(common_meta: dict, script_meta: dict) -> str | None:
    hardware = common_meta.get("hardware", {})
    build = common_meta.get("build", {})
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
    target: str,
    header: dict,
    common_meta: dict,
    method_meta: dict,
    script_meta: dict,
) -> str:
    run_meta = common_meta.get("run", {})
    config_name = compact_path(
        run_meta.get("config_path") or script_meta.get("config")
    ) or "epq_train_refined.json"
    env_summary = summarize_env(common_meta, script_meta)
    method_cfg = summarize_method_cfg(method_meta)
    run_tag = compact_text(script_meta.get("run_tag"))
    topk = header.get("topk") or run_meta.get("topk") or 1000
    metric_topk = header.get("metric_topk") or run_meta.get("metric_topk") or topk
    recon_sample = (
        header.get("recon_sample")
        or run_meta.get("recon_sample")
        or "min(200000,nb)"
    )
    parts = [
        f"config={config_name}",
        f"protocol=flat {header.get('mode', 'adc')}",
        "splits=train/full query/full",
        f"topk={topk}",
        f"metric_topk={metric_topk}",
        f"recon_sample={recon_sample}",
        "peak_ram_gb=/usr/bin/time -v maxrss",
        "index_size_mb=serialized bytes /(1024^2) when available",
        f"log_stem={stem}",
    ]
    if run_tag:
        parts.append(f"run_tag={run_tag}")
    if env_summary:
        parts.append(f"env={env_summary}")
    if method_cfg:
        parts.append(f"method_cfg={method_cfg}")
    if target == "rabitq":
        parts.append(
            "RaBitQ nominal budget labels follow the shared 64/128-bit benchmark convention; actual footprint is captured by index_size_mb"
        )
    if target == "avq":
        parts.append(
            "official ScaNN AVQ via embedded Python binding; actual build cost is accounted in add/encode rather than train"
        )
    if target == "dpopq":
        parts.append(
            "in-repository DP-OPQ implementation; no public reference implementation available; flat ADC uses the in-repository scan kernel"
        )
    return "; ".join(parts)


def sort_key(row):
    return (DATASET_ORDER[row["dataset"]], int(row["budget_b"]))


def resolve_size_log_path(args, full_log_path: Path) -> Path:
    return args.size_log_dir / full_log_path.name


def load_rows(args):
    log_dirs = {
        "full": args.full_log_dir,
        "avq": args.avq_log_dir,
        "dpopq": args.dpopq_log_dir,
    }
    rows_by_target = {target: [] for target in CSV_TARGETS}
    for target, meta in CSV_TARGETS.items():
        log_dir = log_dirs[meta["log_dir_key"]]
        if not log_dir.exists():
            raise RuntimeError(f"log dir does not exist for {target}: {log_dir}")
        for full_log_path in sorted(log_dir.glob(f"flat_*_{target}.log")):
            stem = full_log_path.stem
            full_text = full_log_path.read_text()
            header = parse_header(full_text)
            summary = parse_summary(full_text)
            common_meta = {
                "run": parse_meta_json(full_text, "meta.run"),
                "hardware": parse_meta_json(full_text, "meta.hardware"),
                "build": parse_meta_json(full_text, "meta.build"),
                "threads": parse_meta_json(full_text, "meta.threads"),
                "dataset": parse_meta_json(full_text, "meta.dataset"),
                "config": parse_meta_json(full_text, "meta.config"),
            }
            method_meta = parse_meta_json(full_text, "meta.method")
            script_meta = parse_script_metadata(full_text)
            peak_ram_gb = parse_peak_ram_gb(full_text)
            timestamp = resolve_timestamp(
                full_log_path,
                common_meta["hardware"],
                script_meta,
            )

            if meta["needs_index_size"]:
                try:
                    index_size_mib = parse_index_size_mib(full_text)
                except RuntimeError:
                    size_log_path = resolve_size_log_path(args, full_log_path)
                    if not size_log_path.exists():
                        raise RuntimeError(
                            f"missing size log for {full_log_path.name}: {size_log_path}"
                        )
                    index_size_mib = parse_index_size_mib(size_log_path.read_text())
            else:
                index_size_mib = math.nan

            bits = header["bits"]
            budget_bits = (
                rabitq_min_budget_bits(header["d"]) if target == "rabitq" else bits
            )
            row = {
                "timestamp": timestamp,
                "dataset": header["dataset"],
                "d": fmt_int(header["d"]),
                "nb": fmt_int(header["nb"]),
                "nq": fmt_int(header["nq"]),
                "budget_b": fmt_int(budget_bits),
                "train_rows": fmt_int(header["nt"]),
                "threads": resolve_threads(header, common_meta, script_meta),
                "impl_backend": meta["impl_backend"](bits),
                "structure_s": fmt_float(summary["structure_s"]),
                "prep_s": fmt_float(summary["prep_s"]),
                "codebook_s": fmt_float(summary["codebook_s"]),
                "train_total_s": fmt_float(summary["train_total_s"]),
                "peak_ram_gb": fmt_float(peak_ram_gb, 3),
                "add_encode_s": fmt_float(summary["add_encode_s"]),
                "encode_us_per_vec": fmt_float(summary["encode_s_per_vec"] * 1e6, 3),
                "search_total_s": fmt_float(summary["search_total_s"]),
                "search_ms_per_q": fmt_float(summary["search_s_per_q"] * 1e3, 3),
                "qps": fmt_float(summary["qps"], 3),
                "recall_1": fmt_float(summary["recall_1"], 4),
                "recall_10": fmt_float(summary["recall_10"], 4),
                "recall_100": fmt_float(summary["recall_100"], 4),
                "recall_1000": fmt_float(summary["recall_1000"], 4),
                "overlap_1000": fmt_float(summary["overlap_1000"], 4),
                "J": fmt_float(summary["J"], 4),
                "index_size_mb": fmt_float(index_size_mib, 3),
                "notes": build_notes(
                    stem,
                    target,
                    header,
                    common_meta,
                    method_meta,
                    script_meta,
                ),
            }
            rows_by_target[target].append(row)
    return rows_by_target


def write_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=sort_key):
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-log-dir",
        type=Path,
        default=DEFAULT_FULL_LOG_DIR,
        help="directory containing PQ/OPQ/EPQ/BAPQ/RaBitQ flat logs",
    )
    parser.add_argument(
        "--size-log-dir",
        type=Path,
        default=DEFAULT_SIZE_LOG_DIR,
        help="directory containing index-size fallback logs",
    )
    parser.add_argument(
        "--avq-log-dir",
        type=Path,
        default=DEFAULT_AVQ_LOG_DIR,
        help="directory containing AVQ flat logs",
    )
    parser.add_argument(
        "--dpopq-log-dir",
        type=Path,
        default=DEFAULT_DPOPQ_LOG_DIR,
        help="directory containing DP-OPQ flat logs",
    )
    parser.add_argument(
        "--targets",
        default=",".join(CSV_TARGETS.keys()),
        help="comma-separated target names to backfill",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=6,
        help="expected row count per selected target",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown = sorted(set(selected_targets) - set(CSV_TARGETS))
    if unknown:
        raise RuntimeError(f"unknown targets: {unknown}")
    rows_by_target = load_rows(args)
    for target in selected_targets:
        meta = CSV_TARGETS[target]
        rows = rows_by_target[target]
        if len(rows) != args.expected_rows:
            raise RuntimeError(
                f"{meta['csv_name']} expected {args.expected_rows} rows from flat unified runs, got {len(rows)}"
            )
        write_csv(meta["path"], rows)
        print(f"wrote {len(rows)} rows to {meta['path']}")


if __name__ == "__main__":
    main()
