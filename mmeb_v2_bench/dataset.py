from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

from .types import Candidate, MediaPart, QueryExample, TaskDataset, TaskSpec
from .utils import join_prompt_text


def _ensure_tuple(parts: Iterable[MediaPart]) -> tuple[MediaPart, ...]:
    return tuple(part for part in parts if part.value)


def _candidate_from_text(name: str, text: str) -> Candidate:
    return Candidate(name=name, parts=_ensure_tuple([MediaPart(kind="text", value=text)]))


def _candidate_from_image(name: str, image_path: Path, caption: str | None = None) -> Candidate:
    parts = []
    if caption and caption.strip():
        parts.append(MediaPart(kind="text", value=caption.strip()))
    parts.append(MediaPart(kind="image", value=str(image_path), mime_type=None))
    return Candidate(name=name, parts=_ensure_tuple(parts))


def _query_with_image(query_id: str, prompt: str, image_path: Path, labels: list[str], candidate_names: list[str]) -> QueryExample:
    parts = _ensure_tuple(
        [
            MediaPart(kind="text", value=prompt),
            MediaPart(kind="image", value=str(image_path), mime_type=None),
        ]
    )
    return QueryExample(
        query_id=query_id,
        parts=parts,
        labels=tuple(labels),
        candidate_names=tuple(candidate_names),
    )


def _query_text_only(query_id: str, prompt: str, labels: list[str], candidate_names: list[str]) -> QueryExample:
    return QueryExample(
        query_id=query_id,
        parts=_ensure_tuple([MediaPart(kind="text", value=prompt)]),
        labels=tuple(labels),
        candidate_names=tuple(candidate_names),
    )


def _query_with_parts(
    query_id: str,
    parts: Iterable[MediaPart],
    labels: list[str],
    candidate_names: list[str],
) -> QueryExample:
    return QueryExample(
        query_id=query_id,
        parts=_ensure_tuple(parts),
        labels=tuple(labels),
        candidate_names=tuple(candidate_names),
    )


_FRAME_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_frame_files(frames_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _FRAME_SUFFIXES
    )


def _sample_evenly(items: list[Path], max_items: int) -> list[Path]:
    if max_items <= 0 or len(items) <= max_items:
        return items
    if max_items == 1:
        return [items[0]]
    last = len(items) - 1
    indices = {round(i * last / (max_items - 1)) for i in range(max_items)}
    return [items[idx] for idx in sorted(indices)]


def _frame_parts(frames_dir: Path, *, max_frames: int = 6) -> tuple[MediaPart, ...]:
    frame_files = _sample_evenly(_iter_frame_files(frames_dir), max_frames)
    if not frame_files:
        raise FileNotFoundError(f"no image frames found under {frames_dir}")
    return _ensure_tuple(MediaPart(kind="image", value=str(path), mime_type=None) for path in frame_files)


def _parts_with_frames(prompt: str, frames_dir: Path, *, max_frames: int = 6) -> tuple[MediaPart, ...]:
    parts: list[MediaPart] = []
    if prompt.strip():
        parts.append(MediaPart(kind="text", value=prompt.strip()))
    parts.extend(_frame_parts(frames_dir, max_frames=max_frames))
    return _ensure_tuple(parts)


def _candidate_from_frames(name: str, frames_dir: Path, *, max_frames: int = 6) -> Candidate:
    return Candidate(name=name, parts=_frame_parts(frames_dir, max_frames=max_frames))


def _task_source_name(spec: TaskSpec) -> str:
    return spec.source_name or spec.name


def _task_media_dir(media_root: Path, spec: TaskSpec) -> Path:
    if spec.media_subdir:
        return media_root / spec.media_subdir
    return media_root


def _load_jsonl_rows(path: Path, *, num_samples: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            rows.append(json.loads(line))
            if num_samples is not None and num_samples > 0 and line_idx >= num_samples:
                break
    return rows


@lru_cache(maxsize=32)
def _directory_lookup(root: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    root_path = Path(root)
    if not root_path.exists():
        return mapping
    for path in root_path.rglob("*"):
        if not path.is_dir():
            continue
        keys = {
            path.name,
            path.stem,
            path.name.removeprefix("v_"),
            path.stem.removeprefix("v_"),
        }
        for key in keys:
            if key:
                mapping.setdefault(key, path)
    return mapping


def _lookup_directory(root: Path, *keys: str) -> Path:
    mapping = _directory_lookup(str(root.resolve()))
    for key in keys:
        if key and key in mapping:
            return mapping[key]
    tried = [key for key in keys if key]
    raise FileNotFoundError(f"failed to locate local directory under {root} for keys={tried}")


def _row_value(row: dict, *keys: str, default=None):
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return default


def _save_image_value(image_value, out_path: Path) -> Path:
    actual_path = out_path if out_path.suffix else out_path.with_suffix(".png")
    actual_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image_value, "save"):
        if actual_path.suffix:
            image_value.save(actual_path)
        else:
            image_value.save(actual_path, format="PNG")
        return actual_path
    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes") or image_value.get("data")
        if image_bytes:
            actual_path.write_bytes(image_bytes)
            return actual_path
    raise RuntimeError(f"unable to materialize image payload to {actual_path}")


def _visrag_short_name(raw_name: str) -> str:
    base = Path(raw_name).stem
    ext = Path(raw_name).suffix or ".png"
    short_base = base[:50] + "_" + hashlib.md5(raw_name.encode("utf-8")).hexdigest()[:8]
    return short_base + ext


def _resolve_visdoc_image_path(image_root: Path, corpus_id: str) -> Path:
    suffix = Path(corpus_id).suffix
    candidates = [image_root / (corpus_id if suffix else f"{corpus_id}.png")]
    if suffix:
        candidates.append(image_root / corpus_id)
    if not suffix:
        candidates.append(image_root / f"{corpus_id}.png")
    hashed = _visrag_short_name(corpus_id)
    candidates.append(image_root / hashed)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_manifest_task(manifest_path: str | Path, task_name: str = "manifest") -> TaskDataset:
    manifest_path = Path(manifest_path)
    corpus_map: dict[str, Candidate] = {}
    queries: list[QueryExample] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            row = json.loads(line)
            query_id = str(row.get("query_id", line_idx))
            labels = [str(label) for label in row.get("labels", [])]
            candidate_names: list[str] = []
            for candidate_row in row["candidates"]:
                candidate = Candidate(
                    name=str(candidate_row["name"]),
                    parts=tuple(MediaPart(**part) for part in candidate_row["parts"]),
                )
                corpus_map.setdefault(candidate.name, candidate)
                candidate_names.append(candidate.name)
            queries.append(
                QueryExample(
                    query_id=query_id,
                    parts=tuple(MediaPart(**part) for part in row["query_parts"]),
                    labels=tuple(labels),
                    candidate_names=tuple(candidate_names),
                )
            )
    spec = TaskSpec(name=task_name, group="manifest", dataset_parser="manifest")
    return TaskDataset(spec=spec, queries=queries, corpus=list(corpus_map.values()))


def _annotation_cache_task_dir(annotation_cache_dir: str | Path, annotation_source: str, task_name: str, split: str) -> Path:
    root = Path(annotation_cache_dir)
    repo_key = str(annotation_source).replace("/", "__")
    return root / repo_key / task_name / split


def _load_local_parquet_dataset(local_dir: str | Path):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for MMEB-V2 loading. Install dependencies with "
            "`pip install -r mmeb_v2_bench/requirements.txt`."
        ) from exc

    local_dir = Path(local_dir)
    parquet_files = sorted(str(path) for path in local_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files found in local annotation cache dir: {local_dir}")
    return load_dataset("parquet", data_files={"train": parquet_files}, split="train")


def _materialize_hf_parquet_dataset(
    annotation_source: str,
    task_name: str,
    split: str,
    *,
    annotation_cache_dir: str | Path,
):
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for HF annotation persistence. "
            "Install dependencies with `pip install -r mmeb_v2_bench/requirements.txt`."
        ) from exc

    task_dir = _annotation_cache_task_dir(annotation_cache_dir, annotation_source, task_name, split)
    manifest_path = task_dir / "_manifest.json"
    complete_path = task_dir / ".complete"
    if complete_path.exists() and manifest_path.exists():
        return task_dir

    task_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=annotation_source, repo_type="dataset")
    prefix = f"{task_name}/{split}"
    parquet_files = sorted(
        path for path in repo_files if path.startswith(prefix) and path.endswith(".parquet")
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"no parquet files found for task={task_name!r} split={split!r} in dataset {annotation_source!r}"
        )

    downloaded_files: list[str] = []
    for path in parquet_files:
        downloaded_path = hf_hub_download(
            repo_id=annotation_source,
            repo_type="dataset",
            filename=path,
            local_dir=str(task_dir),
        )
        downloaded_files.append(str(Path(downloaded_path).resolve()))

    manifest = {
        "annotation_source": annotation_source,
        "task_name": task_name,
        "split": split,
        "files": downloaded_files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    complete_path.write_text("ok\n", encoding="utf-8")
    return task_dir


def _load_hf_dataset(
    annotation_source: str,
    task_name: str,
    split: str,
    *,
    annotation_cache_dir: str | Path | None = None,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for MMEB-V2 loading. Install dependencies with "
            "`pip install -r mmeb_v2_bench/requirements.txt`."
        ) from exc

    source_path = str(annotation_source)
    if Path(source_path).exists():
        return load_dataset(source_path, task_name, split=split)
    if annotation_cache_dir is not None:
        task_dir = _annotation_cache_task_dir(annotation_cache_dir, annotation_source, task_name, split)
        if (task_dir / ".complete").exists():
            print(f"[annotation cache] using local parquet cache: {task_dir}")
            return _load_local_parquet_dataset(task_dir)
        print(f"[annotation cache] materializing HF parquet to local cache: {task_dir}")
        task_dir = _materialize_hf_parquet_dataset(
            annotation_source,
            task_name,
            split,
            annotation_cache_dir=annotation_cache_dir,
        )
        return _load_local_parquet_dataset(task_dir)
    try:
        return load_dataset(source_path, task_name, split=split)
    except Exception as exc:
        try:
            return _load_hf_parquet_dataset(source_path, task_name, split)
        except Exception as parquet_exc:
            raise RuntimeError(
                "failed to load HF dataset via standard builder and direct parquet fallback. "
                f"builder_error={exc!r}; parquet_error={parquet_exc!r}"
            ) from parquet_exc


def _load_hf_parquet_dataset(annotation_source: str, task_name: str, split: str):
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi, hf_hub_url
    except ImportError as exc:
        raise RuntimeError(
            "datasets and huggingface_hub are required for HF parquet fallback. "
            "Install dependencies with `pip install -r mmeb_v2_bench/requirements.txt`."
        ) from exc

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=annotation_source, repo_type="dataset")
    prefix = f"{task_name}/{split}"
    parquet_files = sorted(
        path for path in repo_files if path.startswith(prefix) and path.endswith(".parquet")
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"no parquet files found for task={task_name!r} split={split!r} in dataset {annotation_source!r}"
        )
    data_files = [hf_hub_url(repo_id=annotation_source, filename=path, repo_type="dataset") for path in parquet_files]
    return load_dataset("parquet", data_files={"train": data_files}, split="train")


def _parse_modelscope_id(dataset_id: str) -> tuple[str | None, str]:
    parts = str(dataset_id).split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, parts[0]


def _load_modelscope_dataset(annotation_source: str, task_name: str, split: str):
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as exc:
        raise RuntimeError(
            "modelscope is required for ModelScope dataset fallback. Install dependencies with "
            "`pip install -r mmeb_v2_bench/requirements.txt`."
        ) from exc

    namespace, dataset_name = _parse_modelscope_id(annotation_source)

    def _load_once(**kwargs):
        if namespace is None:
            return MsDataset.load(dataset_name, **kwargs)
        return MsDataset.load(dataset_name, namespace=namespace, **kwargs)

    errors: list[Exception] = []
    ms_dataset = None
    for kwargs in (
        {"subset_name": task_name, "split": split},
        {"split": split},
        {},
    ):
        try:
            ms_dataset = _load_once(**kwargs)
            break
        except Exception as exc:
            errors.append(exc)
            continue

    if ms_dataset is None:
        raise RuntimeError(
            "failed to load ModelScope dataset with all supported argument patterns. "
            + " | ".join(repr(err) for err in errors)
        )

    if isinstance(ms_dataset, dict):
        if split in ms_dataset:
            ms_dataset = ms_dataset[split]
        elif "default" in ms_dataset:
            ms_dataset = ms_dataset["default"]
        else:
            first_value = next(iter(ms_dataset.values()))
            ms_dataset = first_value

    # Prefer a HuggingFace-style dataset object when ModelScope exposes one.
    for attr in ("to_hf_dataset", "to_dataset"):
        fn = getattr(ms_dataset, attr, None)
        if callable(fn):
            converted = fn()
            if converted is not None:
                return converted

    hf_dataset = getattr(ms_dataset, "_hf_ds", None)
    if hf_dataset is not None:
        return hf_dataset

    return list(ms_dataset)


def _resolve_media_root(dataset_root: Path, spec: TaskSpec) -> Path:
    dataset_root = dataset_root.resolve()
    direct = dataset_root / spec.media_root
    if direct.exists():
        return direct
    if dataset_root.name == spec.media_root:
        return dataset_root
    return direct


def _filter_rows_for_task(dataset, task_name: str, split: str | None = None):
    candidate_keys = ("dataset_name", "task_name", "subset_name", "source_dataset", "name")
    rows = list(_iter_rows(dataset))
    if not rows:
        return rows

    for key in candidate_keys:
        if key not in rows[0]:
            continue
        filtered = [row for row in rows if str(row.get(key)) == task_name]
        if filtered:
            rows = filtered
            break

    if split is not None:
        split_keys = ("split", "dataset_split")
        for key in split_keys:
            if key not in rows[0]:
                continue
            filtered = [row for row in rows if str(row.get(key)) == split]
            if filtered:
                rows = filtered
                break
    return rows


def _load_annotation_dataset(
    *,
    annotation_source: str,
    annotation_backend: str,
    annotation_fallback_source: str | None,
    annotation_cache_dir: str | Path | None,
    task_name: str,
    split: str,
):
    backend = str(annotation_backend).lower()
    local_path = Path(annotation_source)
    if backend == "local" or local_path.exists():
        return _load_hf_dataset(str(local_path), task_name, split, annotation_cache_dir=None)
    if backend == "modelscope":
        return _load_modelscope_dataset(annotation_source, task_name, split)
    if backend == "hf":
        return _load_hf_dataset(
            annotation_source,
            task_name,
            split,
            annotation_cache_dir=annotation_cache_dir,
        )

    if backend != "auto":
        raise ValueError(f"unsupported annotation backend: {annotation_backend}")

    hf_error: Exception | None = None
    try:
        return _load_hf_dataset(
            annotation_source,
            task_name,
            split,
            annotation_cache_dir=annotation_cache_dir,
        )
    except Exception as exc:
        hf_error = exc

    if annotation_fallback_source:
        try:
            return _load_modelscope_dataset(annotation_fallback_source, task_name, split)
        except Exception as fallback_exc:
            raise RuntimeError(
                "failed to load MMEB annotations from both Hugging Face and ModelScope. "
                f"HF source={annotation_source!r} error={hf_error!r}; "
                f"ModelScope source={annotation_fallback_source!r} error={fallback_exc!r}"
            ) from fallback_exc

    raise RuntimeError(
        "failed to load MMEB annotations from Hugging Face and no ModelScope fallback source was configured. "
        f"HF source={annotation_source!r} error={hf_error!r}"
    ) from hf_error


def _slice_dataset(dataset, num_samples: int | None):
    if num_samples is None or num_samples <= 0:
        return dataset
    if hasattr(dataset, "num_rows") and hasattr(dataset, "select"):
        return dataset.select(range(min(num_samples, dataset.num_rows)))
    return list(_iter_rows(dataset, num_samples=num_samples))


def _iter_rows(dataset, *, num_samples: int | None = None) -> Iterator[dict]:
    count = 0
    for row in dataset:
        yield row
        count += 1
        if num_samples is not None and count >= num_samples:
            break


def _load_local_video_cls_task(spec: TaskSpec, *, dataset_root: Path, num_samples: int | None = None) -> TaskDataset:
    media_root = _resolve_media_root(dataset_root, spec)
    data_path = media_root / "data" / f"{_task_source_name(spec)}.jsonl"
    frame_root = _task_media_dir(media_root, spec)
    if not data_path.exists():
        raise FileNotFoundError(f"local video annotation jsonl not found: {data_path}")
    if not frame_root.exists():
        raise FileNotFoundError(f"local video frame root not found: {frame_root}")

    rows = _load_jsonl_rows(data_path, num_samples=num_samples)
    corpus_map: dict[str, Candidate] = {}
    queries: list[QueryExample] = []

    global_labels = sorted({str(_row_value(row, "pos_text", default="")) for row in rows if _row_value(row, "pos_text", default="")})
    for label in global_labels:
        corpus_map.setdefault(label, _candidate_from_text(label, label))

    for row_idx, row in enumerate(rows):
        label = str(_row_value(row, "pos_text", default=""))
        video_id = str(_row_value(row, "video_id", default=""))
        video_path = str(_row_value(row, "video_path", default=""))
        frame_dir = _lookup_directory(frame_root, video_id, Path(video_path).stem, Path(video_path).name)
        prompt = join_prompt_text(
            str(_row_value(row, "qry_instruction", default="")),
            str(_row_value(row, "qry_text", default="")),
        ) or "Recognize the category of the video content."

        candidate_names = list(global_labels)
        if spec.source_name == "ssv2":
            candidate_names = [str(item) for item in row.get("neg_text", [])]
            if label and label not in candidate_names:
                candidate_names.insert(0, label)
            for candidate_name in candidate_names:
                corpus_map.setdefault(candidate_name, _candidate_from_text(candidate_name, candidate_name))

        queries.append(
            _query_with_parts(
                query_id=f"{spec.name}:{row_idx}",
                parts=_parts_with_frames(prompt, frame_dir),
                labels=[label],
                candidate_names=candidate_names,
            )
        )

    return TaskDataset(spec=spec, queries=queries, corpus=list(corpus_map.values()))


def _load_local_video_qa_binary_task(spec: TaskSpec, *, dataset_root: Path, num_samples: int | None = None) -> TaskDataset:
    media_root = _resolve_media_root(dataset_root, spec)
    data_path = media_root / "data" / f"{_task_source_name(spec)}.jsonl"
    frame_root = _task_media_dir(media_root, spec)
    if not data_path.exists():
        raise FileNotFoundError(f"local video QA annotation jsonl not found: {data_path}")
    if not frame_root.exists():
        raise FileNotFoundError(f"local video QA frame root not found: {frame_root}")

    rows = _load_jsonl_rows(data_path, num_samples=num_samples)
    candidate_names = ["yes", "no"]
    corpus = [_candidate_from_text(name, name) for name in candidate_names]
    queries: list[QueryExample] = []

    for row_idx, row in enumerate(rows):
        video_name = str(_row_value(row, "video_name", default=""))
        answer = str(_row_value(row, "answer", default="")).lower()
        question = str(_row_value(row, "question", default="")).strip()
        frame_dir = _lookup_directory(
            frame_root,
            f"v_{video_name}",
            video_name,
            f"v_{video_name}.mp4",
            f"{video_name}.mp4",
        )
        prompt = f"Given a video and a question, answer with yes or no.\nQuestion: {question}"
        queries.append(
            _query_with_parts(
                query_id=f"{spec.name}:{row_idx}",
                parts=_parts_with_frames(prompt, frame_dir),
                labels=[answer],
                candidate_names=candidate_names,
            )
        )

    return TaskDataset(spec=spec, queries=queries, corpus=corpus)


def _load_local_video_mret_task(spec: TaskSpec, *, dataset_root: Path, num_samples: int | None = None) -> TaskDataset:
    media_root = _resolve_media_root(dataset_root, spec)
    data_path = media_root / "data" / f"{_task_source_name(spec)}.jsonl"
    clip_root = _task_media_dir(media_root, spec)
    if not data_path.exists():
        raise FileNotFoundError(f"local moment retrieval annotation jsonl not found: {data_path}")
    if not clip_root.exists():
        raise FileNotFoundError(f"local moment retrieval frame root not found: {clip_root}")

    rows = _load_jsonl_rows(data_path, num_samples=num_samples)
    corpus_map: dict[str, Candidate] = {}
    queries: list[QueryExample] = []

    for row_idx, row in enumerate(rows):
        clip_key = Path(str(_row_value(row, "clips_dir_path", default=""))).name
        if not clip_key:
            raise RuntimeError(f"missing clips_dir_path in row for task {spec.name}")
        clip_dir = _lookup_directory(clip_root, clip_key)
        candidate_names: list[str] = []
        labels: list[str] = []

        for candidate_dir in sorted(path for path in clip_dir.iterdir() if path.is_dir()):
            candidate_name = str(candidate_dir.relative_to(clip_root))
            corpus_map.setdefault(candidate_name, _candidate_from_frames(candidate_name, candidate_dir))
            candidate_names.append(candidate_name)
            if "positive" in candidate_dir.name.lower():
                labels.append(candidate_name)

        if not candidate_names or not labels:
            raise RuntimeError(f"failed to build moment retrieval candidates under {clip_dir}")

        queries.append(
            _query_text_only(
                query_id=f"{spec.name}:{row_idx}",
                prompt=str(_row_value(row, "query", default="")).strip(),
                labels=labels,
                candidate_names=candidate_names,
            )
        )

    return TaskDataset(spec=spec, queries=queries, corpus=list(corpus_map.values()))


def _load_local_visdoc_beir_task(spec: TaskSpec, *, dataset_root: Path, num_samples: int | None = None) -> TaskDataset:
    media_root = _resolve_media_root(dataset_root, spec)
    task_key = _task_source_name(spec)
    data_root = media_root / "data" / task_key
    image_root = _task_media_dir(media_root, spec)
    queries_dir = data_root / "queries"
    qrels_dir = data_root / "qrels"
    corpus_dir = data_root / "corpus"
    for path in (queries_dir, qrels_dir, corpus_dir):
        if not path.exists():
            raise FileNotFoundError(f"local visdoc split dir not found: {path}")

    queries_ds = _slice_dataset(_load_local_parquet_dataset(queries_dir), num_samples)
    qrels_ds = _load_local_parquet_dataset(qrels_dir)
    corpus_ds = _load_local_parquet_dataset(corpus_dir)

    qrels_mapping: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in _iter_rows(qrels_ds):
        query_id = str(_row_value(row, "query-id", "query_id"))
        corpus_id = str(_row_value(row, "corpus-id", "corpus_id"))
        score = float(_row_value(row, "score", "relevance", default=1) or 0)
        qrels_mapping[query_id].append((corpus_id, score))

    corpus_map: dict[str, Candidate] = {}
    image_root.mkdir(parents=True, exist_ok=True)
    for row in _iter_rows(corpus_ds):
        corpus_id = str(_row_value(row, "corpus-id", "corpus_id"))
        image_path = _resolve_visdoc_image_path(image_root, corpus_id)
        if not image_path.exists():
            image_value = _row_value(row, "image", default=None)
            if image_value is not None:
                image_path = _save_image_value(image_value, image_path)
        if not image_path.exists():
            raise FileNotFoundError(
                f"failed to resolve visdoc image for corpus_id={corpus_id!r} under {image_root}"
            )
        corpus_map.setdefault(corpus_id, _candidate_from_image(corpus_id, image_path))

    queries: list[QueryExample] = []
    for row_idx, row in enumerate(_iter_rows(queries_ds)):
        query_id = str(_row_value(row, "query-id", "query_id"))
        prompt = str(_row_value(row, "query", "qry_text", default="")).strip()
        rels = qrels_mapping.get(query_id, [])
        candidate_names = [corpus_id for corpus_id, _ in rels if corpus_id in corpus_map]
        labels = [corpus_id for corpus_id, score in rels if score > 0 and corpus_id in corpus_map]
        if not candidate_names or not labels:
            continue
        queries.append(
            _query_text_only(
                query_id=f"{spec.name}:{row_idx}",
                prompt=prompt,
                labels=labels,
                candidate_names=candidate_names,
            )
        )

    return TaskDataset(spec=spec, queries=queries, corpus=list(corpus_map.values()))


def load_hf_mmeb_task(
    spec: TaskSpec,
    *,
    annotation_source: str,
    annotation_backend: str = "auto",
    annotation_fallback_source: str | None = "TIGER-Lab/MMEB-V2",
    annotation_cache_dir: str | Path | None = None,
    dataset_root: str | Path,
    num_samples: int | None = None,
) -> TaskDataset:
    dataset_root = Path(dataset_root)
    if spec.dataset_parser == "video_cls_local":
        return _load_local_video_cls_task(spec, dataset_root=dataset_root, num_samples=num_samples)
    if spec.dataset_parser == "video_qa_binary_local":
        return _load_local_video_qa_binary_task(spec, dataset_root=dataset_root, num_samples=num_samples)
    if spec.dataset_parser == "video_mret_local":
        return _load_local_video_mret_task(spec, dataset_root=dataset_root, num_samples=num_samples)
    if spec.dataset_parser == "visdoc_beir_local":
        return _load_local_visdoc_beir_task(spec, dataset_root=dataset_root, num_samples=num_samples)

    media_root = _resolve_media_root(dataset_root, spec)
    dataset = _load_annotation_dataset(
        annotation_source=annotation_source,
        annotation_backend=annotation_backend,
        annotation_fallback_source=annotation_fallback_source,
        annotation_cache_dir=annotation_cache_dir,
        task_name=spec.name,
        split=spec.dataset_split,
    )
    dataset = _filter_rows_for_task(dataset, spec.name, split=spec.dataset_split)
    dataset = _slice_dataset(dataset, num_samples)

    if not media_root.exists():
        raise FileNotFoundError(
            f"resolved media root does not exist: {media_root}. "
            f"Pass --dataset-root as the MMEB-V2 root directory or the concrete {spec.media_root} directory."
        )

    corpus_map: dict[str, Candidate] = {}
    queries: list[QueryExample] = []

    for row_idx, row in enumerate(_iter_rows(dataset)):
        query_id = f"{spec.name}:{row_idx}"
        parser = spec.dataset_parser
        if parser in {"image_cls", "image_qa", "image_i2t"}:
            prompt = join_prompt_text(str(row.get("qry_inst", "")), str(row.get("qry_text", "")))
            image_path = media_root / str(row["qry_img_path"])
            target_texts = [str(item) for item in row["tgt_text"]]
            for target_text in target_texts:
                candidate = _candidate_from_text(name=target_text, text=target_text)
                corpus_map.setdefault(candidate.name, candidate)
            queries.append(
                _query_with_image(
                    query_id=query_id,
                    prompt=prompt,
                    image_path=image_path,
                    labels=[target_texts[0]],
                    candidate_names=target_texts,
                )
            )
            continue

        if parser == "image_t2i":
            prompt = join_prompt_text(str(row.get("qry_inst", "")), str(row.get("qry_text", "")))
            target_paths = [str(item) for item in row["tgt_img_path"]]
            target_captions = [str(item) for item in row["tgt_text"]]
            candidate_names: list[str] = []
            for target_path, caption in zip(target_paths, target_captions):
                name = target_path
                candidate = _candidate_from_image(name=name, image_path=media_root / target_path, caption=caption)
                corpus_map.setdefault(candidate.name, candidate)
                candidate_names.append(name)
            queries.append(
                _query_text_only(
                    query_id=query_id,
                    prompt=prompt,
                    labels=[candidate_names[0]],
                    candidate_names=candidate_names,
                )
            )
            continue

        if parser == "image_i2i_vg":
            prompt = join_prompt_text(str(row.get("qry_inst", "")), str(row.get("qry_text", "")))
            query_image = media_root / str(row["qry_img_path"])
            target_paths = [str(item) for item in row["tgt_img_path"]]
            target_captions = [str(item) for item in row["tgt_text"]]
            candidate_names = []
            for target_path, caption in zip(target_paths, target_captions):
                name = f"{target_path}:{caption.strip()}"
                candidate = _candidate_from_image(name=name, image_path=media_root / target_path, caption=caption)
                corpus_map.setdefault(candidate.name, candidate)
                candidate_names.append(name)
            queries.append(
                _query_with_image(
                    query_id=query_id,
                    prompt=prompt,
                    image_path=query_image,
                    labels=[candidate_names[0]],
                    candidate_names=candidate_names,
                )
            )
            continue

        raise NotImplementedError(
            f"dataset parser {parser!r} is not implemented in this sandbox yet"
        )

    return TaskDataset(spec=spec, queries=queries, corpus=list(corpus_map.values()))
