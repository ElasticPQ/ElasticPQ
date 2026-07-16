from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar

import numpy as np

from .types import MediaPart

T = TypeVar("T")


def chunked(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def normalize_text(value: str) -> str:
    value = value.replace("<|image_1|>", " ").replace("<image>", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in value.splitlines()).strip()


def join_prompt_text(*parts: str) -> str:
    pieces = [normalize_text(part) for part in parts if normalize_text(part)]
    return "\n".join(pieces).strip()


def guess_mime_type(path: str | Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type:
        return mime_type
    suffix = Path(path).suffix.lower()
    fallback = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".pdf": "application/pdf",
    }
    if suffix in fallback:
        return fallback[suffix]
    raise ValueError(f"cannot infer mime type for {path}")


def media_signature(parts: Iterable[MediaPart]) -> str:
    payload: list[dict[str, str | int | None]] = []
    for part in parts:
        row: dict[str, str | int | None] = {
            "kind": part.kind,
            "value": part.value,
            "mime_type": part.mime_type,
        }
        if part.kind != "text":
            candidate = Path(part.value)
            if candidate.exists():
                stat = candidate.stat()
                row["size"] = int(stat.st_size)
                row["mtime_ns"] = int(stat.st_mtime_ns)
        payload.append(row)
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
