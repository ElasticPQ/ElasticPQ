from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Protocol

import numpy as np
from tqdm import tqdm

from .embed_cache import EmbeddingCache
from .types import MediaPart
from .utils import chunked, guess_mime_type, media_signature, normalize_rows


@dataclass
class EmbeddingBatchResult:
    vectors: np.ndarray
    kept_indices: list[int]
    skipped_indices: list[int]


class Embedder(Protocol):
    def embed(self, parts_batch: list[tuple[MediaPart, ...]], *, is_query: bool) -> EmbeddingBatchResult:
        ...


@dataclass
class GeminiEmbedderConfig:
    model: str = "gemini-embedding-2-preview"
    output_dimensionality: int = 768
    batch_size: int = 8
    normalize: bool = True
    timeout_seconds: float = 120.0
    quota_backoff_seconds: float = 120.0
    max_quota_backoff_seconds: float = 600.0
    min_request_interval_seconds: float = 0.0
    max_request_interval_seconds: float = 60.0
    quota_interval_scale: float = 1.5


class GeminiApiError(RuntimeError):
    def __init__(self, *, status_code: int | None, status: str | None, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.status = status
        self.message = message


class GeminiEmbedding2Embedder:
    def __init__(self, cfg: GeminiEmbedderConfig, *, cache: EmbeddingCache | None = None):
        self.cfg = cfg
        self.cache = cache
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY must be set for Gemini embeddings.")
        self._api_key = api_key
        self._endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.cfg.model}:batchEmbedContents"
        )
        self._base_request_interval_seconds = max(0.0, float(self.cfg.min_request_interval_seconds))
        self._request_interval_seconds = self._base_request_interval_seconds
        self._next_request_not_before = 0.0
        self._success_streak = 0

    def _cache_key(self, parts: tuple[MediaPart, ...], is_query: bool) -> str:
        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        return f"{self.cfg.model}:{self.cfg.output_dimensionality}:{task_type}:{media_signature(parts)}"

    def _part_to_payload(self, part: MediaPart) -> dict[str, object]:
        if part.kind == "text":
            return {"text": part.value}
        path = Path(part.value)
        mime_type = part.mime_type or guess_mime_type(path)
        with path.open("rb") as handle:
            data = handle.read()
        return {
            "inlineData": {
                "mimeType": mime_type,
                "data": base64.b64encode(data).decode("ascii"),
            }
        }

    def _parts_to_request(self, parts: tuple[MediaPart, ...], *, task_type: str) -> dict[str, object]:
        return {
            "model": f"models/{self.cfg.model}",
            "content": {"parts": [self._part_to_payload(part) for part in parts]},
            "taskType": task_type,
            "outputDimensionality": int(self.cfg.output_dimensionality),
        }

    def _summarize_parts(self, parts: tuple[MediaPart, ...]) -> str:
        rows: list[str] = []
        for idx, part in enumerate(parts):
            if part.kind == "text":
                text = part.value.replace("\n", " ").strip()
                if len(text) > 80:
                    text = text[:77] + "..."
                rows.append(f"{idx}:{part.kind}:{text!r}")
                continue
            path = Path(part.value)
            exists = path.exists()
            size = path.stat().st_size if exists else "missing"
            rows.append(
                f"{idx}:{part.kind}:path={path} exists={exists} size={size} mime={part.mime_type or 'auto'}"
            )
        return " | ".join(rows)

    def _is_invalid_argument_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        status = getattr(exc, "status", None)
        if str(status).upper() == "INVALID_ARGUMENT":
            return True

        if status_code == 400:
            message = getattr(exc, "message", None)
            text = f"{message or ''} {exc}".upper()
            if "INVALID_ARGUMENT" in text:
                return True
            return False

        code = getattr(exc, "code", None)
        if code == 400 and str(status).upper() == "INVALID_ARGUMENT":
            return True

        message = getattr(exc, "message", None)
        if message and "INVALID_ARGUMENT" in str(message).upper():
            return True

        text = str(exc)
        return "INVALID_ARGUMENT" in text.upper()

    def _is_retryable_quota_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        status = str(getattr(exc, "status", "") or "").upper()
        message = str(getattr(exc, "message", "") or exc).upper()
        if status_code == 429:
            return True
        if status in {"RESOURCE_EXHAUSTED", "TOO_MANY_REQUESTS"}:
            return True
        return "QUOTA EXCEEDED" in message or "RATE LIMIT" in message

    def _post_json(self, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload)
        env = os.environ.copy()
        env["GOOGLE_API_KEY"] = self._api_key
        env["GEMINI_CURL_URL"] = self._endpoint
        env["GEMINI_CURL_MAX_TIME"] = str(float(self.cfg.timeout_seconds))
        cmd = r"""
set -euo pipefail
curl --silent --show-error --fail-with-body \
  --connect-timeout 20 \
  --max-time "$GEMINI_CURL_MAX_TIME" \
  --url "$GEMINI_CURL_URL" \
  --config <(
    printf '%s\n' \
      "header = \"x-goog-api-key: $GOOGLE_API_KEY\"" \
      'header = "Content-Type: application/json"'
  ) \
  --data-binary @-
"""
        try:
            result = subprocess.run(
                ["bash", "-lc", cmd],
                input=body,
                text=True,
                capture_output=True,
                env=env,
                timeout=float(self.cfg.timeout_seconds) + 10.0,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Gemini transport timeout after {self.cfg.timeout_seconds:.1f}s"
            ) from exc

        raw = result.stdout
        if result.returncode != 0:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
            error_obj = payload.get("error", {}) if isinstance(payload, dict) else {}
            message = str(error_obj.get("message") or raw or result.stderr.strip() or "curl failed")
            status = error_obj.get("status")
            status_code = error_obj.get("code")
            raise GeminiApiError(
                status_code=None if status_code is None else int(status_code),
                status=None if status is None else str(status),
                message=message,
            )

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Gemini returned non-JSON response: {raw[:400] or result.stderr.strip()}"
            ) from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Gemini returned unexpected response type: {type(parsed).__name__}")
        return parsed

    def _wait_for_request_slot(self):
        now = time.monotonic()
        if now < self._next_request_not_before:
            time.sleep(self._next_request_not_before - now)

    def _after_success(self):
        self._success_streak += 1
        if self._request_interval_seconds > self._base_request_interval_seconds and self._success_streak >= 16:
            self._request_interval_seconds = max(
                self._base_request_interval_seconds,
                self._request_interval_seconds / max(1.05, float(self.cfg.quota_interval_scale)),
            )
            self._success_streak = 0
        self._next_request_not_before = time.monotonic() + self._request_interval_seconds

    def _after_quota_error(self):
        self._success_streak = 0
        grown = max(
            self._base_request_interval_seconds,
            max(self._request_interval_seconds, 0.0) * float(self.cfg.quota_interval_scale),
        )
        if grown == 0.0:
            grown = self._base_request_interval_seconds
        self._request_interval_seconds = min(float(self.cfg.max_request_interval_seconds), grown)
        self._next_request_not_before = time.monotonic() + self._request_interval_seconds

    def _embed_request(
        self,
        parts_chunk: list[tuple[MediaPart, ...]],
        *,
        task_type: str,
        phase_name: str,
        is_query: bool,
    ) -> list[np.ndarray | None]:
        try:
            retry_count = 0
            while True:
                try:
                    self._wait_for_request_slot()
                    response = self._post_json(
                        {"requests": [self._parts_to_request(parts, task_type=task_type) for parts in parts_chunk]}
                    )
                    self._after_success()
                    break
                except GeminiApiError as exc:
                    if not self._is_retryable_quota_error(exc):
                        raise
                    self._after_quota_error()
                    sleep_s = min(
                        float(self.cfg.max_quota_backoff_seconds),
                        float(self.cfg.quota_backoff_seconds) * (2 ** retry_count),
                    )
                    print(
                        f"[gemini backoff] phase={phase_name} model={self.cfg.model} "
                        f"chunk={len(parts_chunk)} retry={retry_count + 1} sleep={sleep_s:.0f}s "
                        f"next_interval={self._request_interval_seconds:.1f}s err={exc}"
                    )
                    time.sleep(sleep_s)
                    retry_count += 1
            embeddings = response.get("embeddings")
            self._validate_embeddings(embeddings, expected_count=len(parts_chunk))
            vectors: list[np.ndarray] = []
            for chunk_pos, embedding in enumerate(embeddings):
                values = embedding.get("values") if isinstance(embedding, dict) else None
                if values is None:
                    raise RuntimeError(
                        f"Gemini embedding item at chunk_pos={chunk_pos} is missing `values`."
                    )
                vectors.append(
                    self._validate_vector(
                        values,
                        chunk_pos=chunk_pos,
                        expected_dim=int(self.cfg.output_dimensionality),
                    )
                )
            return vectors
        except Exception as exc:
            if len(parts_chunk) > 1 and self._is_invalid_argument_error(exc):
                mid = len(parts_chunk) // 2
                print(
                    f"[gemini isolate] phase={phase_name} model={self.cfg.model} "
                    f"chunk={len(parts_chunk)} err={type(exc).__name__}"
                )
                return self._embed_request(
                    parts_chunk[:mid],
                    task_type=task_type,
                    phase_name=phase_name,
                    is_query=is_query,
                ) + self._embed_request(
                    parts_chunk[mid:],
                    task_type=task_type,
                    phase_name=phase_name,
                    is_query=is_query,
                )

            if len(parts_chunk) == 1 and self._is_invalid_argument_error(exc):
                parts = parts_chunk[0]
                cache_key = self._cache_key(parts, is_query=is_query)
                error_text = (
                    "Gemini rejected one input instance. "
                    f"phase={phase_name} model={self.cfg.model} "
                    f"parts={self._summarize_parts(parts)} "
                    f"cache_key={cache_key}"
                )
                print(f"[gemini skip] {error_text}")
                if self.cache is not None:
                    self.cache.mark_unavailable(
                        cache_key,
                        model=self.cfg.model,
                        task_type=task_type,
                        error=error_text,
                    )
                return [None]
            if len(parts_chunk) == 1:
                parts = parts_chunk[0]
                raise RuntimeError(
                    "Gemini request failed on one input instance. "
                    f"phase={phase_name} model={self.cfg.model} "
                    f"parts={self._summarize_parts(parts)} "
                    f"cache_key={self._cache_key(parts, is_query=is_query)}"
                ) from exc
            raise

    def _validate_embeddings(self, embeddings, expected_count: int):
        if embeddings is None:
            raise RuntimeError("Gemini response is missing embeddings.")
        if len(embeddings) != expected_count:
            raise RuntimeError(
                f"Gemini returned mismatched embedding count: got={len(embeddings)} expected={expected_count}"
            )

    def _validate_vector(self, vector, *, chunk_pos: int, expected_dim: int) -> np.ndarray:
        if vector is None:
            raise RuntimeError(f"Gemini embedding at chunk_pos={chunk_pos} is None.")

        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise RuntimeError(f"Gemini embedding at chunk_pos={chunk_pos} is empty.")
        if arr.shape[0] != int(expected_dim):
            raise RuntimeError(
                "Gemini returned unexpected embedding dimensionality: "
                f"got={arr.shape[0]} expected={expected_dim} chunk_pos={chunk_pos}"
            )
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"Gemini embedding at chunk_pos={chunk_pos} contains NaN/Inf.")
        if np.all(arr == 0):
            raise RuntimeError(f"Gemini embedding at chunk_pos={chunk_pos} is all zeros.")
        if np.all(arr == -1):
            raise RuntimeError(f"Gemini embedding at chunk_pos={chunk_pos} is all -1.")

        return arr

    def embed(self, parts_batch: list[tuple[MediaPart, ...]], *, is_query: bool) -> EmbeddingBatchResult:
        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        phase_name = "query" if is_query else "document"
        output: list[np.ndarray | None] = [None] * len(parts_batch)
        skipped_indices: list[int] = []
        pending_idx: list[int] = []
        pending_parts: list[tuple[MediaPart, ...]] = []

        for idx, parts in enumerate(parts_batch):
            cache_key = self._cache_key(parts, is_query=is_query)
            status, cached = (None, None) if self.cache is None else self.cache.lookup(cache_key)
            if status == "ok" and cached is not None:
                output[idx] = cached
                continue
            if status == "unavailable":
                skipped_indices.append(idx)
                continue
            pending_idx.append(idx)
            pending_parts.append(parts)

        pending_total = len(pending_parts)
        total = len(parts_batch)
        print(
            f"[gemini cache] phase={phase_name} model={self.cfg.model} left={pending_total}/{total}"
        )

        if pending_total > 0:
            cached_now = 0
            progress = tqdm(
                total=pending_total,
                desc=f"embed:{phase_name}",
                unit="item",
                leave=False,
            )
            try:
                for idx_chunk, parts_chunk in zip(
                    chunked(pending_idx, self.cfg.batch_size),
                    chunked(pending_parts, self.cfg.batch_size),
                ):
                    vectors = self._embed_request(
                        list(parts_chunk),
                        task_type=task_type,
                        phase_name=phase_name,
                        is_query=is_query,
                    )

                    for original_idx, parts, vector in zip(idx_chunk, parts_chunk, vectors):
                        if vector is None:
                            skipped_indices.append(original_idx)
                            continue
                        output[original_idx] = vector
                        if self.cache is not None:
                            self.cache.put(
                                self._cache_key(parts, is_query=is_query),
                                model=self.cfg.model,
                                task_type=task_type,
                                vector=vector,
                            )
                            cached_now += 1
                    progress.update(len(parts_chunk))
            finally:
                progress.close()
            print(
                f"[gemini cache] phase={phase_name} model={self.cfg.model} cached_now={cached_now}"
            )

        kept_indices = [idx for idx, vector in enumerate(output) if vector is not None]
        rows = [vector for vector in output if vector is not None]
        if rows:
            matrix = np.stack(rows, axis=0)
        else:
            matrix = np.zeros((0, int(self.cfg.output_dimensionality)), dtype=np.float32)
        if self.cfg.normalize and matrix.shape[0] > 0:
            matrix = normalize_rows(matrix)
        return EmbeddingBatchResult(
            vectors=matrix,
            kept_indices=kept_indices,
            skipped_indices=sorted(set(skipped_indices)),
        )


@dataclass
class MockEmbedderConfig:
    output_dimensionality: int = 128
    normalize: bool = True


class MockEmbedder:
    def __init__(self, cfg: MockEmbedderConfig):
        self.cfg = cfg

    def embed(self, parts_batch: list[tuple[MediaPart, ...]], *, is_query: bool) -> EmbeddingBatchResult:
        if not parts_batch:
            return EmbeddingBatchResult(
                vectors=np.zeros((0, int(self.cfg.output_dimensionality)), dtype=np.float32),
                kept_indices=[],
                skipped_indices=[],
            )
        rows: list[np.ndarray] = []
        for parts in parts_batch:
            key = media_signature(parts) + (":q" if is_query else ":d")
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            seed = int(digest[:16], 16) ^ int(digest[-16:], 16)
            rng = np.random.default_rng(seed)
            row = rng.standard_normal(self.cfg.output_dimensionality, dtype=np.float32)
            rows.append(row)
        matrix = np.stack(rows, axis=0)
        if self.cfg.normalize:
            matrix = normalize_rows(matrix)
        return EmbeddingBatchResult(
            vectors=matrix,
            kept_indices=list(range(len(parts_batch))),
            skipped_indices=[],
        )
