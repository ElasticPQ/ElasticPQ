from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
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


class GeminiEmbedding2Embedder:
    def __init__(self, cfg: GeminiEmbedderConfig, *, cache: EmbeddingCache | None = None):
        self.cfg = cfg
        self.cache = cache
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for Gemini embeddings. Install dependencies with "
                "`pip install -r mmeb_v2_bench/requirements.txt`."
            ) from exc

        self._genai = genai
        self._types = types
        self._client = genai.Client()

    def _cache_key(self, parts: tuple[MediaPart, ...], is_query: bool) -> str:
        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        return f"{self.cfg.model}:{self.cfg.output_dimensionality}:{task_type}:{media_signature(parts)}"

    def _part_to_google(self, part: MediaPart):
        if part.kind == "text":
            return self._types.Part(text=part.value)
        path = Path(part.value)
        mime_type = part.mime_type or guess_mime_type(path)
        with path.open("rb") as handle:
            data = handle.read()
        return self._types.Part.from_bytes(data=data, mime_type=mime_type)

    def _parts_to_content(self, parts: tuple[MediaPart, ...]):
        return self._types.Content(parts=[self._part_to_google(part) for part in parts])

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
        if status_code == 400:
            return True

        status = getattr(exc, "status", None)
        if status == 400 or str(status).upper() == "INVALID_ARGUMENT":
            return True

        code = getattr(exc, "code", None)
        if code == 400:
            return True

        message = getattr(exc, "message", None)
        if message and "INVALID_ARGUMENT" in str(message).upper():
            return True

        text = str(exc)
        return "INVALID_ARGUMENT" in text.upper()

    def _embed_request(
        self,
        parts_chunk: list[tuple[MediaPart, ...]],
        *,
        task_type: str,
        phase_name: str,
        is_query: bool,
    ) -> list[np.ndarray | None]:
        try:
            response = self._client.models.embed_content(
                model=self.cfg.model,
                contents=[self._parts_to_content(parts) for parts in parts_chunk],
                config=self._types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.cfg.output_dimensionality,
                ),
            )
            self._validate_embeddings(response.embeddings, expected_count=len(parts_chunk))
            vectors: list[np.ndarray] = []
            for chunk_pos, embedding in enumerate(response.embeddings):
                values = getattr(embedding, "values", None)
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
