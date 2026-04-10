from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence


PartKind = Literal["text", "image", "video", "audio", "pdf"]


@dataclass(frozen=True)
class MediaPart:
    kind: PartKind
    value: str
    mime_type: str | None = None


@dataclass(frozen=True)
class Candidate:
    name: str
    parts: tuple[MediaPart, ...]


@dataclass(frozen=True)
class QueryExample:
    query_id: str
    parts: tuple[MediaPart, ...]
    labels: tuple[str, ...]
    candidate_names: tuple[str, ...]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    group: str
    dataset_parser: str
    dataset_split: str = "test"
    media_root: str = "image-tasks"
    source_name: str | None = None
    media_subdir: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass
class TaskDataset:
    spec: TaskSpec
    queries: list[QueryExample] = field(default_factory=list)
    corpus: list[Candidate] = field(default_factory=list)

    @property
    def candidate_names(self) -> list[str]:
        return [candidate.name for candidate in self.corpus]

    def label_set_sizes(self) -> Sequence[int]:
        return [len(query.labels) for query in self.queries]
