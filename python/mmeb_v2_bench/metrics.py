from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricsConfig:
    k_values: tuple[int, ...] = (1, 5, 10)


def _precision_at_k(prediction: list[str], labels: set[str], k: int) -> float:
    top = prediction[:k]
    if not top:
        return 0.0
    return float(len(set(top) & labels) / k)


def _recall_at_k(prediction: list[str], labels: set[str], k: int) -> float:
    if not labels:
        return 0.0
    return float(len(set(prediction[:k]) & labels) / len(labels))


def _hit_at_k(prediction: list[str], labels: set[str], k: int) -> float:
    return float(bool(set(prediction[:k]) & labels))


def _mrr_at_k(prediction: list[str], labels: set[str], k: int) -> float:
    for rank, name in enumerate(prediction[:k], start=1):
        if name in labels:
            return 1.0 / rank
    return 0.0


def evaluate_rankings(
    *,
    predictions: list[list[str]],
    labels: list[tuple[str, ...]],
    cfg: MetricsConfig | None = None,
) -> dict[str, float]:
    cfg = cfg or MetricsConfig()
    metrics: dict[str, list[float]] = {}
    for k in cfg.k_values:
        metrics[f"hit@{k}"] = []
        metrics[f"precision@{k}"] = []
        metrics[f"recall@{k}"] = []
        metrics[f"mrr@{k}"] = []

    for prediction, label_tuple in zip(predictions, labels):
        label_set = set(label_tuple)
        for k in cfg.k_values:
            metrics[f"hit@{k}"].append(_hit_at_k(prediction, label_set, k))
            metrics[f"precision@{k}"].append(_precision_at_k(prediction, label_set, k))
            metrics[f"recall@{k}"].append(_recall_at_k(prediction, label_set, k))
            metrics[f"mrr@{k}"].append(_mrr_at_k(prediction, label_set, k))

    return {name: float(np.mean(values) if values else 0.0) for name, values in metrics.items()}
