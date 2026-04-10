from __future__ import annotations

from pathlib import Path

from .types import TaskSpec


CATALOG_DIR = Path(__file__).resolve().parent / "catalogs"
DEFAULT_CATALOG_PATHS = tuple(sorted(CATALOG_DIR.glob("mmeb_v2_*_tasks.yaml")))


def _load_one_catalog(catalog_path: Path) -> dict[str, TaskSpec]:
    import yaml

    with catalog_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    tasks: dict[str, TaskSpec] = {}
    for task_name, config in raw.items():
        tasks[task_name] = TaskSpec(
            name=task_name,
            group=str(config["group"]),
            dataset_parser=str(config["dataset_parser"]),
            dataset_split=str(config.get("dataset_split", "test")),
            media_root=str(config.get("media_root", "image-tasks")),
            source_name=(
                str(config["source_name"])
                if config.get("source_name") is not None
                else None
            ),
            media_subdir=(
                str(config["media_subdir"])
                if config.get("media_subdir") is not None
                else None
            ),
            aliases=tuple(str(item) for item in config.get("aliases", [])),
        )
    return tasks


def load_catalog(path: str | Path | None = None) -> dict[str, TaskSpec]:
    if path is not None:
        return _load_one_catalog(Path(path))

    tasks: dict[str, TaskSpec] = {}
    for catalog_path in DEFAULT_CATALOG_PATHS:
        tasks.update(_load_one_catalog(catalog_path))
    return tasks
