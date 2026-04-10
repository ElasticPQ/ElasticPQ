from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np


class EmbeddingCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                task_type TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL
            )
            """
        )
        columns = {
            row[1] for row in self.conn.execute("PRAGMA table_info(embeddings)").fetchall()
        }
        if "status" not in columns:
            self.conn.execute("ALTER TABLE embeddings ADD COLUMN status TEXT NOT NULL DEFAULT 'ok'")
        if "error" not in columns:
            self.conn.execute("ALTER TABLE embeddings ADD COLUMN error TEXT")
        self.conn.commit()

    def lookup(self, cache_key: str) -> tuple[str | None, np.ndarray | None]:
        row = self.conn.execute(
            "SELECT dim, vector, COALESCE(status, 'ok') FROM embeddings WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None, None
        dim, blob, status = row
        if status != "ok":
            return str(status), None
        arr = np.frombuffer(blob, dtype=np.float32)
        return "ok", np.asarray(arr.reshape(dim), dtype=np.float32)

    def get(self, cache_key: str) -> np.ndarray | None:
        status, vector = self.lookup(cache_key)
        if status != "ok":
            return None
        return vector

    def put(self, cache_key: str, *, model: str, task_type: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO embeddings (cache_key, model, task_type, dim, vector, status, error)
            VALUES (?, ?, ?, ?, ?, 'ok', NULL)
            """,
            (cache_key, model, task_type, int(vector.shape[0]), sqlite3.Binary(vector.tobytes())),
        )
        self.conn.commit()

    def mark_unavailable(self, cache_key: str, *, model: str, task_type: str, error: str) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO embeddings (cache_key, model, task_type, dim, vector, status, error)
            VALUES (?, ?, ?, 0, ?, 'unavailable', ?)
            """,
            (cache_key, model, task_type, sqlite3.Binary(b""), error[:2000]),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
