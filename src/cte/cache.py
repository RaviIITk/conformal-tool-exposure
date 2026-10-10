"""Small on-disk caches so every score is computed (or paid for) once."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np


def text_key(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:24]


class ArrayCache:
    """Maps string keys to fixed-length float vectors, persisted as one .npz file."""

    def __init__(self, path: Path, dim: int):
        self.path, self.dim = path, dim
        self._rows: dict[str, np.ndarray] = {}
        if path.exists():
            z = np.load(path)
            self._rows = dict(zip(z["keys"].astype(str), z["vals"]))

    def get_many(self, keys: list[str]) -> tuple[np.ndarray, list[int]]:
        """Return (values, positions of missing keys); missing rows are NaN."""
        out = np.full((len(keys), self.dim), np.nan, dtype=np.float32)
        missing = []
        for i, k in enumerate(keys):
            v = self._rows.get(k)
            if v is None:
                missing.append(i)
            else:
                out[i] = v
        return out, missing

    def put_many(self, keys: list[str], vals: np.ndarray) -> None:
        for k, v in zip(keys, np.asarray(vals, dtype=np.float32).reshape(len(keys), self.dim)):
            self._rows[k] = v

    def save(self) -> None:
        if not self._rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._rows), dtype="S24")
        np.savez(self.path, keys=keys, vals=np.stack(list(self._rows.values())))


class JsonlCache:
    """Append-only key -> JSON record cache for API responses (thread-safe)."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._rows: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._rows[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self._rows.get(key)

    def put(self, key: str, rec: dict) -> None:
        rec = {"key": key, **rec}
        with self._lock:
            self._rows[key] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
