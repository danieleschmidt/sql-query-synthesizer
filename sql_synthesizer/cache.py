"""Simple in-memory TTL cache utilities."""

from __future__ import annotations

import time
from typing import Any, Dict


class TTLCache:
    """A minimal time-based cache."""

    def __init__(self, ttl: int = 0) -> None:
        self.ttl = ttl
        self._items: Dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any:
        now = time.time()
        ts_val = self._items.get(key)
        if ts_val and (self.ttl <= 0 or now - ts_val[0] <= self.ttl):
            return ts_val[1]
        if key in self._items:
            del self._items[key]
        raise KeyError(key)

    def set(self, key: str, value: Any) -> None:
        self._items[key] = (time.time(), value)

    def clear(self) -> None:
        self._items.clear()
