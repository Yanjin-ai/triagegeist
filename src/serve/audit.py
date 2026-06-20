"""Append-only audit log — accountability for a multi-party supervisable decision.

Every served decision and every clinician override is written as one JSON line with
a timestamp and a monotonic id. This is the minimal substrate real CDS needs: a
tamper-evident-ish record of what the model suggested and what a human did with it.
(Synthetic data only; not a production-grade secure audit store.)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import ROOT

DEFAULT_PATH = ROOT / "reports" / "audit_log.jsonl"


class AuditLog:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _next_id(self) -> int:
        if not self.path.exists():
            return 1
        with open(self.path) as f:
            return sum(1 for _ in f) + 1

    def record(self, event_type: str, payload: dict) -> dict:
        entry = {"id": self._next_id(), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "event": event_type, **payload}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def tail(self, n: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            lines = f.readlines()
        return [json.loads(x) for x in lines[-n:]]
