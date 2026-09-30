from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path


class AuditProgress:
    """Keep progress outside stdout, which is reserved for the final report."""

    def __init__(self, path: Path | None, interval: float = 10.0) -> None:
        self.path = path
        self.interval = interval
        self.phase = "starting"
        self.started = time.monotonic()
        self.last_write = 0.0
        self.completed = 0
        self.total = 0

    def begin(self, phase: str, total: int) -> None:
        self.phase = phase
        self.started = time.monotonic()
        self.completed = 0
        self.total = total
        self.write(force=True)

    def update(self, completed: int, total: int) -> None:
        self.completed, self.total = completed, total
        self.write(force=completed == total)

    def write(
        self, *, force: bool = False, status: str = "running", error: str = ""
    ) -> None:
        now = time.monotonic()
        if not force and now - self.last_write < self.interval:
            return
        elapsed = now - self.started
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        payload = {
            "status": status,
            "phase": self.phase,
            "completed": self.completed,
            "total": self.total,
            "elapsedSeconds": elapsed,
            "itemsPerSecond": rate,
            "phaseRemainingSeconds": (self.total - self.completed) / rate
            if rate
            else None,
            "updatedAt": datetime.now(UTC).isoformat(),
        }
        if error:
            payload["error"] = error
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False)
            try:
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)
        self.last_write = now
