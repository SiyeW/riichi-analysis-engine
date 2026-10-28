"""Atomic publication with bounded handling of brief file-sharing conflicts."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


def temporary_sibling(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )


def replace_file(
    source: Path,
    destination: Path,
    attempts: int = 8,
    retry_delay: float = 0.01,
) -> None:
    """Retry sharing/access conflicts, never unrelated errors or forever.

    The caller owns the temporary file, including preserving it on exhaustion.
    No training random generator is used by the deterministic backoff.
    """
    if attempts < 1 or retry_delay < 0:
        raise ValueError("invalid replacement retry policy")
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            sharing = isinstance(error, PermissionError) or getattr(
                error, "winerror", None
            ) in {5, 32, 33}
            if not sharing or attempt + 1 == attempts:
                raise
            time.sleep(retry_delay * min(attempt + 1, 8))


def publish_json(
    destination: Path,
    value: object,
    *,
    attempts: int = 20,
    retry_delay: float = 0.025,
) -> None:
    """Flush a unique sibling before replacement; retain evidence on failure."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_sibling(destination)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    replace_file(temporary, destination, attempts, retry_delay)
