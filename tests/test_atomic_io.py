import json
import os
import random
import threading
import time
from pathlib import Path

import pytest

from riichi_analysis_engine import atomic_io


def test_json_publication_retries_without_changing_rng(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text('{"old": true}', encoding="utf-8")
    original = os.replace
    attempts = []

    def transient(source, destination):
        attempts.append(Path(source))
        if len(attempts) < 3:
            assert json.loads(target.read_text()) == {"old": True}
            raise PermissionError("reader is open")
        original(source, destination)

    monkeypatch.setattr(atomic_io.os, "replace", transient)
    before = random.getstate()
    atomic_io.publish_json(target, {"new": True}, retry_delay=0)
    assert random.getstate() == before
    assert json.loads(target.read_text()) == {"new": True}
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert not attempts[0].exists()


def test_exhaustion_preserves_old_target_and_complete_temporary(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_bytes(b"old bytes")
    calls = []

    def blocked(source, destination):
        calls.append(Path(source))
        raise PermissionError("still open")

    monkeypatch.setattr(atomic_io.os, "replace", blocked)
    with pytest.raises(PermissionError):
        atomic_io.publish_json(target, {"new": 2}, attempts=2, retry_delay=0)
    assert target.read_bytes() == b"old bytes"
    assert len(calls) == 2 and calls[0] == calls[1]
    assert json.loads(calls[0].read_text()) == {"new": 2}


def test_unrelated_errors_are_not_retried(tmp_path, monkeypatch):
    calls = []

    def missing(*args):
        calls.append(args)
        raise FileNotFoundError("missing source")

    monkeypatch.setattr(atomic_io.os, "replace", missing)
    with pytest.raises(FileNotFoundError):
        atomic_io.replace_file(tmp_path / "a", tmp_path / "b", retry_delay=0)
    assert len(calls) == 1


def test_temporary_names_are_siblings_and_collision_free(tmp_path):
    target = tmp_path / "checkpoint.pt"
    first = atomic_io.temporary_sibling(target)
    second = atomic_io.temporary_sibling(target)
    assert first != second
    assert first.parent == second.parent == target.parent


@pytest.mark.skipif(os.name != "nt", reason="actual Windows destination sharing")
def test_actual_windows_reader_release_allows_bounded_publication(tmp_path):
    target = tmp_path / "state.json"
    target.write_text('{"old": true}', encoding="utf-8")
    opened = threading.Event()

    def reader():
        with target.open("rb"):
            opened.set()
            time.sleep(0.15)

    thread = threading.Thread(target=reader)
    thread.start()
    assert opened.wait(2)
    try:
        atomic_io.publish_json(target, {"new": True})
    finally:
        thread.join(3)
    assert not thread.is_alive()
    assert json.loads(target.read_text()) == {"new": True}


@pytest.mark.parametrize("attempts,delay", [(0, 0), (1, -1)])
def test_invalid_policy_is_rejected(tmp_path, attempts, delay):
    with pytest.raises(ValueError):
        atomic_io.replace_file(tmp_path / "a", tmp_path / "b", attempts, delay)
