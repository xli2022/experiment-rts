"""Evaluation reports remain atomic while Windows readers briefly hold them."""

import json
from pathlib import Path

import pytest

from rtsml import evaluate


def report_paths(tmp_path):
    output = tmp_path / "evaluation.json"
    output.write_text('{"state": "running"}\n')
    return output, output.with_suffix(".json.tmp")


def test_report_replace_retries_transient_permission_failure_then_publishes_atomically(tmp_path, monkeypatch):
    output, temporary = report_paths(tmp_path)
    report = {"state": "completed", "matches": [{"won": True}]}
    actual_replace = Path.replace
    calls = []
    sleeps = []

    def reader_locked(source, target):
        calls.append((source, target))
        assert json.loads(output.read_text()) == {"state": "running"}
        assert json.loads(temporary.read_text()) == report
        if len(calls) <= 2:
            raise PermissionError("a reader briefly holds the target")
        return actual_replace(source, target)

    monkeypatch.setattr(Path, "replace", reader_locked)
    monkeypatch.setattr(evaluate.time, "sleep", sleeps.append)
    evaluate.write_report(output, report)
    assert calls == [(temporary, output)] * 3
    assert sleeps == [0.05, 0.1]
    assert json.loads(output.read_text()) == report
    assert not temporary.exists()


def test_report_replace_does_not_retry_unrelated_io_errors(tmp_path, monkeypatch):
    output, temporary = report_paths(tmp_path)
    calls = []

    def fail(source, target):
        calls.append((source, target))
        raise OSError("unrelated IO failure")

    monkeypatch.setattr(Path, "replace", fail)
    monkeypatch.setattr(evaluate.time, "sleep", lambda _: pytest.fail("unrelated errors must not retry"))
    with pytest.raises(OSError, match="unrelated IO failure"):
        evaluate.write_report(output, {"state": "completed"})
    assert calls == [(temporary, output)]
    assert json.loads(output.read_text()) == {"state": "running"}
    assert json.loads(temporary.read_text()) == {"state": "completed"}


def test_persistent_permission_failure_is_bounded_and_retains_both_reports(tmp_path, monkeypatch):
    output, temporary = report_paths(tmp_path)
    report = {"state": "completed", "matches": [{"won": False}]}
    calls = []
    sleeps = []

    def locked(source, target):
        calls.append((source, target))
        raise PermissionError("target stays locked")

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(evaluate.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError, match="target stays locked"):
        evaluate.write_report(output, report)
    assert calls == [(temporary, output)] * 10
    assert len(sleeps) == 9 and sum(sleeps) == pytest.approx(1.75)
    assert json.loads(output.read_text()) == {"state": "running"}
    assert json.loads(temporary.read_text()) == report
