"""Contracts for the regression harness; these do not require a database."""

import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "fts_support", Path(__file__).parents[1] / "integration_tests/embedded_fulltext_support.py"
)
fts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fts)


def report(scenario):
    return {
        "scenario": scenario,
        "completed": True,
        "errors": 0,
        "rows": fts.ROWS,
        "first_pass_successes": fts.QUERIES,
        "fts_successes": fts.QUERIES * fts.PASSES,
        "worker_successes": [1] * fts.WORKERS,
        "freeze_successes": 1,
        "elapsed_seconds": fts.SECONDS,
    }


@pytest.mark.parametrize("scenario", ["first-pass", "scan-freeze"])
def test_valid_report(scenario):
    fts.validate_report(report(scenario), scenario)


@pytest.mark.parametrize(
    "scenario,field,value",
    [
        ("first-pass", "first_pass_successes", 299),
        ("first-pass", "fts_successes", 1199),
        ("first-pass", "errors", 1),
        ("first-pass", "completed", False),
        ("scan-freeze", "errors", 1),
        ("scan-freeze", "worker_successes", [0] * 8),
        ("scan-freeze", "worker_successes", [1] * 7),
        ("scan-freeze", "freeze_successes", 0),
        ("scan-freeze", "elapsed_seconds", 74),
        ("scan-freeze", "rows", 0),
    ],
)
def test_failed_or_incomplete_work_cannot_pass(scenario, field, value):
    value_report = report(scenario)
    value_report[field] = value
    with pytest.raises(AssertionError):
        fts.validate_report(value_report, scenario)


def test_wrapped_numeric_error_is_not_inferred_from_prose():
    error = RuntimeError("Failed to execute query")
    error.__cause__ = RuntimeError("execute failed: code=4016")
    assert fts.error_details(error)["error_codes"] == [4016]
    assert fts.error_details(RuntimeError("internal error"))["error_codes"] == []


def test_released_binding_symbolic_error_preserves_numeric_code():
    error = RuntimeError("Failed to execute query")
    error.__cause__ = RuntimeError("execute sql failed OB_ERR_UNEXPECTED(4016): %s")
    assert fts.error_details(error)["error_codes"] == [4016]
    assert fts.error_details(RuntimeError("query ordinal4016 returned no rows"))["error_codes"] == []


def test_large_rotated_log_uses_total_tail_budget(tmp_path):
    log = tmp_path / "seekdb.log.20260916"
    with log.open("wb") as stream:
        stream.seek(300 * 1024 * 1024)
        stream.write(b"\nread_barrier_:true ret=-4016\n")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    fts.capture_logs(tmp_path, evidence, "early", budget=4096, output_limit=1024)
    assert "ret=-4016" in (evidence / "early-db-context.log").read_text()
    inventory = json.loads((evidence / "early-log-inventory.json").read_text())
    assert inventory["files"][0]["read_bytes"] == 4096
    assert inventory["files"][0]["size"] > 256 * 1024 * 1024
    assert inventory["scan_capped"]
    assert (evidence / "early-db-context.log").stat().st_size <= 1024


def test_first_failure_is_captured_immediately_and_never_reset(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (tmp_path / "seekdb.log").write_text("ret=-4016 data/schema type does not match\n")
    recorder = fts.Recorder(tmp_path, evidence)
    recorder.error(RuntimeError("code=4016"), "scan")
    assert (evidence / "first-error.json").exists()
    assert (evidence / "early-db-context.log").exists()
    assert not (evidence / "summary.json").exists()  # before the end of the workload
    recorder.error(RuntimeError("code=4016"), "scan")
    assert recorder.errors == 2
    assert recorder.first["error_codes"] == [4016]


def test_early_trace_error_wins_over_noise_and_other_errors(tmp_path):
    trace = "YB427F000001-00065BA8AA327DFF-0-0"
    # All lower-priority buffers fill before the failing trace is encountered.
    noise = (
        "read_barrier_:false ordinary INFO\n" * 200
        + "read_barrier_:true release_head_memtable_\n" * 200
        + "ret=-4016 unrelated session\n" * 200
        + f"[{trace}] normal statement context\n" * 200
    )
    failure = f"[{trace}] send_error_packet ob_error=-4016\n"
    (tmp_path / "seekdb.log").write_text(noise + failure)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    recorder = fts.Recorder(tmp_path, evidence)
    # Exercise the real immediate-error entry point, not just final collection.
    original = fts.capture_logs
    from unittest.mock import patch

    with patch.object(fts, "capture_logs", side_effect=lambda *args: original(*args, output_limit=1024)):
        recorder.error(RuntimeError(f"code=4016 [{trace}]"), "scan")
    context = (evidence / "early-db-context.log").read_text()
    assert failure in context
    assert context.index(failure) < context.index("[preceding context]")
    assert not (evidence / "summary.json").exists()
    inventory = json.loads((evidence / "early-log-inventory.json").read_text())
    assert inventory["priority_matches"]["trace-error"] == 1
    assert inventory["priority_capped"]["memtable-context"]
    assert inventory["context_bytes"] <= 1024


def test_false_read_barrier_alone_does_not_consume_output(tmp_path):
    (tmp_path / "seekdb.log").write_text("read_barrier_:false ordinary INFO\n" * 100)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    fts.capture_logs(tmp_path, evidence, "early", output_limit=128)
    assert (evidence / "early-db-context.log").read_bytes() == b""


def test_scan_freeze_uses_separate_connections_and_no_vector_index(tmp_path, monkeypatch):
    # Shorten only this harness unit test, not the real integration defaults.
    for key, value in (("ROWS", 20), ("WORKERS", 2), ("SECONDS", 0.03), ("INTERVAL", 0.005)):
        monkeypatch.setattr(fts, key, value)
    clients = []
    # Fake queries do no I/O and can monopolize the GIL for the entire 30ms
    # window. Model one completed round deterministically instead of depending
    # on OS scheduling. This clock is local to the fake-client unit test only.
    first_round = threading.Event()
    round_barrier = threading.Barrier(fts.WORKERS + 1, action=first_round.set)
    monkeypatch.setattr(
        fts, "time", SimpleNamespace(time=time.time, monotonic=lambda: fts.SECONDS if first_round.is_set() else 0.0)
    )

    class Client:
        def __init__(self):
            self._server = self
            self.commands = []
            self.closed = False

        def _execute(self, sql):
            self.commands.append(sql)
            if sql == fts.SEARCH or sql == "ALTER SYSTEM MINOR FREEZE":
                round_barrier.wait(timeout=5)
            if sql.startswith("SHOW CREATE"):
                return [("fixture", "CREATE TABLE fixture (... FULLTEXT ...)")]
            if sql.startswith("SELECT COUNT"):
                return [(20,)]
            return [("doc-1", 1.0)]

        def close(self):
            self.closed = True

    def make_client():
        client = Client()
        clients.append(client)
        return client

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    recorder = fts.Recorder(tmp_path, evidence)
    value_report = {"scenario": "scan-freeze", "completed": True, "errors": 0}
    setup = Client()
    fts.scan_freeze(setup, make_client, recorder, value_report)
    fts.validate_report(value_report, "scan-freeze")
    assert recorder.errors == 0
    assert len(clients) == 3 and all(client.closed for client in clients)
    assert "VECTOR" not in setup.commands[0].upper()
    assert sum("ALTER SYSTEM MINOR FREEZE" in client.commands for client in clients) == 1
