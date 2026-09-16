"""Contracts for the regression harness; these do not require a database."""

import importlib.util
import json
from pathlib import Path

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


def test_scan_freeze_uses_separate_connections_and_no_vector_index(tmp_path, monkeypatch):
    # Shorten only this harness unit test, not the real integration defaults.
    for key, value in (("ROWS", 20), ("WORKERS", 2), ("SECONDS", 0.03), ("INTERVAL", 0.005)):
        monkeypatch.setattr(fts, key, value)
    clients = []

    class Client:
        def __init__(self):
            self._server = self
            self.commands = []
            self.closed = False

        def _execute(self, sql):
            self.commands.append(sql)
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
