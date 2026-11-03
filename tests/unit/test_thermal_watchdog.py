"""Spec 14: the benchmark's thermal stop, without touching real jobs or sleeping the machine."""

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("thermal_watchdog", ROOT / "scripts" / "thermal_watchdog.py")
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


def test_stop_rules():
    assert watchdog.stop_reason(0, None, 1000.0) is None
    assert watchdog.stop_reason(1, None, 1000.0) is None
    assert watchdog.stop_reason(2, 1000.0, 1299.0) is None
    assert watchdog.stop_reason(2, 1000.0, 1300.0) == "serious for 5 minutes"
    assert watchdog.stop_reason(3, None, 1000.0) == "critical"


def test_reads_a_valid_state():
    assert watchdog.thermal_state() in (0, 1, 2, 3)


def test_stop_processes_ends_only_what_matches():
    target = subprocess.Popen(["/bin/sleep", "98761"])
    other = subprocess.Popen(["/bin/sleep", "98762"])
    try:
        assert watchdog.stop_processes(["sleep 98761"], dry_run=True) == [target.pid]
        assert target.poll() is None
        assert watchdog.stop_processes(["sleep 98761"], grace=0.2) == [target.pid]
        assert target.wait(timeout=5) != 0
        assert other.poll() is None
    finally:
        target.kill()
        other.kill()
