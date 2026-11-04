"""Stop the benchmark and put the Mac to sleep if macOS reports a dangerous thermal state.

A separate process, independent of the benchmark and of any assistant session. It reads NSProcessInfo.thermalState, which
needs no sudo: 0 nominal, 1 fair, 2 serious, 3 critical. It stops on critical at once and on serious held for five minutes.
Nothing restarts afterwards: the owner decides when to continue. A failed reading is logged and never stops anything.

    nohup /usr/bin/python3 scripts/thermal_watchdog.py --log benchmarks/results/m3max-48gb/thermal.log >/dev/null 2>&1 &

--dry-run --fake-state 3 shows what it would stop, without stopping anything.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import time

# Scripts first, so that nothing starts the next job; the workers last.
WATCHED = [
    r"benchmarks/results/m3max-48gb/resume\.sh",
    r"benchmarks/results/m3max-48gb/after_all\.sh",
    r"benchmarks/results/m3max-48gb/run_all\.sh",
    r"caffeinate -dims uv run .*scripts/benchmark\.py",
    r"uv run .*scripts/(benchmark|generate_comparison|memory_probe)\.py",
    r"scripts/(benchmark|generate_comparison|memory_probe)\.py",
]
SERIOUS_SECONDS = 300.0


def thermal_state() -> int:
    ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.objc_msgSend.restype = ctypes.c_void_p
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    info = objc.objc_msgSend(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
    return objc.objc_msgSend(info, objc.sel_registerName(b"thermalState")) or 0


def stop_reason(state: int, serious_since: float | None, now: float) -> str | None:
    if state >= 3:
        return "critical"
    if state == 2 and serious_since is not None and now - serious_since >= SERIOUS_SECONDS:
        return "serious for 5 minutes"
    return None


def matching(pattern: str) -> list[int]:
    # -a: by default pgrep leaves out its ancestors, which include the queue script when that script started this watchdog.
    out = subprocess.run(["/usr/bin/pgrep", "-a", "-f", pattern], capture_output=True, text=True, check=False).stdout
    return [pid for pid in map(int, out.split()) if pid != os.getpid()]


def stop_processes(patterns: list[str], dry_run: bool = False, grace: float = 5.0) -> list[int]:
    stopped = []
    for pattern in patterns:
        for pid in matching(pattern):
            stopped.append(pid)
            if not dry_run:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    if dry_run:
        return stopped
    time.sleep(grace)
    for pattern in patterns:
        for pid in matching(pattern):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return stopped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", required=True)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fake-state", type=int)
    args = parser.parse_args()

    def log(message: str) -> None:
        with open(args.log, "a") as handle:
            handle.write(f"{time.strftime('%F %T')} {message}\n")

    log(f"started, pid {os.getpid()}, state {thermal_state()}")
    serious_since, previous, idle = None, None, 0
    while True:
        try:
            state = thermal_state() if args.fake_state is None else args.fake_state
        except Exception as error:  # noqa: BLE001 (any failed reading must not stop the benchmark or this watchdog)
            log(f"reading failed: {error}")
            time.sleep(args.interval)
            continue
        if state != previous:
            log(f"state {state}")
            previous = state
        now = time.monotonic()
        serious_since = (serious_since or now) if state == 2 else None
        reason = stop_reason(state, serious_since, now)
        if reason:
            pids = stop_processes(WATCHED, dry_run=args.dry_run)
            log(f"STOP: {reason}; stopped {pids}{' (dry run)' if args.dry_run else ''}")
            if not args.dry_run:
                done = subprocess.run(["/usr/bin/pmset", "sleepnow"], capture_output=True, text=True, check=False)
                log(f"pmset sleepnow: exit {done.returncode} {done.stdout.strip()} {done.stderr.strip()}")
            return 0
        idle = 0 if any(matching(p) for p in WATCHED[:3]) else idle + 1
        if idle >= 3:
            log("benchmark and queue finished, exiting")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
