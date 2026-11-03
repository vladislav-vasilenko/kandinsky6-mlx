"""Spec 14: the statistics of scripts/benchmark_report.py, on data with known answers."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("benchmark_report", ROOT / "scripts" / "benchmark_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_sign_flip_minimum_and_null():
    assert report.sign_flip_p([0.3] * 10) == pytest.approx(2 / 2**10)
    assert report.sign_flip_p([0.3] * 5) == pytest.approx(2 / 2**5)  # n = 5 can never reach 0.05
    assert report.sign_flip_p([1, -1, 1, -1]) == 1.0


def test_holm_known_values():
    assert report.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert report.holm([0.9, 0.9]) == pytest.approx([1.0, 1.0])


def test_ratio_effect_recovers_a_known_ratio():
    rng = np.random.default_rng(0)
    base = 10 * np.exp(rng.normal(0, 0.02, 10))
    effect = report.ratio_effect(list(1.5 * base * np.exp(rng.normal(0, 0.01, 10))), list(base))
    assert effect["ratio"] == pytest.approx(1.5, rel=0.02)
    assert effect["ci_low"] < effect["ratio"] < effect["ci_high"] and effect["ci_low"] > 1.4
    assert effect["p"] == pytest.approx(2 / 2**10)


@pytest.mark.parametrize(
    ("ratio", "low", "high", "p", "expected"),
    [
        (1.5, 1.4, 1.6, 0.01, "MLX lower"),
        (0.7, 0.6, 0.8, 0.01, "diffusers lower"),
        (1.02, 1.01, 1.03, 0.01, "no practically relevant difference"),
        (1.5, 1.4, 1.6, 0.2, "no demonstrated difference"),
    ],
)
def test_verdict(ratio, low, high, p, expected):
    assert report.verdict({"ratio": ratio, "ci_low": low, "ci_high": high, "p_holm": p}) == expected


def synthetic(cell_id: str, impl: str, rounds: int, seconds: float, footprint: float, status="ok", group="smoke"):
    rng = np.random.default_rng(abs(hash(cell_id)) % 2**32)
    records = []
    for index in range(rounds):
        noise = float(math.exp(rng.normal(0, 0.01)))
        run = {
            "seconds": {"denoise": seconds * noise, "video_decode": 2.0, "audio_decode": 0.5},
            "framework_peak_gib": {"denoise": footprint / 2},
            "footprint_peak_gib": {"denoise": footprint / 2, "video_decode": footprint, "audio_decode": 1.0},
            "step_seconds": [5.0, 1.0, 1.0],
        }
        cold = {**run, "run": "cold", "seconds": {"load": 3.0, **run["seconds"]}}
        warm = {**run, "run": "warm1"}
        records.append(
            {
                "cell": {"id": cell_id, "impl": impl, "group": group},
                "round": index,
                "status": status,
                "runs": [cold, warm] if status == "ok" else [],
                "footprint_peak_gib": footprint,
                "max_swap_delta_mb": 0,
                "power": "AC Power",
                "speed_limit": 100,
            }
        )
    return records


def test_report_end_to_end(tmp_path):
    pytest.importorskip("matplotlib")
    records = []
    for name, seconds in (("", 10.0), ("-i2v", 12.0)):
        records += synthetic(f"smoke/distill/mlx-bf16{name}", "mlx", 10, seconds, 30.0)
        records += synthetic(f"smoke/distill/diffusers-mps-bf16{name}", "diffusers", 10, seconds * 1.3, 40.0)
    records += synthetic(
        "smoke/distill/diffusers-mps-bf16-default-watermark", "diffusers", 1, 0, 43.5, status="guard: x"
    )
    results = tmp_path / "results"
    results.mkdir()
    (results / "results.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    out = tmp_path / "out"
    assert report.main([str(results), "--out", str(out)]) == 0
    stats = json.loads((out / "stats.json").read_text())
    time_effects = stats["primary"]["time"]
    assert len(time_effects) == 4  # two configurations, cold and warm
    for effect in time_effects:
        assert effect["ratio"] > 1.1 and effect["verdict"] == "MLX lower"
    assert "smoke/distill/diffusers-mps-bf16-default-watermark" in stats["failed_or_stopped"]
    for name in ("ratio_ci", "time_per_stage", "cold_vs_warm", "memory_per_stage"):
        assert (out / f"{name}.png").is_file() and (out / f"{name}.svg").is_file()
    assert "MLX lower" in (out / "tables.md").read_text()
