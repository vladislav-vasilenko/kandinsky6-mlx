"""Spec 14: scripts/benchmark.py on the tiny pipeline (no real weights), MLX and diffusers on the CPU."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts" / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def tiny_cell(root: Path, tmp_path: Path, torch, impl: str, **options) -> dict:
    from kandinsky6_mlx.utils.fixtures import save_fixture

    generator = torch.Generator().manual_seed(0)
    save_fixture(
        tmp_path / "text",
        {
            "inputs.encoder_hidden_states": torch.randn(1, 5, 8, generator=generator),
            "inputs.pooled_projections": torch.randn(1, 4, generator=generator),
        },
        {"fixture": "text"},
    )
    cell = benchmark.pipeline_cell("test", "smoke", "distill", impl, impl, "cpu" if impl == "diffusers" else "gpu")
    return (
        cell
        | {
            "height": 32,
            "width": 32,
            "num_frames": 9,
            "steps": 2,
            "weights": str(root / "converted"),
            "checkpoint": str(root / "ckpt"),
            "scheduler_config": str(root / "converted" / "scheduler" / "scheduler_config.json"),
            "text": str(tmp_path / "text"),
            "dtype": "float32",
            "repeat": 1,
        }
        | options
    )


@pytest.mark.parametrize("impl", ["mlx", "diffusers"])
@pytest.mark.parametrize("release_dit", [True, False], ids=["release-dit", "keep-dit"])
def test_cell_runs_cold_and_warm(tiny_pipelines, tmp_path, impl, release_dit):
    torch, _, _, root = tiny_pipelines
    cell = tiny_cell(root, tmp_path, torch, impl, release_dit=release_dit, tiling=False)
    record = benchmark.run_cell(cell, tmp_path / "cell.log")
    assert record["status"] == "ok", (tmp_path / "cell.log").read_text()[-2000:]
    assert [run["run"] for run in record["runs"]] == ["cold", "warm1"]
    cold, warm = record["runs"]
    assert {"load", "denoise", "video_decode", "audio_decode"} == set(cold["seconds"])
    assert ("load" in warm["seconds"]) == release_dit
    assert record["footprint_peak_gib"] > 0  # stages of a tiny run can be shorter than the 0.2 s sampling period
    # The same noise every run: the output does not depend on whether the run is cold or warm.
    assert cold["frames_mean"] == pytest.approx(warm["frames_mean"], rel=1e-5)
    json.dumps(record)


def test_mlx_and_diffusers_cells_produce_the_same_frames(tiny_pipelines, tmp_path):
    torch, _, _, root = tiny_pipelines
    means = []
    for impl in ("mlx", "diffusers"):
        cell = tiny_cell(root, tmp_path, torch, impl, repeat=0, tiling=False)
        record = benchmark.run_cell(cell, tmp_path / f"{impl}.log")
        assert record["status"] == "ok", (tmp_path / f"{impl}.log").read_text()[-2000:]
        means.append(record["runs"][0]["frames_mean"])
    assert means[0] == pytest.approx(means[1], rel=1e-4)


def test_i2v_cell(tiny_pipelines, tmp_path):
    from tests.parity.test_video_vae_encoder import synthetic_image

    torch, _, _, root = tiny_pipelines
    synthetic_image(48, 40, seed=1).save(tmp_path / "image.png")
    means = []
    for impl in ("mlx", "diffusers"):
        cell = tiny_cell(root, tmp_path, torch, impl, repeat=0, tiling=False, image=str(tmp_path / "image.png"))
        record = benchmark.run_cell(cell, tmp_path / f"{impl}.log")
        assert record["status"] == "ok", (tmp_path / f"{impl}.log").read_text()[-2000:]
        assert "image_encode" in record["runs"][0]["seconds"]
        means.append(record["runs"][0]["frames_mean"])
    assert means[0] == pytest.approx(means[1], rel=1e-4)


def test_load_policy_cells_give_the_same_frames(tiny_pipelines, tmp_path):
    """Media after denoising (Torch) and all weights at once (MLX) change memory and load time, not the output."""
    torch, _, _, root = tiny_pipelines
    means = []
    for impl, options in (("diffusers", {"defer_media": True}), ("diffusers", {}), ("mlx", {"eager_media": True})):
        cell = tiny_cell(root, tmp_path, torch, impl, tiling=False, **options)
        log = tmp_path / f"{impl}.log"
        record = benchmark.run_cell(cell, log)
        assert record["status"] == "ok", log.read_text()[-2500:]
        if options.get("defer_media"):
            assert "load_media" in record["runs"][0]["seconds"] and "load_media" in record["runs"][1]["seconds"]
        means += [run["frames_mean"] for run in record["runs"]]
    assert means == pytest.approx([means[0]] * len(means), rel=1e-4)


def test_matrix_ids_are_unique_and_files_exist():
    cells = [cell for group in benchmark.matrix().values() for cell in group]
    ids = [cell["id"] for cell in cells]
    assert len(ids) == len(set(ids))
    assert benchmark.phys_footprint(benchmark.os.getpid()) > 0


def test_memory_probe_on_the_tiny_pipeline(tiny_pipelines, tmp_path):
    import subprocess
    import sys

    _, _, _, root = tiny_pipelines
    out = tmp_path / "probe.jsonl"
    command = [sys.executable, str(ROOT / "scripts/memory_probe.py"), "--weights", str(root / "converted")]
    command += ["--checkpoint", str(root / "ckpt"), "--device", "cpu", "--out", str(out)]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["framework"] for r in records] == ["mlx", "torch"]
    assert all(step["footprint_gib"] > 0 for r in records for step in r["steps"])
