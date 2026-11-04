"""Compare the port's float32 rule with diffusers' on real weights (spec 14).

The port holds every time embedding and modulation in float32; diffusers, through its exact name matching, holds only 4 DiT
tensors. Each run is a fresh process with the same recorded inputs, compared with the float32 reference and Torch bf16's drift.

    uv run --extra text --group reference python scripts/compare_fp32_rule.py --out benchmarks/results/m3max-48gb/fp32_rule.jsonl

Acceptance of the diffusers rule, fixed before any result was seen (2026-11-04), on all 6 runs:
1. video and audio rel_l2 <= 2x Torch bf16's drift (AGENTS.md);
2. frames PSNR on each seed >= Torch bf16's - 1 dB (as in tests/e2e/test_pipeline_real.py);
3. median paired PSNR loss against the port's rule <= 2 dB, the median absolute MLX-Torch difference in reports 09 and 13;
4. cosine not below the minimums in the tests of these models;
5. the owner compares frames side by side on one or two seeds.
Audio SNR is reported only: there is no Torch figure or noise estimate for it. Six pairs detect a clear loss, not its absence.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("lite-distill", seed) for seed in (42, 1, 2, 3)] + [("lite-base", seed) for seed in (42, 1)]
RULES = ("substring", "diffusers")


def fixtures(model: str, seed: int) -> tuple[Path, Path, Path, Path]:
    root = ROOT / "fixtures" / "real"
    if model == "lite-base":
        stem = f"base_smoke_seed{seed}"
    else:
        stem = "pipeline_smoke" if seed == 42 else f"pipeline_smoke_seed{seed}"
    return tuple(root / f"{stem}_{name}" for name in ("fp32", "bf16", "frames", "audio"))


def worker(model: str, seed: int, rule: str, weights_root: Path, save_dir: Path | None = None) -> dict:
    import importlib.util

    import mlx.core as mx

    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    spec = importlib.util.spec_from_file_location("pipeline_real", ROOT / "tests/e2e/test_pipeline_real.py")
    pipeline_real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_real)
    reference, drift, frames, audio = fixtures(model, seed)
    result = pipeline_real.stage_b_metrics(
        weights_root / model, reference, drift, mx.bfloat16, frames, audio, fp32_rule=rule
    )
    output, _, _ = result.pop("_outputs")
    if save_dir is not None:
        from kandinsky6_mlx.pipelines.export import write_mp4

        save_dir.mkdir(parents=True, exist_ok=True)
        write_mp4(
            save_dir / f"{model}_seed{seed}_{rule}.mp4",
            output.frames[0],
            output.frame_rate,
            output.audio[0],
            output.sample_rate,
        )
    last = result["steps"][-1]
    cosines = [step[name]["cosine"] for step in result["steps"] for name in ("video", "audio")]
    ratios = {
        name: max(step[name]["rel_l2"] / step[name]["torch_bf16_rel_l2"] for step in result["steps"])
        for name in ("video", "audio")
    }
    return {
        "model": model,
        "seed": seed,
        "rule": rule,
        "frames_psnr_db": result["frames_psnr_db"],
        "torch_bf16_frames_psnr_db": result.get("torch_bf16_frames_psnr_db"),
        "audio_snr_db": result["audio_snr_db"],
        "final_rel_l2": {name: last[name]["rel_l2"] for name in ("video", "audio")},
        "final_torch_bf16_rel_l2": {name: last[name].get("torch_bf16_rel_l2") for name in ("video", "audio")},
        "min_cosine": min(cosines),
        "max_step_rel_l2_over_torch_bf16": ratios,
        "seconds": result["seconds"],
        "peak_memory_gib": result["peak_memory_gib"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", nargs=3, metavar=("MODEL", "SEED", "RULE"), help=argparse.SUPPRESS)
    parser.add_argument("--weights-root", type=Path, default=ROOT / "weights")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rules", nargs="+", default=list(RULES), choices=RULES)
    parser.add_argument("--only", nargs="+", help="runs as model:seed, e.g. lite-distill:42")
    parser.add_argument("--save-dir", type=Path, help="write each run's MP4 here")
    args = parser.parse_args(argv)
    if args.worker:
        model, seed, rule = args.worker
        print("RESULT " + json.dumps(worker(model, int(seed), rule, args.weights_root, args.save_dir)), flush=True)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    runs = [run for run in RUNS if args.only is None or f"{run[0]}:{run[1]}" in args.only]
    for index, (model, seed) in enumerate(runs):
        # Alternate which rule goes first, so that neither always runs on a colder machine.
        for rule in args.rules if index % 2 == 0 else reversed(args.rules):
            command = [sys.executable, __file__, "--worker", model, str(seed), rule]
            command += ["--weights-root", str(args.weights_root)]
            if args.save_dir is not None:
                command += ["--save-dir", str(args.save_dir)]
            done = subprocess.run(command, capture_output=True, text=True, check=False)
            lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
            record = (
                json.loads(lines[-1][7:])
                if lines
                else {"model": model, "seed": seed, "rule": rule, "error": done.stderr[-1500:]}
            )
            failed += not lines
            with args.out.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                {k: record.get(k) for k in ("model", "seed", "rule", "frames_psnr_db", "audio_snr_db", "error")},
                flush=True,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
