"""Measure where each framework's memory goes, one load at a time (spec 14).

Run in a fresh process per framework; after each step it records the process's physical footprint (and Torch's driver
allocation). It answers what benchmark footprints cannot separate: the libraries themselves, each component's weights,
what MLX defers by loading lazily, CPU copies left by Torch's loading, and whether releasing the DiT returns its memory.

    uv run --extra text --group reference python scripts/memory_probe.py --out benchmarks/results/m3max-48gb/memory_probe.jsonl
"""

import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts" / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def probe_mlx(args) -> list[dict]:
    steps = []

    def mark(name):
        steps.append({"step": name, "footprint_gib": benchmark.phys_footprint(os.getpid()) / GIB})

    mark("python")
    import mlx.core as mx

    sys.path.insert(0, str(ROOT / "src"))
    from kandinsky6_mlx.pipelines.t2v import Kandinsky6Pipeline

    def mark_mlx(name):
        mark(name)
        steps[-1] |= {"active_gib": mx.get_active_memory() / GIB, "cache_gib": mx.get_cache_memory() / GIB}

    mark_mlx("import mlx and the pipeline")
    pipeline = Kandinsky6Pipeline.from_converted(args.weights, mx.bfloat16)
    mark_mlx("from_converted (lazy media)")
    mx.eval(pipeline.transformer.parameters())
    mark_mlx("DiT evaluated")
    mx.eval(pipeline.vae.parameters())
    mark_mlx("VAE decoder evaluated")
    mx.eval(pipeline.audio_vae.parameters())
    mark_mlx("audio VAE decoder evaluated")
    mx.eval(pipeline.vocoder.parameters())
    mark_mlx("vocoder evaluated")
    pipeline.transformer = None
    gc.collect()
    mx.clear_cache()
    mark_mlx("DiT released, cache cleared")
    return steps


def probe_torch(args) -> list[dict]:
    steps = []

    def mark(name):
        steps.append({"step": name, "footprint_gib": benchmark.phys_footprint(os.getpid()) / GIB})

    mark("python")
    import torch
    from diffusers import AutoencoderKLHunyuanVideo, Kandinsky6Transformer3DModel  # noqa: F401
    from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE
    from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder

    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    from reference.record_audio_real import load_reference as load_audio
    from reference.record_video_real import load_reference as load_vae

    device = args.device
    mps = device == "mps"

    def mark_torch(name):
        if mps:
            torch.mps.synchronize()
        mark(name)
        if mps:
            steps[-1] |= {
                "driver_gib": torch.mps.driver_allocated_memory() / GIB,
                "allocated_gib": torch.mps.current_allocated_memory() / GIB,
            }

    mark_torch("import torch and diffusers")
    transformer = Kandinsky6Transformer3DModel.from_pretrained(
        args.checkpoint / "transformer", torch_dtype=torch.bfloat16
    )
    mark_torch("DiT on the CPU")
    transformer = transformer.to(device)
    gc.collect()
    mark_torch("DiT on the device")
    vae = load_vae(args.weights / "vae").to(device)
    gc.collect()
    mark_torch("VAE (with encoder) on the device")
    audio_vae = load_audio(args.weights / "audio_vae", "audio_vae", MMAudioVAE).to(device)
    gc.collect()
    mark_torch("audio VAE (with encoder) on the device")
    vocoder = load_audio(args.weights / "vocoder", "vocoder", MMAudioVocoder).to(device)
    gc.collect()
    mark_torch("vocoder on the device")
    del transformer
    gc.collect()
    if mps:
        torch.mps.empty_cache()
    mark_torch("DiT released, cache emptied")
    del vae, audio_vae, vocoder
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--framework", choices=("mlx", "torch"))
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/lite-distill")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/lite-distill-diffusers")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.framework is None:
        for framework in ("mlx", "torch"):  # a fresh process each, or the first would inflate the second
            command = [sys.executable, __file__, "--framework", framework, "--weights", str(args.weights)]
            command += ["--checkpoint", str(args.checkpoint), "--device", args.device, "--out", str(args.out)]
            subprocess.run(command, check=True)
        return 0
    steps = probe_mlx(args) if args.framework == "mlx" else probe_torch(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as handle:
        handle.write(json.dumps({"framework": args.framework, "device": args.device, "steps": steps}) + "\n")
    for step in steps:
        print(args.framework, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in step.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
