"""Measure where each framework's memory goes, one load at a time (spec 14).

Run in a fresh process per framework; after each step it records the process's physical footprint (and Torch's driver
allocation). It answers what benchmark footprints cannot separate: the libraries themselves, each component's weights,
what MLX defers by loading lazily, CPU copies left by Torch's loading, and whether releasing the DiT returns its memory.

    uv run --extra text --group reference python scripts/memory_probe.py --out benchmarks/results/m3max-48gb/memory_probe.jsonl
"""

import argparse
import ctypes
import gc
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts" / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


UNITS = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}


def dirty_categories(pid: int, minimum_gib: float = 0.05) -> dict[str, float]:
    """What macOS's `footprint` tool counts as dirty memory per category, in GiB: malloc, IOAccelerator (GPU), mapped files."""
    out = subprocess.run(["/usr/bin/footprint", str(pid)], capture_output=True, text=True, check=False).stdout
    categories = {}
    for line in out.splitlines():
        match = re.match(r"\s*(\d+) (B|KB|MB|GB)\s+\d+ (?:B|KB|MB|GB)\s+\d+ (?:B|KB|MB|GB)\s+\d+\s+(\S.*)$", line)
        if match and int(match[1]) * UNITS[match[2]] / GIB >= minimum_gib:
            categories[match[3].strip()] = round(int(match[1]) * UNITS[match[2]] / GIB, 3)
    return categories


def relieve_allocator() -> None:
    """Ask malloc to return freed pages to the system: what this frees was freed by the program but kept by the allocator."""
    ctypes.CDLL(None).malloc_zone_pressure_relief(None, 0)


def mark_step(steps: list[dict], name: str) -> None:
    pid = os.getpid()
    steps.append({"step": name, "footprint_gib": benchmark.phys_footprint(pid) / GIB, "dirty": dirty_categories(pid)})


def probe_mlx_as_stored(args) -> list[dict]:
    """The DiT with every weight in its stored dtype, without the float32 policy for time embeddings and modulation."""
    import mlx.core as mx

    sys.path.insert(0, str(ROOT / "src"))
    from mlx.utils import tree_flatten

    from kandinsky6_mlx.models.dit.model import Kandinsky6Transformer3DModel

    steps = []
    path = args.weights / "transformer"
    model = Kandinsky6Transformer3DModel.from_config(json.loads((path / "config.json").read_text()))
    model.load_weights(str(path / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    mark_step(steps, "DiT with stored dtypes")
    steps[-1] |= {
        "active_gib": mx.get_active_memory() / GIB,
        "parameters_gib": sum(v.nbytes for _, v in tree_flatten(model.parameters())) / GIB,
    }
    return steps


def probe_mlx(args) -> list[dict]:
    steps = []

    def mark(name):
        mark_step(steps, name)

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
    from mlx.utils import tree_flatten

    by_dtype = {}
    for _, value in tree_flatten(pipeline.transformer.parameters()):
        by_dtype[str(value.dtype)] = by_dtype.get(str(value.dtype), 0) + value.nbytes / GIB
    steps[-1]["parameters_gib_by_dtype"] = by_dtype
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
    relieve_allocator()
    mark_mlx("allocator asked to return memory")
    return steps


def probe_torch(args) -> list[dict]:
    steps = []

    def mark(name):
        mark_step(steps, name)

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
    by_dtype = {}
    for parameter in transformer.parameters():
        name = str(parameter.dtype)
        by_dtype[name] = by_dtype.get(name, 0) + parameter.numel() * parameter.element_size() / GIB
    steps[-1]["parameters_gib_by_dtype"] = by_dtype
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
    relieve_allocator()
    mark_torch("allocator asked to return memory")
    del transformer
    gc.collect()
    if mps:
        torch.mps.empty_cache()
    mark_torch("DiT released, cache emptied")
    relieve_allocator()
    mark_torch("DiT released, allocator asked to return memory")
    del vae, audio_vae, vocoder
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--framework", choices=("mlx", "mlx-as-stored", "torch"))
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/lite-distill")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/lite-distill-diffusers")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.framework is None:
        for framework in (
            "mlx",
            "mlx-as-stored",
            "torch",
        ):  # a fresh process each, or the first would inflate the second
            command = [sys.executable, __file__, "--framework", framework, "--weights", str(args.weights)]
            command += ["--checkpoint", str(args.checkpoint), "--device", args.device, "--out", str(args.out)]
            subprocess.run(command, check=True)
        return 0
    steps = {"mlx": probe_mlx, "mlx-as-stored": probe_mlx_as_stored, "torch": probe_torch}[args.framework](args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as handle:
        handle.write(json.dumps({"framework": args.framework, "device": args.device, "steps": steps}) + "\n")
    for step in steps:
        print(args.framework, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in step.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
