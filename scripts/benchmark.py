"""Benchmark the MLX Kandinsky 6 pipeline against diffusers on the same Mac (spec 14).

Every cell of the matrix runs in its own process, so its first run is a true cold start (weights read, kernels
compiled); the following runs in that process are warm. For statistics each cell is measured in several independent
processes (rounds): the process is the unit of analysis, the cells of a round run in a random order (seeded) with a
pause between them so that heating and drift do not favour one side, and all cells of a round use the same noise seed,
so MLX and diffusers pair up by round. The driver samples the worker's physical footprint from outside
(`proc_pid_rusage`, `ri_phys_footprint`, every 0.2 s), so the OS-level memory is measured the same way for MLX and
torch; the worker adds the framework's own peak. Stage boundaries travel from the worker on stdout. The driver also
guards the machine: it stops a cell if swap grows too much or free disk falls too low, and records that. Results are
appended as JSON lines, and (cell, round) pairs already present are skipped, so an interrupted benchmark continues
where it stopped.

    uv run --extra text --group reference python scripts/benchmark.py --list
    uv run --extra text --group reference python scripts/benchmark.py --matrix smoke --out benchmarks/results/m3max-48gb
    uv run --extra text --group reference python scripts/benchmark.py --only 'smoke/distill/mlx-bf16' --rounds 1

Inputs are identical on both sides: the same cached text embeddings, NumPy noise (video, audio, image sample, in that
order) and input image. Text encoding is measured in its own cells.
"""

import argparse
import ctypes
import gc
import itertools
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30
SIZES = {"smoke": (256, 432, 25), "baseline": (480, 864, 121)}
SWAP_LIMIT_MB = 4096
MIN_FREE_DISK_GB = 8
# A cell is stopped before macOS starts swapping heavily: swap can fill the disk within seconds.
MAX_FOOTPRINT_FRACTION = 0.9
QWEN_DEFAULT = Path.home() / "ComfyUI-Shared/models/text_encoders/qwen_2.5_vl_7b.safetensors"

MODELS = {
    "distill": {
        "weights": "weights/lite-distill",
        "checkpoint": "weights/lite-distill-diffusers",
        "scheduler_config": "fixtures/configs/lite-distill-scheduler.json",
        "steps": 10,
        "guidance": 1.0,
    },
    "base": {
        "weights": "weights/lite-base",
        "checkpoint": "weights/lite-base-diffusers",
        "scheduler_config": "weights/lite-base/scheduler/scheduler_config.json",
        "steps": 50,
        "guidance": 5.0,
    },
}
TEXT = {"smoke": "fixtures/real/smoke_text", "baseline": "fixtures/real/baseline_text"}
NEGATIVE = "fixtures/real/negative_text"
IMAGE = "fixtures/real_inputs/i2v_input.png"
# Independent processes per cell. An exact paired test cannot go below p = 2 / 2**n: n = 10 for the primary comparisons
# (cells with "primary") leaves room for the Holm correction over 8 tests, n = 5 describes the other cells, and Baseline
# runs take too long for more than a few (hours per run with guidance: a single, labelled measurement).
PRIMARY_ROUNDS = 10
ROUNDS = {
    "text": 5,
    "smoke": 5,
    "tradeoffs": 5,
    "profiles": 5,
    "baseline": 3,
    "profiles-baseline": 1,
    "baseline-cfg": 1,
    "torch-memory": 3,
}
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def cell_rounds(cell: dict, override: int | None = None) -> int:
    if override:
        return override
    if "rounds" in cell:
        return cell["rounds"]
    return PRIMARY_ROUNDS if cell.get("primary") else ROUNDS.get(cell["group"], 1)


ORDER_SEED = 20261103
COOLDOWN_SECONDS = 20


def pipeline_cell(group: str, size: str, model: str, name: str, impl: str, device: str = "gpu", **options) -> dict:
    """One pipeline cell. `impl` is mlx or diffusers; options override the defaults below."""
    spec = MODELS[model]
    height, width, num_frames = SIZES[size]
    cell = {
        "id": f"{group}/{model}/{name}",
        "kind": "pipeline",
        "group": group,
        "impl": impl,
        "device": device,
        "model": model,
        "preset": size,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "weights": spec["weights"],
        "checkpoint": spec["checkpoint"],
        "scheduler_config": spec["scheduler_config"],
        "steps": spec["steps"],
        "guidance": spec["guidance"],
        "text": TEXT[size],
        "negative": NEGATIVE if spec["guidance"] > 1 else None,
        "image": None,
        "dtype": "bfloat16",
        "audio": True,
        "decode": True,
        # The study notebook's configuration, on both sides: spatial VAE tiles, the DiT freed before decoding.
        "tiling": True,
        "cfg_batched": False,
        "clear_cache": True,
        "release_dit": True,
        "chunk_gib": None,
        "cache_limit_gib": None,
        # torch's default 1.7 lets MPS run far past the GPU working set into swap (51 GiB on 48 GB when probed).
        "mps_high_watermark": 1.0 if device == "mps" else None,
        "memory_limit_gib": None,
        # Warm runs per process: guided cells take minutes per run, and the rounds carry the statistics.
        "repeat": 2 if spec["steps"] <= 10 else 1,
        "seed": 42,
    }
    return cell | options


def text_cell(name: str, impl: str, **options) -> dict:
    return {
        "id": f"text/{name}",
        "kind": "text",
        "group": "text",
        "impl": impl,
        "device": "gpu",
        "text_root": "weights/lite-distill-text",
        "qwen_weights": str(QWEN_DEFAULT),
        "prompt_file": "fixtures/real_inputs/smoke_prompt.txt",
        "dtype": "bfloat16",
        "bits": None,
        "repeat": 3,
    } | options


def matrix() -> dict[str, list[dict]]:
    """The spec 14 matrix, by group. Cheap groups first; Baseline with guidance is one overnight run."""
    smoke = []
    for model in ("distill", "base"):
        smoke += [
            pipeline_cell("smoke", "smoke", model, "mlx-bf16", "mlx", primary=True),
            pipeline_cell("smoke", "smoke", model, "diffusers-mps-bf16", "diffusers", "mps", primary=True),
            pipeline_cell("smoke", "smoke", model, "mlx-fp32", "mlx", dtype="float32"),
            pipeline_cell("smoke", "smoke", model, "mlx-bf16-i2v", "mlx", image=IMAGE, primary=True),
            pipeline_cell(
                "smoke", "smoke", model, "diffusers-mps-bf16-i2v", "diffusers", "mps", image=IMAGE, primary=True
            ),
        ]
    smoke += [
        pipeline_cell("smoke", "smoke", "base", "mlx-bf16-cfg-batched", "mlx", cfg_batched=True),
        pipeline_cell("smoke", "smoke", "distill", "mlx-bf16-no-audio", "mlx", audio=False),
        pipeline_cell("smoke", "smoke", "distill", "diffusers-mps-bf16-no-audio", "diffusers", "mps", audio=False),
        pipeline_cell("smoke", "smoke", "distill", "mlx-bf16-untiled", "mlx", tiling=False),
        # Torch's default watermark does not fit 48 GB: recorded as such when the guard stops it.
        pipeline_cell(
            "smoke",
            "smoke",
            "distill",
            "diffusers-mps-bf16-default-watermark",
            "diffusers",
            "mps",
            mps_high_watermark=None,
            repeat=0,
            rounds=1,
        ),
        pipeline_cell(
            "smoke",
            "smoke",
            "distill",
            "diffusers-mps-bf16-untiled",
            "diffusers",
            "mps",
            tiling=False,
            repeat=0,
            rounds=1,
        ),
        # Context only: denoising on the CPU; the torch CPU VAE decode needs 25-40 GB and is not run.
        pipeline_cell(
            "smoke",
            "smoke",
            "distill",
            "diffusers-cpu-fp32-denoise",
            "diffusers",
            "cpu",
            dtype="float32",
            decode=False,
            audio=False,
            repeat=0,
        ),
    ]
    tradeoffs = [
        pipeline_cell("tradeoffs", "smoke", "distill", "default", "mlx"),
        pipeline_cell("tradeoffs", "smoke", "distill", "keep-dit", "mlx", release_dit=False),
        pipeline_cell("tradeoffs", "smoke", "distill", "keep-cache", "mlx", clear_cache=False),
        pipeline_cell("tradeoffs", "smoke", "distill", "untiled-keep-dit", "mlx", tiling=False, release_dit=False),
        pipeline_cell("tradeoffs", "smoke", "distill", "cache-0", "mlx", cache_limit_gib=0),
        pipeline_cell("tradeoffs", "smoke", "distill", "cache-1gib", "mlx", cache_limit_gib=1),
        pipeline_cell("tradeoffs", "smoke", "distill", "cache-4gib", "mlx", cache_limit_gib=4),
        pipeline_cell("tradeoffs", "smoke", "distill", "cache-8gib", "mlx", cache_limit_gib=8),
        pipeline_cell("tradeoffs", "smoke", "distill", "cache-16gib", "mlx", cache_limit_gib=16),
        pipeline_cell("tradeoffs", "smoke", "distill", "chunk-0.25gib", "mlx", chunk_gib=0.25),
        pipeline_cell("tradeoffs", "smoke", "distill", "chunk-4gib", "mlx", chunk_gib=4.0),
    ]
    text = [
        text_cell("mlx-bf16", "mlx"),
        text_cell("mlx-8bit", "mlx", bits=8),
        text_cell("torch-mps-bf16", "torch"),
    ]
    baseline = [
        pipeline_cell("baseline", "baseline", "distill", "mlx-bf16", "mlx", repeat=1),
        pipeline_cell("baseline", "baseline", "distill", "diffusers-mps-bf16", "diffusers", "mps", repeat=0),
    ]
    overnight = [pipeline_cell("baseline-cfg", "baseline", "base", "mlx-bf16", "mlx", repeat=0)]
    profiles, profiles_baseline = [], []
    # RAM, budget for the pipeline, MLX cache limit that keeps the OS footprint within the budget at Smoke.
    for ram, budget, cache in ((16, 10, 1), (24, 16, 4), (32, 22, 8), (48, 32, 16)):
        options = {"cache_limit_gib": cache, "memory_limit_gib": budget, "profile_ram_gb": ram}
        profiles.append(pipeline_cell("profiles", "smoke", "distill", f"smoke-{ram}gb", "mlx", **options))
        profiles_baseline.append(
            pipeline_cell("profiles-baseline", "baseline", "distill", f"baseline-{ram}gb", "mlx", repeat=0, **options)
        )
    # Torch MPS's smallest footprint, to compare with the MLX cache-limit cells: an out-of-memory exit is a result here.
    torch_memory = [
        pipeline_cell(
            "torch-memory",
            "smoke",
            "distill",
            f"watermark-{ratio}",
            "diffusers",
            "mps",
            mps_high_watermark=ratio,
            repeat=1,
        )
        for ratio in (0.5, 0.35, 0.25)
    ]
    return {
        "torch-memory": torch_memory,
        "text": text,
        "smoke": smoke,
        "tradeoffs": tradeoffs,
        "baseline": baseline,
        "profiles": profiles,
        "profiles-baseline": profiles_baseline,
        "baseline-cfg": overnight,
    }


# ------------------------------------------------------------------------------------------------- machine facts


def phys_footprint(pid: int) -> int | None:
    """`ri_phys_footprint` of a process in bytes (macOS): what Activity Monitor calls Memory, GPU buffers included."""
    if sys.platform != "darwin":
        return None
    libc = ctypes.CDLL("/usr/lib/libproc.dylib")
    buffer = ctypes.create_string_buffer(512)
    if libc.proc_pid_rusage(pid, 0, buffer) != 0:  # RUSAGE_INFO_V0
        return None
    # rusage_info_v0: 16-byte uuid, then uint64 user, system, pkg idle, interrupts, pageins, wired, resident, footprint.
    return int.from_bytes(buffer.raw[16 + 7 * 8 : 16 + 8 * 8], "little")


def swap_used_mb() -> int:
    if sys.platform != "darwin":
        return 0
    out = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | {"LC_ALL": "C"},
    ).stdout
    match = re.search(r"used = ([0-9]+)[.,]", out)
    return int(match.group(1)) if match else 0


def speed_limit() -> int | None:
    """macOS's CPU speed limit in percent (below 100 means thermal throttling), when it reports one."""
    if sys.platform != "darwin":
        return None
    out = subprocess.run(["/usr/bin/pmset", "-g", "therm"], capture_output=True, text=True, check=False).stdout
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    return int(match.group(1)) if match else 100


def power_source() -> str | None:
    if sys.platform != "darwin":
        return None
    out = subprocess.run(["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True, check=False).stdout
    match = re.search(r"'(.+?)'", out)
    return match.group(1) if match else None


def machine() -> dict:
    def sysctl(name):
        return subprocess.run(
            ["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, check=False
        ).stdout.strip()

    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    facts = {"platform": platform.platform(), "python": platform.python_version(), "commit": commit.stdout.strip()}
    if sys.platform == "darwin":
        gpu = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, check=False
        ).stdout
        match = re.search(r"Total Number of Cores: (\d+)", gpu)
        power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=False).stdout.splitlines()
        facts |= {
            "chip": sysctl("machdep.cpu.brand_string"),
            "memory_gb": int(sysctl("hw.memsize")) // GIB,
            "gpu_cores": int(match.group(1)) if match else None,
            "macos": platform.mac_ver()[0],
            "power": power[0] if power else None,
        }
    for module in ("mlx", "mlx-vlm", "torch", "diffusers", "transformers"):
        try:
            facts[module] = version(module)
        except PackageNotFoundError:
            facts[module] = None
    return facts


# --------------------------------------------------------------------------------------------------------- worker


class Stages:
    """Stage boundaries of one run: tells the driver and accumulates the seconds per stage."""

    def __init__(self):
        self.run = "setup"
        self.current = None
        self.started = 0.0
        self.seconds: dict[str, float] = {}

    def begin_run(self, run: str) -> None:
        self.end()
        self.run, self.seconds = run, {}

    def __call__(self, stage: str) -> None:
        self.end()
        self.current, self.started = stage, time.perf_counter()
        print(f"@@stage {self.run} {stage}", flush=True)

    def end(self) -> None:
        if self.current is not None:
            self.seconds[self.current] = self.seconds.get(self.current, 0.0) + time.perf_counter() - self.started
            self.current = None
        print(f"@@stage {self.run} -", flush=True)


def noise_arrays(cell: dict, audio_length: int):
    import numpy as np

    height, width, frames = cell["height"], cell["width"], cell["num_frames"]
    rng = np.random.default_rng(cell["seed"])
    video = rng.standard_normal((1, 16, (frames - 1) // 4 + 1, height // 8, width // 8)).astype(np.float32)
    audio = rng.standard_normal((1, 40, audio_length)).astype(np.float32)
    image = rng.standard_normal((1, 16, 1, height // 8, width // 8)).astype(np.float32)
    return video, audio, image


def path(value) -> Path | None:
    if value is None:
        return None
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def save_clip(cell: dict, frames, audio, sample_rate: int) -> None:
    """Write the run's video to `cell["save_mp4"]`, if asked; not part of any timed stage."""
    if cell.get("save_mp4"):
        from kandinsky6_mlx.pipelines.export import write_mp4

        write_mp4(path(cell["save_mp4"]), frames[0], 24.0, None if audio is None else audio[0], sample_rate)


def run_mlx_pipeline(cell: dict, stages: Stages) -> list[dict]:
    import mlx.core as mx

    from kandinsky6_mlx.models.video_vae.decoder import HunyuanVideoCausalConv3d, TilingConfig
    from kandinsky6_mlx.pipelines.t2v import Kandinsky6Pipeline, load_text_embeds

    if cell["memory_limit_gib"]:
        mx.set_memory_limit(int(cell["memory_limit_gib"] * GIB))
    if cell["chunk_gib"]:
        HunyuanVideoCausalConv3d.chunk_bytes = int(cell["chunk_gib"] * GIB)
    if cell.get("cache_limit_gib") is not None:
        mx.set_cache_limit(int(cell["cache_limit_gib"] * GIB))
    dtype = {"bfloat16": mx.bfloat16, "float32": mx.float32}[cell["dtype"]]
    text = load_text_embeds(path(cell["text"]))
    negative = load_text_embeds(path(cell["negative"])) if cell["negative"] else None
    runs, pipeline = [], None
    for index in range(1 + cell["repeat"]):
        stages.begin_run("cold" if index == 0 else f"warm{index}")
        framework = {}
        if pipeline is None or pipeline.transformer is None:
            stages("load")
            mx.reset_peak_memory()
            pipeline = Kandinsky6Pipeline.from_converted(path(cell["weights"]), dtype, audio=cell["audio"])
            mx.eval(pipeline.transformer.parameters())
            framework["load"] = mx.get_peak_memory() / GIB
        length = pipeline.latent_shapes(cell["height"], cell["width"], cell["num_frames"], 24.0)[1][2]
        latents, audio_latents, image_noise = noise_arrays(cell, length)
        output = pipeline(
            text,
            height=cell["height"],
            width=cell["width"],
            num_frames=cell["num_frames"],
            num_inference_steps=cell["steps"],
            guidance_scale=cell["guidance"],
            negative_text_embeds=negative,
            cfg_batched=cell["cfg_batched"],
            latents=latents,
            audio_latents=audio_latents if cell["audio"] else None,
            sample_audio=cell["audio"],
            vae_tiling=TilingConfig() if cell["tiling"] else None,
            image=path(cell["image"]),
            image_noise=image_noise if cell["image"] else None,
            release_transformer=cell["release_dit"],
            clear_cache=cell["clear_cache"],
            on_stage=stages,
        )
        stages.end()
        save_clip(cell, output.frames, output.audio, output.sample_rate)
        runs.append(
            {
                "run": stages.run,
                "seconds": dict(stages.seconds),
                "framework_peak_gib": framework | output.peak_memory_gib,
                "frames_mean": float(output.frames.mean()),
                "step_seconds": output.step_seconds,
            }
        )
    return runs


def run_diffusers_pipeline(cell: dict, stages: Stages) -> list[dict]:
    import numpy as np
    import torch

    from reference.record_pipeline_run import build_pipeline, load_text

    device = {"gpu": "mps"}.get(cell["device"], cell["device"])
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cell["dtype"]]
    mps = device == "mps"
    peak = {"value": 0}
    stop = threading.Event()

    def sample():
        while not stop.wait(0.2):
            peak["value"] = max(peak["value"], torch.mps.driver_allocated_memory())

    if mps:
        threading.Thread(target=sample, daemon=True).start()
    if cell.get("memory_limit_gib") and mps:
        limit = torch.mps.recommended_max_memory()
        torch.mps.set_per_process_memory_fraction(min(1.0, cell["memory_limit_gib"] * GIB / limit))
    embeds, pooled = load_text(path(cell["text"]))
    negative = negative_pooled = None
    if cell["negative"]:
        negative, negative_pooled = load_text(path(cell["negative"]))

    stages.begin_run("cold")
    stages("load")
    pipe = build_pipeline(
        path(cell["checkpoint"]),
        dtype,
        device,
        path(cell["weights"]),
        path(cell["scheduler_config"]),
        cell["decode"] or cell["image"] is not None,
    )
    if cell["tiling"] and pipe.vae is not None:
        pipe.vae.enable_tiling()
    from diffusers.models.autoencoders import vae as vae_module

    def take_peak() -> float:
        if not mps:
            return 0.0
        torch.mps.synchronize()
        value, peak["value"] = max(peak["value"], torch.mps.driver_allocated_memory()), 0
        return value / GIB

    runs = []
    encode_image = pipe.encode_image

    def on_step_end(times, _pipe, _step, _timestep, _tensors):
        if mps:
            torch.mps.synchronize()  # MPS runs lazily: without this a step's time lands in a later step
        times.append(time.perf_counter())
        return {}

    def reload_transformer():
        from diffusers import Kandinsky6Transformer3DModel

        transformer = Kandinsky6Transformer3DModel.from_pretrained(
            path(cell["checkpoint"]) / "transformer", torch_dtype=dtype
        )
        pipe.register_modules(transformer=transformer.to(device))

    def staged_encode_image(*args, **kwargs):
        stages("image_encode")
        result = encode_image(*args, **kwargs)
        if mps:
            torch.mps.synchronize()
        stages("denoise")
        return result

    pipe.encode_image = staged_encode_image
    for index in range(1 + cell["repeat"]):
        if index:
            stages.begin_run(f"warm{index}")
            if pipe.transformer is None:
                stages("load")
                reload_transformer()
        framework = {"load": take_peak()} if index == 0 or cell["release_dit"] else {}
        length = (
            int(
                np.ceil(
                    ((cell["num_frames"] - 1) // 4 * 4 + 1)
                    / 24.0
                    * pipe.audio_sample_rate
                    / pipe.audio_latent_hop_length
                )
            )
            if pipe.audio_vae is not None
            else 45
        )
        latents, audio_latents, image_noise = noise_arrays(cell, length)
        image = None
        if cell["image"]:
            from PIL import Image

            image = Image.open(path(cell["image"]))
            noise = torch.from_numpy(image_noise)
            vae_module.DiagonalGaussianDistribution.sample = lambda self, generator=None, n=noise: (
                self.mean + self.std * n.to(self.mean.device, self.mean.dtype)
            )
        stages("denoise")
        step_times = [time.perf_counter()]

        with torch.no_grad():
            video_latents, audio_out = pipe(
                callback_on_step_end=partial(on_step_end, step_times),
                image=image,
                prompt_embeds=embeds.to(device),
                pooled_prompt_embeds=pooled.to(device),
                negative_prompt_embeds=None if negative is None else negative.to(device),
                negative_pooled_prompt_embeds=None if negative_pooled is None else negative_pooled.to(device),
                latents=torch.from_numpy(latents).to(device),
                audio_latents=torch.from_numpy(audio_latents).to(device),
                sample_audio=cell["audio"],
                height=cell["height"],
                width=cell["width"],
                num_frames=cell["num_frames"],
                num_inference_steps=cell["steps"],
                guidance_scale=cell["guidance"],
                output_type="latent",
                return_dict=False,
            )
            framework["denoise"] = take_peak()
            if cell["release_dit"]:
                pipe.register_modules(transformer=None)
                gc.collect()
                if mps:
                    torch.mps.empty_cache()
            frames_mean = None
            if cell["decode"]:
                stages("video_decode")
                z = video_latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
                video = pipe.vae.decode(z, return_dict=False)[0]
                frames = pipe.video_processor.postprocess_video(video, output_type="np")
                frames_mean = float(np.mean(frames))
                del video, z
                framework["video_decode"] = take_peak()
                if cell["audio"]:
                    stages("audio_decode")
                    z = audio_out.to(pipe.audio_vae.dtype) / pipe.audio_vae.config.scaling_factor
                    mel = pipe.audio_vae.decode(z, return_dict=False)[0]
                    waveform = (
                        pipe.vocoder(mel.to(pipe.vocoder.dtype), return_dict=False)[0][:, 0].float().cpu().numpy()
                    )
                    framework["audio_decode"] = take_peak()
        stages.end()
        if cell["decode"]:
            save_clip(cell, frames, waveform if cell["audio"] else None, pipe.audio_sample_rate)
        runs.append(
            {
                "run": stages.run,
                "seconds": dict(stages.seconds),
                "framework_peak_gib": framework if mps else {},
                "frames_mean": frames_mean,
                "step_seconds": [b - a for a, b in itertools.pairwise(step_times)],
            }
        )
    stop.set()
    return runs


def run_text(cell: dict, stages: Stages) -> list[dict]:
    prompt = path(cell["prompt_file"]).read_text()
    root = path(cell["text_root"])
    runs = []
    if cell["impl"] == "mlx":
        import mlx.core as mx

        from kandinsky6_mlx.text.encode import TextEncoders

        stages.begin_run("cold")
        stages("load")
        mx.reset_peak_memory()
        encoders = TextEncoders.from_files(root, [path(cell["qwen_weights"])], qwen_bits=cell["bits"])
        framework = {"load": mx.get_peak_memory() / GIB}
        for index in range(1 + cell["repeat"]):
            if index:
                stages.begin_run(f"warm{index}")
                framework = {}
            stages("encode")
            mx.reset_peak_memory()
            encoders.encode_prompt(prompt)
            framework["encode"] = mx.get_peak_memory() / GIB
            stages.end()
            runs.append({"run": stages.run, "seconds": dict(stages.seconds), "framework_peak_gib": framework})
        return runs

    import inspect

    import torch
    from diffusers import Kandinsky6TI2VAPipeline
    from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

    device = "mps"
    # from_pretrained needs the weights next to the config: a folder of links to the single-file Qwen.
    linked = ROOT / "weights" / ".benchmark-qwen"
    linked.mkdir(exist_ok=True)
    for name, target in (
        ("config.json", root / "text_encoder" / "config.json"),
        ("model.safetensors", path(cell["qwen_weights"])),
    ):
        link = linked / name
        if not link.exists():
            link.symlink_to(target.resolve())
    stages.begin_run("cold")
    stages("load")
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(linked, dtype=torch.bfloat16).eval().to(device)
    clip = CLIPTextModel.from_pretrained(root / "text_encoder_2", dtype=torch.float32).eval().to(device)
    names = ("self", "text_encoder", "tokenizer", "text_encoder_2", "tokenizer_2")
    others = inspect.signature(Kandinsky6TI2VAPipeline.__init__).parameters
    pipe = Kandinsky6TI2VAPipeline(
        **{name: None for name in others if name not in names},
        text_encoder=qwen,
        tokenizer=Qwen2_5_VLProcessor.from_pretrained(root / "tokenizer"),
        text_encoder_2=clip,
        tokenizer_2=CLIPTokenizer.from_pretrained(root / "tokenizer_2"),
    )
    framework = {"load": torch.mps.driver_allocated_memory() / GIB}
    for index in range(1 + cell["repeat"]):
        if index:
            stages.begin_run(f"warm{index}")
            framework = {}
        stages("encode")
        with torch.no_grad():
            pipe.encode_prompt(
                prompt, do_classifier_free_guidance=False, device=torch.device(device), dtype=torch.float32
            )
        torch.mps.synchronize()
        framework["encode"] = torch.mps.driver_allocated_memory() / GIB
        stages.end()
        runs.append({"run": stages.run, "seconds": dict(stages.seconds), "framework_peak_gib": framework})
    return runs


def worker(cell: dict) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    stages = Stages()
    if cell["kind"] == "text":
        runs = run_text(cell, stages)
    elif cell["impl"] == "mlx":
        runs = run_mlx_pipeline(cell, stages)
    else:
        runs = run_diffusers_pipeline(cell, stages)
    print("@@result " + json.dumps(runs), flush=True)


# --------------------------------------------------------------------------------------------------------- driver


def run_cell(cell: dict, log_path: Path, sample_seconds: float = 0.2) -> dict:
    """Run one cell in a fresh process; returns its record, with the OS-level peak per run and stage."""
    load_before = round(os.getloadavg()[0], 2)
    environment = os.environ | {"PYTORCH_ENABLE_MPS_FALLBACK": "0", "TOKENIZERS_PARALLELISM": "false"}
    if cell.get("mps_high_watermark") is not None:
        environment["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(cell["mps_high_watermark"])
        environment["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(cell["mps_high_watermark"] * 0.8)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(
            [sys.executable, __file__, "--worker", json.dumps(cell)],
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
            env=environment,
        )
        state = {"run": "setup", "stage": "-", "runs": None}

        def read_stdout():
            for line in process.stdout:
                if line.startswith("@@stage "):
                    _, state["run"], state["stage"] = line.split()
                elif line.startswith("@@result "):
                    state["runs"] = json.loads(line[len("@@result ") :])

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()
        swap_base, peaks, overall, status, last_guard = swap_used_mb(), {}, 0, "ok", 0.0
        max_footprint = MAX_FOOTPRINT_FRACTION * os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        max_swap, min_disk = 0, shutil.disk_usage(ROOT).free // GIB
        started = time.time()
        while process.poll() is None:
            time.sleep(sample_seconds)
            footprint = phys_footprint(process.pid)
            if footprint:
                overall = max(overall, footprint)
                key = (state["run"], state["stage"])
                peaks[key] = max(peaks.get(key, 0), footprint)
            if footprint and footprint > max_footprint and status == "ok":
                status = f"guard: footprint {footprint / GIB:.1f} GiB in {state['run']}/{state['stage']}"
                process.kill()
            if time.time() - last_guard >= 1:
                last_guard = time.time()
                swap, disk = swap_used_mb() - swap_base, shutil.disk_usage(ROOT).free // GIB
                max_swap, min_disk = max(max_swap, swap), min(min_disk, disk)
                if (swap > SWAP_LIMIT_MB or disk < MIN_FREE_DISK_GB) and status == "ok":
                    status = f"guard: swap +{swap} MB, disk {disk} GB in {state['run']}/{state['stage']}"
                    process.kill()
        process.wait()
        reader.join(timeout=5)
    if status == "ok" and process.returncode != 0:
        status = f"exit {process.returncode}"
    runs = state["runs"] or []
    for run in runs:
        run["footprint_peak_gib"] = {
            stage: round(value / GIB, 2)
            for (name, stage), value in peaks.items()
            if name == run["run"] and stage != "-"
        }
    return {
        "cell": cell,
        "status": status,
        "runs": runs,
        "footprint_peak_gib": round(overall / GIB, 2),
        "partial_footprint_gib": {f"{r}/{s}": round(v / GIB, 2) for (r, s), v in peaks.items()} if not runs else None,
        "max_swap_delta_mb": max_swap,
        "min_free_disk_gb": min_disk,
        "wall_seconds": round(time.time() - started, 1),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "speed_limit": speed_limit(),
        "power": power_source(),
        "load_average_before": load_before,
    }


def summary_line(record: dict) -> str:
    runs = record["runs"]
    if not runs:
        return f"{record['cell']['id']}: {record['status']}"
    totals = [sum(r["seconds"].values()) for r in runs]
    warm = statistics.median(totals[1:]) if len(totals) > 1 else None
    warm_text = f"warm {warm:.1f} s" if warm is not None else "no warm runs"
    return (
        f"{record['cell']['id']}: {record['status']}, cold {totals[0]:.1f} s, {warm_text}, "
        f"footprint {record['footprint_peak_gib']} GiB, swap +{record['max_swap_delta_mb']} MB"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--matrix", nargs="+", default=[], help="groups to run, in order")
    parser.add_argument("--only", help="regular expression over cell ids (from all groups)")
    parser.add_argument("--cells-file", type=Path, help="JSON list of cells instead of the built-in matrix")
    parser.add_argument("--repeat", type=int, help="override the warm repeats")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results" / "local")
    parser.add_argument("--rounds", type=int, help="independent processes per cell (default: by group)")
    parser.add_argument("--cooldown", type=int, default=COOLDOWN_SECONDS, help="seconds of idle between cells")
    parser.add_argument("--rerun", action="store_true", help="run cells already in the results again")
    args = parser.parse_args(argv)
    if args.worker:
        worker(json.loads(args.worker))
        return 0

    groups = matrix()
    if args.list:
        for name, cells in groups.items():
            print(f"{name}: {len(cells)} cells")
            for cell in cells:
                print(f"  {cell['id']}: {cell_rounds(cell)} rounds")
        return 0
    if args.cells_file:
        cells = json.loads(args.cells_file.read_text())
    elif args.only:
        cells = [c for group in groups.values() for c in group if re.search(args.only, c["id"])]
    else:
        cells = [c for name in args.matrix for c in groups[name]]
    if not cells:
        parser.error("no cells selected")
    args.out.mkdir(parents=True, exist_ok=True)
    results = args.out / "results.jsonl"
    done = set()
    if results.is_file() and not args.rerun:
        done = {
            (record["cell"]["id"], record["round"])
            for record in map(json.loads, filter(str.strip, results.read_text().splitlines()))
            if not record["status"].startswith(
                "exit"
            )  # a crash is a bug to fix and measure again; a guard stop is a result
        }
    (args.out / "machine.json").write_text(json.dumps(machine(), indent=2) + "\n")
    rounds = max(cell_rounds(c, args.rounds) for c in cells)
    failed = 0
    for index in range(rounds):
        # Block design: every round runs each of its cells once, in a random but reproducible order.
        order = [c for c in cells if index < cell_rounds(c, args.rounds)]
        random.Random(ORDER_SEED + index).shuffle(order)
        for cell in order:
            if (cell["id"], index) in done:
                print(f"{cell['id']} round {index}: already measured", flush=True)
                continue
            current = cell | {"seed": SEEDS[index % len(SEEDS)]}
            if args.repeat is not None:
                current["repeat"] = args.repeat
            if power_source() not in (None, "AC Power"):
                print("warning: running on battery; the record says so", flush=True)
            name = f"{cell['id'].replace('/', '_')}.round{index}.log"
            record = run_cell(current, args.out / "logs" / name) | {"round": index}
            with results.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"round {index}: {summary_line(record)}", flush=True)
            failed += record["status"] != "ok"
            time.sleep(args.cooldown)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
