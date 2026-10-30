"""Record the float32 reference video decode (latents -> frames) on real weights for spec 08 stage B.

The reference `AutoencoderKLHunyuanVideo.decode` (diffusers 0.42.1, CPU, float32, default framewise decoding, no
spatial tiling) is loaded from the converted `<weights>/vae` with the convolution weights moved back to the PyTorch
layout (the converter only permutes axes; spec 02 checks values bit-exact).

Latents follow the pipeline: transformer-layout latents `(B, T, H, W, 16)` are permuted to `(B, 16, T, H, W)` and
divided by `scaling_factor`. With `--latents` (a fixture with `inputs.latents` from a Smoke run) those are used;
otherwise `np.random.default_rng(seed)` normal latents of the Smoke shape stand in until spec 09 records real ones.

Usage:
    uv run --group reference python reference/record_video_real.py --weights weights/lite-distill \\
        --out fixtures/real/video_smoke
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kandinsky6_mlx.convert.layouts import conv_rule
from kandinsky6_mlx.utils.fixtures import save_fixture

SMOKE_LATENTS = (1, 7, 32, 54, 16)  # (B, T, H, W, C): 25 frames at 256x432


def load_reference(directory: Path) -> torch.nn.Module:
    from diffusers import AutoencoderKLHunyuanVideo
    from safetensors.torch import load_file

    model = AutoencoderKLHunyuanVideo.from_config(json.loads((directory / "config.json").read_text())).float().eval()
    state = {}
    for name, value in load_file(str(directory / "model.safetensors")).items():
        rule = conv_rule("vae", name)
        state[name] = value.permute(*np.argsort(rule.axes())) if rule is not None else value
    model.load_state_dict(state, strict=True)
    return model


def main(argv: list[str] | None = None) -> int:
    from reference.recorder import library_versions

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, default=Path("weights/lite-distill"))
    parser.add_argument("--latents", type=Path, help="fixture with inputs.latents (B, T, H, W, C) from a Smoke run")
    parser.add_argument("--shape", type=int, nargs=5, default=SMOKE_LATENTS, help="(B, T, H, W, C) without --latents")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("fixtures/real/video_smoke"))
    args = parser.parse_args(argv)

    vae = load_reference(args.weights / "vae")
    if args.latents is not None:
        from safetensors.torch import load_file

        latents = load_file(str(args.latents.with_suffix(".safetensors")))["inputs.latents"].float()
        source = str(args.latents)
    else:
        latents = torch.from_numpy(np.random.default_rng(args.seed).standard_normal(args.shape).astype(np.float32))
        source = f"np.random.default_rng({args.seed}).standard_normal"
    started = time.time()
    with torch.no_grad():
        frames = vae.decode(latents.permute(0, 4, 1, 2, 3) / vae.config.scaling_factor, return_dict=False)[0]
    seconds = time.time() - started
    save_fixture(
        args.out,
        {"inputs.latents": latents, "outputs.frames": frames},
        {
            "fixture": args.out.name,
            "reference": "diffusers.AutoencoderKLHunyuanVideo.decode (framewise decoding, no spatial tiling)",
            "weights": str(args.weights),
            "latents": source,
            "scaling_factor": vae.config.scaling_factor,
            "torch_cpu_seconds": round(seconds, 1),
            "versions": library_versions(),
        },
    )
    print(f"frames {tuple(frames.shape)} in {seconds:.1f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
