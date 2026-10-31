"""Generate a video with sound from cached text embeddings with the MLX Kandinsky 6 pipeline.

    uv run python scripts/generate.py --weights weights/lite-distill \\
        --text-embeds fixtures/real/smoke_text.safetensors --preset smoke --seed 42 --out outputs/09/smoke.mp4

The seed drives NumPy noise, so it does not reproduce a diffusers or ComfyUI run with the same seed; pass
`--latents` (a fixture with `inputs.latents` and `inputs.audio_latents` in the diffusers layouts) for that.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from kandinsky6_mlx.pipelines.export import write_mp4
from kandinsky6_mlx.pipelines.t2v import PRESETS, Kandinsky6Pipeline, load_text_embeds

DTYPES = {"bfloat16": mx.bfloat16, "float32": mx.float32}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True, help="converted checkpoint directory")
    parser.add_argument("--text-embeds", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latents", type=Path, help="fixture with inputs.latents / inputs.audio_latents")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16", help="DiT compute dtype")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    size = PRESETS[args.preset] | {
        k: v for k, v in (("height", args.height), ("width", args.width), ("num_frames", args.num_frames)) if v
    }
    start = time.perf_counter()
    pipeline = Kandinsky6Pipeline.from_converted(args.weights, DTYPES[args.dtype], audio=not args.no_audio)
    load_seconds = time.perf_counter() - start
    noise = {}
    if args.latents is not None:
        tensors = mx.load(str(args.latents.with_suffix(".safetensors")))
        noise = {"latents": tensors["inputs.latents"], "audio_latents": tensors.get("inputs.audio_latents")}
    output = pipeline(
        load_text_embeds(args.text_embeds),
        **size,
        num_inference_steps=args.steps,
        seed=args.seed,
        sample_audio=not args.no_audio,
        release_transformer=True,
        **noise,
    )
    audio = None if output.audio is None else output.audio[0]
    write_mp4(args.out, output.frames[0], output.frame_rate, audio, output.sample_rate)
    report = {
        "out": str(args.out),
        "size": size,
        "steps": args.steps,
        "seed": args.seed,
        "dtype": args.dtype,
        "seconds": {"load": round(load_seconds, 2)} | {k: round(v, 2) for k, v in output.timings.items()},
        "peak_memory_gib": {k: round(v, 2) for k, v in output.peak_memory_gib.items()},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
