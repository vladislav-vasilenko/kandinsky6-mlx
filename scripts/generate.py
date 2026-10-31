"""Generate a video with sound from a prompt (or cached text embeddings) with the MLX Kandinsky 6 pipeline.

    uv run --extra text python scripts/generate.py --weights weights/lite-distill \\
        --prompt "A red fox runs through fresh snow ... <AUDCAP>Crunching snow.<ENDAUDCAP>" --preset smoke --out fox.mp4

    uv run python scripts/generate.py --weights weights/lite-distill \\
        --text-embeds fixtures/real/smoke_text.safetensors --preset smoke --seed 42 --out outputs/09/smoke.mp4

With `--prompt` the text encoders (`--text-root`, extra `text`) run first and are freed before the DiT is loaded.

The seed drives NumPy noise, so it does not reproduce a diffusers or ComfyUI run with the same seed; pass
`--latents` (a fixture with `inputs.latents` and `inputs.audio_latents` in the diffusers layouts) for that.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from kandinsky6_mlx.models.video_vae.decoder import TilingConfig
from kandinsky6_mlx.pipelines.export import write_mp4
from kandinsky6_mlx.pipelines.t2v import DEFAULT_NEGATIVE_PROMPT, PRESETS, Kandinsky6Pipeline, load_text_embeds

try:  # scripts/ is on sys.path when run as a script
    from encode_prompt import add_text_arguments, load_encoders
except ImportError:
    from scripts.encode_prompt import add_text_arguments, load_encoders

DTYPES = {"bfloat16": mx.bfloat16, "float32": mx.float32}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True, help="converted checkpoint directory")
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--prompt")
    text.add_argument("--prompt-file", type=Path)
    text.add_argument("--text-embeds", type=Path, help="cached embeddings (scripts/encode_prompt.py)")
    add_text_arguments(parser)
    parser.add_argument("--image", type=Path, help="reference image for image-to-video (first frame)")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--steps", type=int, help="default: 10 for distilled checkpoints, 50 otherwise")
    parser.add_argument("--guidance", type=float, help="guidance_scale; default 1 (distilled) or 5")
    parser.add_argument("--negative-prompt", help="default: diffusers' default negative prompt (with guidance)")
    parser.add_argument("--negative-text-embeds", type=Path, help="cached negative embeddings (with --text-embeds)")
    parser.add_argument("--cfg-batched", action="store_true", help="one batch-2 DiT call per step (more memory)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latents", type=Path, help="fixture with inputs.latents / inputs.audio_latents")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16", help="DiT compute dtype")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--vae-tiling", choices=("preset", "on", "off"), default="preset", help="spatial VAE tiles")
    parser.add_argument(
        "--save-latents", type=Path, help="also save the final video latents as inputs.latents (B, T, H, W, 16)"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    size = PRESETS[args.preset] | {
        k: v for k, v in (("height", args.height), ("width", args.width), ("num_frames", args.num_frames)) if v
    }
    if args.vae_tiling != "preset":
        size["vae_tiling"] = TilingConfig() if args.vae_tiling == "on" else None
    seconds = {}
    distilled = (
        json.loads((args.weights / "scheduler" / "scheduler_config.json").read_text()).get("_class_name")
        in (None, "PiflowScheduler")
        if (args.weights / "scheduler" / "scheduler_config.json").is_file()
        else True
    )
    guidance = args.guidance if args.guidance is not None else (1.0 if distilled else 5.0)
    negative_embeds = None
    if args.text_embeds is not None:
        text_embeds = load_text_embeds(args.text_embeds)
        if guidance > 1.0:
            if args.negative_text_embeds is None:
                parser.error("guidance > 1 with --text-embeds needs --negative-text-embeds")
            negative_embeds = load_text_embeds(args.negative_text_embeds)
    else:
        prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text()
        start = time.perf_counter()
        encoders = load_encoders(args)
        text_embeds = encoders.encode_prompt(prompt)
        if guidance > 1.0:
            negative_embeds = encoders.encode_prompt(args.negative_prompt or DEFAULT_NEGATIVE_PROMPT)
        del encoders  # freed before the DiT is loaded
        mx.clear_cache()
        seconds["text"] = round(time.perf_counter() - start, 2)
    start = time.perf_counter()
    pipeline = Kandinsky6Pipeline.from_converted(args.weights, DTYPES[args.dtype], audio=not args.no_audio)
    load_seconds = time.perf_counter() - start
    noise = {}
    if args.latents is not None:
        tensors = mx.load(str(args.latents.with_suffix(".safetensors")))
        noise = {"latents": tensors["inputs.latents"], "audio_latents": tensors.get("inputs.audio_latents")}

    def save_latents(latents) -> None:
        """Written right after denoising, so the latents survive a failed decode."""
        args.save_latents.parent.mkdir(parents=True, exist_ok=True)
        tensors = {"inputs.latents": mx.array(latents).transpose(0, 2, 3, 4, 1)}
        mx.save_safetensors(str(args.save_latents.with_suffix(".safetensors")), tensors)
        meta = {
            "fixture": args.save_latents.name,
            "layout": "(B, T, H/8, W/8, 16) like the transformer, not divided by scaling_factor",
        }
        args.save_latents.with_suffix(".json").write_text(json.dumps(meta | {"seed": args.seed}, indent=2) + "\n")
        print(f"saved latents -> {args.save_latents}", flush=True)

    output = pipeline(
        text_embeds,
        **size,
        num_inference_steps=args.steps,
        guidance_scale=guidance,
        negative_text_embeds=negative_embeds,
        cfg_batched=args.cfg_batched,
        seed=args.seed,
        sample_audio=not args.no_audio,
        release_transformer=True,
        progress=True,
        on_latents=save_latents if args.save_latents is not None else None,
        image=args.image,
        **noise,
    )
    audio = None if output.audio is None else output.audio[0]
    write_mp4(args.out, output.frames[0], output.frame_rate, audio, output.sample_rate)

    report = {
        "out": str(args.out),
        "size": {k: v for k, v in size.items() if k != "vae_tiling"},
        "vae_tiling": size.get("vae_tiling") is not None,
        "steps": args.steps or (10 if distilled else 50),
        "guidance_scale": guidance,
        "seed": args.seed,
        "image": str(args.image) if args.image else None,
        "dtype": args.dtype,
        "seconds": seconds | {"load": round(load_seconds, 2)} | {k: round(v, 2) for k, v in output.timings.items()},
        "peak_memory_gib": {k: round(v, 2) for k, v in output.peak_memory_gib.items()},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
