"""Generate videos with sound from the study notebook's prompt presets with the MLX pipeline.

Prompts are compiled like the notebook's `compile_prompt` (`applehawk/kandinsky-6-lite-diffusers-generation`): base
style, room description, lighting, the profile's camera setup, the initial state, the shot length, the timeline actions
and the audio cues inside `<AUDCAP>...<ENDAUDCAP>`. Negative prompts are not used: the distilled model runs without
classifier-free guidance. Only text-to-video presets run here; image-to-video needs spec 12.

All prompts are encoded first (Qwen2.5-VL in MLX, CLIP in torch), then the encoder is freed and the pipeline generates
each video with NumPy noise from `--seed`.

    uv run --extra text --group reference python scripts/generate_presets.py --presets presets.json \\
        --qwen-weights ~/ComfyUI-Shared/models/text_encoders/qwen_2.5_vl_7b.safetensors \\
        --text-root weights/lite-distill-text --weights weights/lite-distill --profile smoke --out outputs/presets
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from kandinsky6_mlx.pipelines.export import write_mp4
from kandinsky6_mlx.pipelines.t2v import PRESETS as SIZES
from kandinsky6_mlx.pipelines.t2v import Kandinsky6Pipeline

SECONDS = {"smoke": 1, "baseline": 5}


def compile_prompt(preset: dict, profile: str) -> str:
    """The study notebook's prompt layout (checked against fixtures/real_inputs/smoke_prompt.txt)."""
    scene, chosen = preset["scene_setup"], preset["profiles"][profile]
    parts = [
        preset["metadata"]["base_style"],
        scene["room_description"],
        scene["lighting"],
        chosen.get("camera_setup", scene["camera_setup"]),
        "Initial state: " + " ".join(preset["elements"]["initial_state"]),
        f"One uninterrupted shot, about {SECONDS[profile]} seconds.",
        *[f"From {step['start']} to {step['end']} seconds: {step['action']}" for step in chosen["timeline"]],
    ]
    audio = " ".join(step["audio_cue"] for step in chosen["timeline"])
    return " ".join(parts) + f" <AUDCAP>{audio}<ENDAUDCAP>"


def encode_all(prompts: dict[str, str], qwen_weights: list[Path], text_root: Path) -> dict[str, dict]:
    from encode_prompt import clip_pooled  # scripts/ is on sys.path when run as a script
    from transformers import AutoTokenizer

    from kandinsky6_mlx.text.encode import encode_qwen
    from kandinsky6_mlx.text.qwen import QwenTextEncoder

    encoder = QwenTextEncoder.from_files(text_root / "text_encoder" / "config.json", qwen_weights)
    tokenizer = AutoTokenizer.from_pretrained(text_root / "tokenizer")
    embeds = {name: encode_qwen(encoder, tokenizer, prompt) for name, prompt in prompts.items()}
    mx.eval(list(embeds.values()))
    del encoder
    mx.clear_cache()
    return {
        name: {"encoder_hidden_states": embeds[name], "pooled_projections": mx.array(clip_pooled(text_root, prompt))}
        for name, prompt in prompts.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--presets", type=Path, required=True, help="JSON of the notebook's PRESETS dict")
    parser.add_argument("--names", nargs="*", help="presets to run (default: every t2v preset)")
    parser.add_argument("--profile", choices=sorted(SECONDS), default="smoke")
    parser.add_argument("--qwen-weights", type=Path, nargs="+", required=True)
    parser.add_argument("--text-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True, help="converted checkpoint directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    presets = json.loads(args.presets.read_text())
    names = args.names or [n for n, p in presets.items() if p["metadata"]["mode"] == "t2v"]
    skipped = [n for n in names if presets[n]["metadata"]["mode"] != "t2v"]
    if skipped:
        raise SystemExit(f"image-to-video presets need spec 12: {skipped}")
    prompts = {name: compile_prompt(presets[name], args.profile) for name in names}
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    texts = encode_all(prompts, args.qwen_weights, args.text_root)
    encode_seconds = time.perf_counter() - started
    pipeline = Kandinsky6Pipeline.from_converted(args.weights)
    summary = {"profile": args.profile, "seed": args.seed, "encode_seconds": round(encode_seconds, 1), "videos": {}}
    for name in names:
        output = pipeline(texts[name], **SIZES[args.profile], num_inference_steps=args.steps, seed=args.seed)
        path = write_mp4(
            args.out / f"{name}_{args.profile}_seed{args.seed}.mp4",
            output.frames[0],
            output.frame_rate,
            output.audio[0],
            output.sample_rate,
        )
        summary["videos"][name] = {
            "mp4": str(path),
            "text_tokens": int(texts[name]["encoder_hidden_states"].shape[1]),
            "seconds": {k: round(v, 1) for k, v in output.timings.items()},
            "peak_memory_gib": {k: round(v, 1) for k, v in output.peak_memory_gib.items()},
            "prompt": prompts[name],
        }
        mx.clear_cache()
        print(json.dumps({name: {k: v for k, v in summary["videos"][name].items() if k != "prompt"}}), flush=True)
    (args.out / f"summary_{args.profile}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
