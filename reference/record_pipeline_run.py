"""Record a diffusers Kandinsky 6 text-to-video+audio run for spec 09 stage B.

Runs `Kandinsky6TI2VAPipeline` (diffusers 0.42.1) from cached text embeddings and NumPy noise and saves, in one
fixture: the initial video and audio latents (diffusers layouts `(1, 16, T, H/8, W/8)` and `(1, 40, L)`), the text
embeddings it used, the latents after every step, the final frames (`output_type="np"`) and the waveform.

Text embeddings: a fixture with `inputs.encoder_hidden_states`, `inputs.pooled_projections` and optionally
`inputs.encoder_attention_mask` (from `record_dit_real.py --save-text-embeds`). The pipeline's `prompt_embeds`
argument takes no mask, so right padding is removed first, as the study notebook does; the MLX pipeline's
`load_text_embeds` does the same, and report 05 shows the result is unchanged by it.

The transformer runs in `--dtype` (float32 for the reference, bfloat16 for the drift basis); the VAEs and the vocoder
stay float32 like the released weights. The media components come from a diffusers snapshot (`--checkpoint`).

    python reference/record_pipeline_run.py --checkpoint <snapshot> --text-embeds smoke_text \\
        --dtype float32 --out pipeline_smoke_fp32
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kandinsky6_mlx.utils.fixtures import save_fixture

PRESETS = {"smoke": (256, 432, 25), "baseline": (480, 864, 121)}
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def find_root(root: Path, component: str) -> Path:
    if (root / component / "config.json").is_file():
        return root
    return next(root.rglob(f"{component}/config.json")).parent.parent


def load_text(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    from safetensors.torch import load_file

    tensors = load_file(str(path.with_suffix(".safetensors")))
    embeds, pooled = tensors["inputs.encoder_hidden_states"].float(), tensors["inputs.pooled_projections"].float()
    mask = tensors.get("inputs.encoder_attention_mask")
    if mask is not None:
        valid = int(mask.bool().sum(dim=1).max())
        if not bool(mask.bool()[0, :valid].all()):
            raise ValueError("expected right padding")
        embeds = embeds[:, :valid]
    return embeds, pooled


def build_pipeline(checkpoint: Path, dtype: torch.dtype, device: str):
    from diffusers import (
        AutoencoderKLHunyuanVideo,
        Kandinsky6TI2VAPipeline,
        Kandinsky6Transformer3DModel,
        PiflowScheduler,
    )
    from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE
    from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder

    root = find_root(checkpoint, "transformer")
    return Kandinsky6TI2VAPipeline(
        transformer=Kandinsky6Transformer3DModel.from_pretrained(root / "transformer", torch_dtype=dtype),
        vae=AutoencoderKLHunyuanVideo.from_pretrained(root / "vae", torch_dtype=torch.float32),
        audio_vae=MMAudioVAE.from_pretrained(root / "audio_vae", torch_dtype=torch.float32),
        vocoder=MMAudioVocoder.from_pretrained(root / "vocoder", torch_dtype=torch.float32),
        scheduler=PiflowScheduler.from_pretrained(root / "scheduler"),
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
    ).to(device)


def main(argv: list[str] | None = None) -> int:
    from reference.recorder import library_versions

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="diffusers snapshot root")
    parser.add_argument("--text-embeds", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    height, width, num_frames = PRESETS[args.preset]
    pipe = build_pipeline(args.checkpoint, DTYPES[args.dtype], args.device)
    embeds, pooled = load_text(args.text_embeds)
    num_latent_frames = (num_frames - 1) // 4 + 1
    audio_length = math.ceil(
        ((num_latent_frames - 1) * 4 + 1) / 24.0 * pipe.audio_sample_rate / pipe.audio_latent_hop_length
    )
    rng = np.random.default_rng(args.seed)
    latents = rng.standard_normal((1, 16, num_latent_frames, height // 8, width // 8)).astype(np.float32)
    audio_latents = rng.standard_normal((1, 40, audio_length)).astype(np.float32)

    steps: list[tuple[torch.Tensor, torch.Tensor]] = []

    def record(_pipe, _i, _t, tensors):
        steps.append((tensors["latents"].float().cpu().clone(), tensors["audio_latents"].float().cpu().clone()))
        return {}

    started = time.time()
    with torch.no_grad():
        video, audio = pipe(
            prompt_embeds=embeds.to(args.device),
            pooled_prompt_embeds=pooled.to(args.device),
            latents=torch.from_numpy(latents).to(args.device),
            audio_latents=torch.from_numpy(audio_latents).to(args.device),
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
            output_type="np",
            return_dict=False,
            callback_on_step_end=record,
            callback_on_step_end_tensor_inputs=["latents", "audio_latents"],
        )
    seconds = time.time() - started
    tensors = {
        "inputs.latents": latents,
        "inputs.audio_latents": audio_latents,
        "inputs.encoder_hidden_states": embeds,
        "inputs.pooled_projections": pooled,
        "outputs.frames": np.asarray(video, np.float32),
        "outputs.audio": np.asarray(audio, np.float32),
    }
    for i, (v, a) in enumerate(steps):
        tensors[f"outputs.step.{i}.latents"] = v
        tensors[f"outputs.step.{i}.audio_latents"] = a
    save_fixture(
        args.out,
        tensors,
        {
            "fixture": args.out.name,
            "reference": "diffusers.Kandinsky6TI2VAPipeline.__call__ (T2V, guidance_scale=1, output_type=np)",
            "checkpoint": str(args.checkpoint),
            "transformer_dtype": args.dtype,
            "device": args.device,
            "preset": args.preset,
            "steps": args.steps,
            "seed": f"np.random.default_rng({args.seed}): video then audio latents",
            "seconds": round(seconds, 1),
            "versions": library_versions() | {"torch": torch.__version__},
        },
    )
    print(f"{len(steps)} steps in {seconds:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
