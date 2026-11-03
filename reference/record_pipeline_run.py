"""Record a diffusers Kandinsky 6 text-to-video+audio run for spec 09 stage B.

Runs `Kandinsky6TI2VAPipeline` (diffusers 0.42.1) from cached text embeddings and NumPy noise and saves, in one
fixture: the initial video and audio latents (diffusers layouts `(1, 16, T, H/8, W/8)` and `(1, 40, L)`), the text
embeddings it used, the latents after every step, the final frames (`output_type="np"`) and the waveform.

Text embeddings: a fixture with `inputs.encoder_hidden_states`, `inputs.pooled_projections` and optionally
`inputs.encoder_attention_mask` (from `record_dit_real.py --save-text-embeds`). The pipeline's `prompt_embeds`
argument takes no mask, so right padding is removed first, as the study notebook does; the MLX pipeline's
`load_text_embeds` does the same, and report 05 shows the result is unchanged by it.

The transformer runs in `--dtype` (float32 for the reference, bfloat16 for the drift basis) and comes from a diffusers
snapshot (`--checkpoint`). The other components come from the same snapshot, or with `--converted` from the converted
MLX directories with the convolutions moved back to the PyTorch layout (as in record_audio_real.py and
record_video_real.py; the converter only permutes axes); the scheduler config then comes from `--scheduler-config`.

`--output-type latent` stops before decoding and also writes `<out>_final` (`inputs.latents` `(B, T, H, W, 16)` and
`inputs.audio_latents` `(B, L, 40)` after the last step), which record_video_real.py and record_audio_real.py decode
separately with `--latents`. That keeps the float32 DiT and the float32 VAE decode (about 25 GB on a CPU) out of
memory at the same time.

    python reference/record_pipeline_run.py --checkpoint weights/lite-distill-diffusers --converted weights/lite-distill \\
        --scheduler-config fixtures/configs/lite-distill-scheduler.json --text-embeds fixtures/real/smoke_text \\
        --dtype float32 --output-type latent --out fixtures/real/pipeline_smoke_fp32
"""

import argparse
import json
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


def scheduler_class(name: str):
    import diffusers

    if name not in ("PiflowScheduler", "FlowMatchEulerDiscreteScheduler"):
        raise NotImplementedError(name)
    return getattr(diffusers, name)


def build_pipeline(
    checkpoint: Path,
    dtype: torch.dtype,
    device: str,
    converted: Path | None = None,
    scheduler_config: Path | None = None,
    with_vae: bool = True,
    media: dict | None = None,
):
    from diffusers import (
        AutoencoderKLHunyuanVideo,
        Kandinsky6TI2VAPipeline,
        Kandinsky6Transformer3DModel,
    )
    from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE
    from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder

    from reference.record_audio_real import load_reference as load_converted_audio
    from reference.record_video_real import load_reference as load_converted_vae

    root = find_root(checkpoint, "transformer")
    if converted is not None:
        vae = audio_vae = vocoder = None
        if media is None:
            vae = load_converted_vae(converted / "vae") if with_vae else None
            audio_vae = load_converted_audio(converted / "audio_vae", "audio_vae", MMAudioVAE)
            vocoder = load_converted_audio(converted / "vocoder", "vocoder", MMAudioVocoder)
        config = json.loads(scheduler_config.read_text())
        scheduler = scheduler_class(config.get("_class_name", "PiflowScheduler")).from_config(config)
    else:
        vae = AutoencoderKLHunyuanVideo.from_pretrained(root / "vae", torch_dtype=torch.float32) if with_vae else None
        audio_vae = MMAudioVAE.from_pretrained(root / "audio_vae", torch_dtype=torch.float32)
        vocoder = MMAudioVocoder.from_pretrained(root / "vocoder", torch_dtype=torch.float32)
        config = json.loads((root / "scheduler" / "scheduler_config.json").read_text())
        scheduler = scheduler_class(config.get("_class_name", "PiflowScheduler")).from_pretrained(root / "scheduler")
    if media is not None:  # the caller's media components, e.g. weightless ones to be replaced after denoising
        return Kandinsky6TI2VAPipeline(
            transformer=Kandinsky6Transformer3DModel.from_pretrained(root / "transformer", torch_dtype=dtype).to(
                device
            ),
            scheduler=scheduler,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            **media,
        )
    return Kandinsky6TI2VAPipeline(
        transformer=Kandinsky6Transformer3DModel.from_pretrained(root / "transformer", torch_dtype=dtype),
        vae=vae,
        audio_vae=audio_vae,
        vocoder=vocoder,
        scheduler=scheduler,
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
    parser.add_argument("--guidance", type=float, default=1.0, help="guidance_scale; > 1 needs --negative-text-embeds")
    parser.add_argument("--negative-text-embeds", type=Path, help="embeddings of the negative prompt (spec 13)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--converted", type=Path, help="converted root for vae/audio_vae/vocoder")
    parser.add_argument("--scheduler-config", type=Path, help="scheduler_config.json (with --converted)")
    parser.add_argument("--output-type", choices=("np", "latent"), default="np")
    parser.add_argument("--image", type=Path, help="reference image for image-to-video (spec 12)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.converted is not None and args.scheduler_config is None:
        parser.error("--converted needs --scheduler-config")

    height, width, num_frames = PRESETS[args.preset]
    latent_only = args.output_type == "latent"
    # The VAE is needed to encode the reference image even when the run stops at the latents.
    pipe = build_pipeline(
        args.checkpoint,
        DTYPES[args.dtype],
        args.device,
        args.converted,
        args.scheduler_config,
        not latent_only or args.image is not None,
    )
    embeds, pooled = load_text(args.text_embeds)
    negative = negative_pooled = None
    if args.guidance > 1.0:
        if args.negative_text_embeds is None:
            parser.error("--guidance > 1 needs --negative-text-embeds")
        negative, negative_pooled = load_text(args.negative_text_embeds)
    num_latent_frames = (num_frames - 1) // 4 + 1
    audio_length = math.ceil(
        ((num_latent_frames - 1) * 4 + 1) / 24.0 * pipe.audio_sample_rate / pipe.audio_latent_hop_length
    )
    rng = np.random.default_rng(args.seed)
    latents = rng.standard_normal((1, 16, num_latent_frames, height // 8, width // 8)).astype(np.float32)
    audio_latents = rng.standard_normal((1, 40, audio_length)).astype(np.float32)
    image = image_noise = None
    if args.image is not None:
        from diffusers.models.autoencoders import vae as vae_module
        from PIL import Image

        # The pipeline samples the image latent distribution with a torch generator; use NumPy noise drawn after the
        # video and audio noise, as the MLX pipeline does, so both start from the same sample.
        image = Image.open(args.image)
        image_noise = rng.standard_normal((1, 16, 1, height // 8, width // 8)).astype(np.float32)
        noise = torch.from_numpy(image_noise)
        original_sample = vae_module.DiagonalGaussianDistribution.sample
        vae_module.DiagonalGaussianDistribution.sample = lambda self, generator=None: (
            self.mean + self.std * noise.to(self.mean.device, self.mean.dtype)
        )

    steps: list[tuple[torch.Tensor, torch.Tensor]] = []

    def record(_pipe, _i, _t, tensors):
        steps.append((tensors["latents"].float().cpu().clone(), tensors["audio_latents"].float().cpu().clone()))
        return {}

    started = time.time()
    try:
        with torch.no_grad():
            video, audio = pipe(
                image=image,
                prompt_embeds=embeds.to(args.device),
                pooled_prompt_embeds=pooled.to(args.device),
                latents=torch.from_numpy(latents).to(args.device),
                audio_latents=torch.from_numpy(audio_latents).to(args.device),
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                negative_prompt_embeds=None if negative is None else negative.to(args.device),
                negative_pooled_prompt_embeds=None if negative_pooled is None else negative_pooled.to(args.device),
                output_type=args.output_type,
                return_dict=False,
                callback_on_step_end=record,
                callback_on_step_end_tensor_inputs=["latents", "audio_latents"],
            )
    finally:
        if image is not None:
            vae_module.DiagonalGaussianDistribution.sample = original_sample
    seconds = time.time() - started
    tensors = {
        "inputs.latents": latents,
        "inputs.audio_latents": audio_latents,
        "inputs.encoder_hidden_states": embeds,
        "inputs.pooled_projections": pooled,
    }
    if image_noise is not None:
        tensors["inputs.image_noise"] = image_noise
    if negative is not None:
        tensors["inputs.negative_encoder_hidden_states"] = negative
        tensors["inputs.negative_pooled_projections"] = negative_pooled
    if not latent_only:
        tensors["outputs.frames"] = np.asarray(video, np.float32)
        tensors["outputs.audio"] = np.asarray(audio, np.float32)
    for i, (v, a) in enumerate(steps):
        tensors[f"outputs.step.{i}.latents"] = v
        tensors[f"outputs.step.{i}.audio_latents"] = a
    save_fixture(
        args.out,
        tensors,
        {
            "fixture": args.out.name,
            "reference": f"diffusers.Kandinsky6TI2VAPipeline.__call__ (guidance_scale={args.guidance}, output_type={args.output_type})",
            "checkpoint": str(args.checkpoint),
            "converted": str(args.converted) if args.converted else None,
            "image": str(args.image) if args.image else None,
            "transformer_dtype": args.dtype,
            "device": args.device,
            "preset": args.preset,
            "steps": args.steps,
            "guidance_scale": args.guidance,
            "scheduler": type(pipe.scheduler).__name__,
            "seed": f"np.random.default_rng({args.seed}): video then audio latents",
            "seconds": round(seconds, 1),
            "versions": library_versions() | {"torch": torch.__version__},
        },
    )
    if latent_only:
        final = args.out.with_name(args.out.name + "_final")
        save_fixture(
            final,
            # Without the image-conditioning tail frame, which the pipeline drops before decoding.
            {"inputs.latents": steps[-1][0][:, : latents.shape[2]], "inputs.audio_latents": steps[-1][1]},
            {"fixture": final.name, "source": f"last step of {args.out.name}"},
        )
    print(f"{len(steps)} steps in {seconds:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
