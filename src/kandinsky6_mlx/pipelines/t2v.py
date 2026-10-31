"""Text-to-video+audio pipeline for distilled Kandinsky 6 checkpoints, in MLX, from cached text embeddings.

Mirrors `Kandinsky6TI2VAPipeline.__call__` (diffusers 0.42.1) for text-to-video without classifier-free guidance
(`guidance_scale = 1`, the distilled checkpoints' setting): PiFlow denoising of video and audio latents with one DiT
call per step and a separate scheduler for audio, then the video VAE and the audio VAE + vocoder.

Initial noise comes from `np.random.default_rng(seed)` or is passed explicitly in the diffusers layouts (video
`(B, 16, T, H/8, W/8)`, audio `(B, 40, L)`), so a diffusers run from the same arrays is directly comparable. NumPy
noise is not the noise diffusers draws from a `torch.Generator` with the same seed.
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from kandinsky6_mlx.models.audio_vae.decoder import MMAudioVAE
from kandinsky6_mlx.models.dit.model import Kandinsky6Transformer3DModel
from kandinsky6_mlx.models.video_vae.decoder import AutoencoderKLHunyuanVideo
from kandinsky6_mlx.models.vocoder.bigvgan import MMAudioVocoder
from kandinsky6_mlx.schedulers.piflow import PiflowScheduler

PRESETS = {
    "smoke": {"height": 256, "width": 432, "num_frames": 25},
    "baseline": {"height": 480, "width": 864, "num_frames": 121},
}


@dataclass
class PipelineOutput:
    frames: np.ndarray  # (B, F, H, W, 3) float32 in [0, 1], like diffusers output_type="np"
    audio: np.ndarray | None  # (B, samples) float32 in [-1, 1]
    sample_rate: int
    frame_rate: float
    timings: dict = field(default_factory=dict)  # seconds per stage
    peak_memory_gib: dict = field(default_factory=dict)  # per stage
    step_latents: list = field(default_factory=list)  # (video, audio) after each step, if requested


def load_text_embeds(path: str | Path, trim_padding: bool = True) -> dict[str, mx.array]:
    """Cached text embeddings (`inputs.encoder_hidden_states`, `inputs.pooled_projections`, optional
    `inputs.encoder_attention_mask`).

    With `trim_padding`, right padding marked by the mask is removed and the mask dropped. Masked keys get zero weight
    and padded queries never reach other tokens, so the outputs are unchanged up to summation order (report 05), and
    the result matches what diffusers receives as `prompt_embeds`, which has no mask argument.
    """
    tensors = mx.load(str(Path(path).with_suffix(".safetensors")))
    text = {
        "encoder_hidden_states": tensors["inputs.encoder_hidden_states"],
        "pooled_projections": tensors["inputs.pooled_projections"],
    }
    mask = tensors.get("inputs.encoder_attention_mask")
    if mask is not None:
        mask_np = np.array(mask).astype(bool)
        valid = int(mask_np.sum(axis=1).max())
        if trim_padding and mask_np.shape[0] == 1 and mask_np[0, :valid].all():
            text["encoder_hidden_states"] = text["encoder_hidden_states"][:, :valid]
        elif not mask_np.all():
            text["encoder_attention_mask"] = mask.astype(mx.bool_)
    return text


class Kandinsky6Pipeline:
    def __init__(
        self,
        transformer: Kandinsky6Transformer3DModel,
        vae: AutoencoderKLHunyuanVideo,
        audio_vae: MMAudioVAE | None,
        vocoder: MMAudioVocoder | None,
        scheduler_config: dict,
    ):
        self.transformer = transformer
        self.vae = vae
        self.audio_vae = audio_vae
        self.vocoder = vocoder
        self.scheduler_config = scheduler_config
        self.vae_scale_factor_spatial = vae.spatial_compression_ratio
        self.vae_scale_factor_temporal = vae.temporal_compression_ratio
        self.patch_size = transformer.patch_size
        self.audio_sample_rate = audio_vae.sample_rate if audio_vae is not None else 44_100
        self.audio_latent_hop_length = audio_vae.latent_hop_length if audio_vae is not None else 1024

    @classmethod
    def from_converted(
        cls, path: str | Path, dtype: mx.Dtype = mx.bfloat16, audio: bool = True
    ) -> "Kandinsky6Pipeline":
        """Load a converted checkpoint directory (`transformer/`, `vae/`, `audio_vae/`, `vocoder/`, `scheduler/`).

        The DiT runs in `dtype` (time embeddings and modulation in float32); the VAEs and the vocoder in float32, as
        their released weights are F32.
        """
        path = Path(path)

        def config(component: str) -> dict:
            return json.loads((path / component / "config.json").read_text())

        transformer = Kandinsky6Transformer3DModel.from_converted(path / "transformer", dtype)
        vae = AutoencoderKLHunyuanVideo.from_config(config("vae"))
        vae.load_weights(AutoencoderKLHunyuanVideo.decoder_weights(mx.load(str(path / "vae" / "model.safetensors"))))
        audio_vae = vocoder = None
        if audio:
            audio_vae = MMAudioVAE.from_config(config("audio_vae"))
            audio_vae.load_weights(MMAudioVAE.decoder_weights(mx.load(str(path / "audio_vae" / "model.safetensors"))))
            vocoder = MMAudioVocoder.from_config(config("vocoder"))
            vocoder.load_weights(str(path / "vocoder" / "model.safetensors"))
        scheduler_file = path / "scheduler" / "scheduler_config.json"
        scheduler_config = json.loads(scheduler_file.read_text()) if scheduler_file.is_file() else {}
        return cls(transformer, vae, audio_vae, vocoder, scheduler_config)

    def latent_shapes(self, height: int, width: int, num_frames: int, frame_rate: float) -> tuple[tuple, tuple]:
        """Diffusers layouts: video `(1, 16, T, H/8, W/8)` and audio `(1, 40, L)`."""
        spatial_multiple = self.vae_scale_factor_spatial * self.patch_size[1]
        if height % spatial_multiple or width % spatial_multiple:
            raise ValueError(f"height and width must be divisible by {spatial_multiple}")
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        video = (
            1,
            self.transformer.in_visual_dim,
            num_latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        audio_length = math.ceil(
            ((num_latent_frames - 1) * self.vae_scale_factor_temporal + 1)
            / frame_rate
            * self.audio_sample_rate
            / self.audio_latent_hop_length
        )
        return video, (1, self.transformer.in_audio_dim, audio_length)

    @staticmethod
    def round_num_frames(num_frames: int, temporal: int = 4) -> int:
        """`num_frames - 1` must be divisible by 4; the reference maps others to `num_frames // 4 * 4 + 1`."""
        if num_frames % temporal != 1:
            num_frames = num_frames // temporal * temporal + 1
        return max(num_frames, 1)

    def __call__(
        self,
        text_embeds: dict[str, mx.array],
        height: int = 256,
        width: int = 432,
        num_frames: int = 25,
        frame_rate: float = 24.0,
        num_inference_steps: int = 10,
        guidance_scale: float = 1.0,
        seed: int | None = None,
        latents: np.ndarray | mx.array | None = None,
        audio_latents: np.ndarray | mx.array | None = None,
        sample_audio: bool = True,
        return_step_latents: bool = False,
        release_transformer: bool = False,
    ) -> PipelineOutput:
        """Generate one video (batch 1). `release_transformer` frees the DiT after denoising, before the decoders run
        (about 8 GB at bf16); the pipeline then cannot be called again."""
        if self.transformer is None:
            raise RuntimeError("the transformer was released by a previous call with release_transformer=True")
        if guidance_scale > 1.0:
            raise NotImplementedError("classifier-free guidance is spec 13; distilled checkpoints use guidance_scale=1")
        if sample_audio and (self.audio_vae is None or self.vocoder is None):
            raise ValueError("sample_audio=True needs the audio VAE and the vocoder")
        num_frames = self.round_num_frames(num_frames, self.vae_scale_factor_temporal)
        video_shape, audio_shape = self.latent_shapes(height, width, num_frames, frame_rate)
        rng = np.random.default_rng(seed)
        if latents is None:
            latents = rng.standard_normal(video_shape).astype(np.float32)
        if sample_audio and audio_latents is None:
            audio_latents = rng.standard_normal(audio_shape).astype(np.float32)

        dtype = self.transformer.dtype
        # Transformer layouts: video (B, T, H, W, C), audio (B, L, C), in the transformer's dtype.
        video = mx.array(latents).transpose(0, 2, 3, 4, 1).astype(dtype)
        audio = mx.array(audio_latents).transpose(0, 2, 1).astype(dtype) if sample_audio else None
        text = {k: (v.astype(dtype) if k != "encoder_attention_mask" else v) for k, v in text_embeds.items()}

        scheduler = PiflowScheduler.from_config(self.scheduler_config)
        scheduler.set_timesteps(num_inference_steps)
        audio_scheduler = PiflowScheduler.from_config(self.scheduler_config) if sample_audio else None
        if audio_scheduler is not None:
            audio_scheduler.set_timesteps(num_inference_steps)

        output = PipelineOutput(np.zeros(0), None, self.audio_sample_rate, frame_rate)
        mx.reset_peak_memory()
        start = time.perf_counter()
        for t in np.array(scheduler.timesteps):
            timestep = mx.array(np.array([t], dtype=np.float32))
            # Visual conditioning channels for text-to-video: zero conditioning latent and zero mask.
            model_input = mx.concatenate(
                [video, mx.zeros_like(video), mx.zeros((*video.shape[:-1], 1), dtype)], axis=-1
            )
            noise_pred = self.transformer(
                hidden_states=model_input,
                encoder_hidden_states=text["encoder_hidden_states"],
                pooled_projections=text["pooled_projections"],
                timestep=timestep,
                audio_hidden_states=audio,
                encoder_attention_mask=text.get("encoder_attention_mask"),
            )
            video = scheduler.step(noise_pred[0], timestep, video)
            if sample_audio:
                audio = audio_scheduler.step(noise_pred[1], timestep, audio)
            mx.eval([x for x in (video, audio) if x is not None])
            if return_step_latents:
                step_audio = None if audio is None else np.array(audio.astype(mx.float32))
                output.step_latents.append((np.array(video.astype(mx.float32)), step_audio))
        output.timings["denoise"] = time.perf_counter() - start
        output.peak_memory_gib["denoise"] = mx.get_peak_memory() / 2**30
        if release_transformer:
            self.transformer = None
            del noise_pred, model_input

        mx.clear_cache()
        mx.reset_peak_memory()
        start = time.perf_counter()
        z = video.transpose(0, 4, 1, 2, 3).astype(mx.float32) / self.vae.scaling_factor
        frames = self.vae.decode(z)
        frames = mx.clip(frames / 2 + 0.5, 0, 1).transpose(0, 2, 3, 4, 1)
        output.frames = np.array(frames)
        output.timings["video_decode"] = time.perf_counter() - start
        output.peak_memory_gib["video_decode"] = mx.get_peak_memory() / 2**30

        if sample_audio:
            mx.clear_cache()
            mx.reset_peak_memory()
            start = time.perf_counter()
            z = audio.transpose(0, 2, 1).astype(mx.float32) / self.audio_vae.scaling_factor
            waveform = self.vocoder(self.audio_vae.decode(z))[:, 0]
            output.audio = np.array(waveform)
            output.timings["audio_decode"] = time.perf_counter() - start
            output.peak_memory_gib["audio_decode"] = mx.get_peak_memory() / 2**30
        return output
