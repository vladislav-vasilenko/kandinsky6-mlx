"""Text/image-to-video+audio pipeline for Kandinsky 6 checkpoints, in MLX, from text embeddings.

Mirrors `Kandinsky6TI2VAPipeline.__call__` (diffusers 0.42.1): denoising of video and audio latents with the scheduler
named in `scheduler/scheduler_config.json` (PiFlow for the distilled checkpoints, FlowMatch Euler for the others) and
a separate scheduler instance for audio, classifier-free guidance when `guidance_scale > 1`, optional first-frame image
conditioning, then the video VAE and the audio VAE + vocoder.

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
from kandinsky6_mlx.models.video_vae.decoder import AutoencoderKLHunyuanVideo, TilingConfig
from kandinsky6_mlx.models.vocoder.bigvgan import MMAudioVocoder
from kandinsky6_mlx.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from kandinsky6_mlx.schedulers.piflow import PiflowScheduler

SCHEDULERS = {"PiflowScheduler": PiflowScheduler, "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler}
# diffusers' Kandinsky6TI2VAPipeline._DEFAULT_NEGATIVE_PROMPT (a test compares).
DEFAULT_NEGATIVE_PROMPT = (
    "Static, 2D cartoon, cartoon, 2d animation, paintings, images, "
    "worst quality, low quality, ugly, deformed, walking backwards"
)


def make_scheduler(config: dict):
    """The scheduler class named by `_class_name` (PiFlow when the config has none, as in older converted dirs)."""
    name = config.get("_class_name", "PiflowScheduler")
    if name not in SCHEDULERS:
        raise NotImplementedError(f"scheduler {name!r}")
    return SCHEDULERS[name].from_config(config)


def pad_text(a: dict, b: dict) -> dict:
    """Two text caches as one batch of 2: token states right-padded to a common length with a padding mask."""
    la, lb = a["encoder_hidden_states"].shape[1], b["encoder_hidden_states"].shape[1]
    length = max(la, lb)

    def pad(text, n):
        states = text["encoder_hidden_states"]
        if n < length:
            states = mx.concatenate([states, mx.zeros((1, length - n, states.shape[-1]), states.dtype)], axis=1)
        mask = text.get("encoder_attention_mask")
        mask = mx.ones((1, n), dtype=mx.bool_) if mask is None else mask.astype(mx.bool_)
        mask = mx.concatenate([mask, mx.zeros((1, length - n), dtype=mx.bool_)], axis=1)
        return states, mask

    (sa, ma), (sb, mb) = pad(a, la), pad(b, lb)
    return {
        "encoder_hidden_states": mx.concatenate([sa, sb], axis=0),
        "pooled_projections": mx.concatenate([a["pooled_projections"], b["pooled_projections"]], axis=0),
        "encoder_attention_mask": mx.concatenate([ma, mb], axis=0),
    }


PRESETS = {
    "smoke": {"height": 256, "width": 432, "num_frames": 25},
    # The study notebook decodes Baseline with `vae.enable_tiling()` at the reference defaults.
    "baseline": {"height": 480, "width": 864, "num_frames": 121, "vae_tiling": TilingConfig()},
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
    latents: np.ndarray | None = None  # final video latents, diffusers layout (B, 16, T, H/8, W/8), float32
    step_seconds: list = field(default_factory=list)  # wall time of each denoising step


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
        self.vae_path = None  # converted vae/ directory: the encoder is loaded from it only for image-to-video

    @classmethod
    def from_converted(
        cls, path: str | Path, dtype: mx.Dtype = mx.bfloat16, audio: bool = True, fp32_rule: str = "substring"
    ) -> "Kandinsky6Pipeline":
        """Load a converted checkpoint directory (`transformer/`, `vae/`, `audio_vae/`, `vocoder/`, `scheduler/`).

        The DiT runs in `dtype` (time embeddings and modulation in float32); the VAEs and the vocoder in float32, as
        their released weights are F32.
        """
        path = Path(path)

        def config(component: str) -> dict:
            return json.loads((path / component / "config.json").read_text())

        transformer = Kandinsky6Transformer3DModel.from_converted(path / "transformer", dtype, fp32_rule)
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
        pipeline = cls(transformer, vae, audio_vae, vocoder, scheduler_config)
        pipeline.vae_path = path / "vae"
        return pipeline

    def encode_image(self, image, height: int, width: int, noise: np.ndarray) -> mx.array:
        """`Kandinsky6TI2VAPipeline.encode_image`: preprocess (resize and centre-crop to `height x width`, [-1, 1]),
        encode one frame, sample the latent distribution with the given standard-normal `noise` `(1, 16, 1, H/8, W/8)`
        and scale by `scaling_factor`. Returns the first-frame latents `(1, H/8, W/8, 16)` in float32. The encoder is
        loaded from the converted `vae/` and freed afterwards."""
        from PIL import Image

        from kandinsky6_mlx.models.video_vae.encoder import HunyuanVideoVAEEncoder
        from kandinsky6_mlx.pipelines.image import preprocess_image

        if self.vae_path is None:
            raise ValueError("image-to-video needs a pipeline loaded with from_converted (for the VAE encoder)")
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        pixels = preprocess_image(image, height, width, self.vae_scale_factor_spatial)[:, :, None]
        encoder = HunyuanVideoVAEEncoder.from_config(json.loads((self.vae_path / "config.json").read_text()))
        encoder.load_weights(
            HunyuanVideoVAEEncoder.encoder_weights(mx.load(str(self.vae_path / "model.safetensors"))), strict=True
        )
        mean, logvar = encoder.encode(mx.array(pixels))
        latents = HunyuanVideoVAEEncoder.sample(mean, logvar, mx.array(noise)) * encoder.scaling_factor
        latents = latents[:, :, 0].transpose(0, 2, 3, 1).astype(mx.float32)
        mx.eval(latents)
        del encoder
        mx.clear_cache()
        return latents

    def image_noise_shape(self, height: int, width: int) -> tuple:
        s = self.vae_scale_factor_spatial
        return (1, self.transformer.in_visual_dim, 1, height // s, width // s)

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
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        negative_text_embeds: dict[str, mx.array] | None = None,
        cfg_batched: bool = False,
        seed: int | None = None,
        latents: np.ndarray | mx.array | None = None,
        audio_latents: np.ndarray | mx.array | None = None,
        sample_audio: bool = True,
        return_step_latents: bool = False,
        release_transformer: bool = False,
        vae_tiling: TilingConfig | None = None,
        progress: bool = False,
        on_latents=None,
        image=None,
        image_noise: np.ndarray | mx.array | None = None,
        on_stage=None,
        clear_cache: bool = True,
    ) -> PipelineOutput:
        """Generate one video (batch 1). `release_transformer` frees the DiT after denoising, before the decoders run
        (about 8 GB at bf16); the pipeline then cannot be called again. `vae_tiling` decodes in spatial tiles as the
        reference does after `vae.enable_tiling(...)` (the Baseline preset sets the reference defaults); without it
        only the reference's default temporal tiling is used. `progress` shows tqdm bars for the denoising steps and
        the VAE tiles. `on_latents(latents)` receives the final video latents (diffusers layout, float32 NumPy) right
        after denoising, before the decoders run, so a caller can keep them even if decoding fails.

        `image` (a PIL image or a path) conditions the first frame as diffusers 0.42.1 does: its encoded latents are
        appended as an extra tail frame that is reset to them before every step, carries a conditioning mask of 1 and
        visual token type 1, and reuses the first frame's rotary position; the tail is dropped before decoding.
        `image_noise` is the standard-normal sample of the encoder's latent distribution (drawn from the seed, after
        the video and audio noise, when not given). Step latents then include the tail frame, like the reference's
        callback.

        `on_stage(name)` is called when a stage starts (`image_encode`, `denoise`, `video_decode`, `audio_decode`).
        `clear_cache=False` keeps MLX's buffer cache between stages: faster decoders, a higher peak."""
        if self.transformer is None:
            raise RuntimeError("the transformer was released by a previous call with release_transformer=True")
        if clear_cache:
            mx.clear_cache()  # buffers left by a previous call's decoders would push the DiT over the GPU working set
        distilled = self.scheduler_config.get("_class_name", "PiflowScheduler") == "PiflowScheduler"
        if num_inference_steps is None:
            num_inference_steps = 10 if distilled else 50
        if guidance_scale is None:
            guidance_scale = 1.0 if distilled else 5.0
        do_cfg = guidance_scale > 1.0
        if do_cfg and negative_text_embeds is None:
            raise ValueError("guidance_scale > 1 needs negative_text_embeds (e.g. of DEFAULT_NEGATIVE_PROMPT)")
        if sample_audio and (self.audio_vae is None or self.vocoder is None):
            raise ValueError("sample_audio=True needs the audio VAE and the vocoder")
        num_frames = self.round_num_frames(num_frames, self.vae_scale_factor_temporal)
        video_shape, audio_shape = self.latent_shapes(height, width, num_frames, frame_rate)
        rng = np.random.default_rng(seed)
        if latents is None:
            latents = rng.standard_normal(video_shape).astype(np.float32)
        if sample_audio and audio_latents is None:
            audio_latents = rng.standard_normal(audio_shape).astype(np.float32)
        first_frame = None
        if image is not None:
            if image_noise is None:
                image_noise = rng.standard_normal(self.image_noise_shape(height, width)).astype(np.float32)
            if on_stage is not None:
                on_stage("image_encode")
            image_start = time.perf_counter()
            first_frame = self.encode_image(image, height, width, image_noise)
            image_seconds = time.perf_counter() - image_start

        dtype = self.transformer.dtype
        # Transformer layouts: video (B, T, H, W, C), audio (B, L, C), in the transformer's dtype.
        video = mx.array(latents).transpose(0, 2, 3, 4, 1).astype(dtype)
        visual_token_type_ids = visual_rope_pos = None
        if first_frame is not None:
            # Tail conditioning: the reference frame as an extra latent frame of token type 1 at the first frame's
            # temporal rotary position.
            first_frame = first_frame.astype(dtype)
            num_latent_frames = video.shape[1]
            video = mx.concatenate([video, first_frame[:, None]], axis=1)
            visual_token_type_ids = mx.array([[0] * num_latent_frames + [1]])
            patch_t, patch_h, patch_w = self.patch_size
            visual_rope_pos = (
                mx.concatenate([mx.arange(num_latent_frames // patch_t), mx.zeros(1, dtype=mx.int32)]),
                mx.arange(video.shape[2] // patch_h),
                mx.arange(video.shape[3] // patch_w),
            )
        audio = mx.array(audio_latents).transpose(0, 2, 1).astype(dtype) if sample_audio else None

        def cast(embeds):
            return {k: (v.astype(dtype) if k != "encoder_attention_mask" else v) for k, v in embeds.items()}

        text = cast(text_embeds)
        negative = cast(negative_text_embeds) if do_cfg else None
        batched_text = pad_text(text, negative) if do_cfg and cfg_batched else None

        scheduler = make_scheduler(self.scheduler_config)
        scheduler.set_timesteps(num_inference_steps)
        audio_scheduler = make_scheduler(self.scheduler_config) if sample_audio else None
        if audio_scheduler is not None:
            audio_scheduler.set_timesteps(num_inference_steps)

        output = PipelineOutput(np.zeros(0), None, self.audio_sample_rate, frame_rate)
        if first_frame is not None:
            output.timings["image_encode"] = image_seconds
        if on_stage is not None:
            on_stage("denoise")
        mx.reset_peak_memory()
        start = time.perf_counter()
        from tqdm.auto import tqdm

        for t in tqdm(np.array(scheduler.timesteps), desc="denoise", unit="step", disable=not progress):
            step_start = time.perf_counter()
            timestep = mx.array(np.array([t], dtype=np.float32))
            # Visual conditioning channels: [latent | conditioning latent (always zero) | mask]; with an image the tail
            # frame is reset to the reference latents and its mask is 1.
            mask = mx.zeros((*video.shape[:-1], 1), dtype)
            if first_frame is not None:
                video = mx.concatenate([video[:, :-1], first_frame[:, None]], axis=1)
                mask = mx.concatenate([mask[:, :-1], mx.ones_like(mask[:, -1:])], axis=1)
            model_input = mx.concatenate([video, mx.zeros_like(video), mask], axis=-1)

            def predict(embeds, hidden, audio_hidden, token_types, t):
                return self.transformer(
                    hidden_states=hidden,
                    encoder_hidden_states=embeds["encoder_hidden_states"],
                    pooled_projections=embeds["pooled_projections"],
                    timestep=t,
                    audio_hidden_states=audio_hidden,
                    encoder_attention_mask=embeds.get("encoder_attention_mask"),
                    visual_rope_pos=visual_rope_pos,
                    visual_token_type_ids=token_types,
                )

            if not do_cfg:
                noise_pred = predict(text, model_input, audio, visual_token_type_ids, timestep)
            else:
                if cfg_batched:  # one call with batch 2 (cond, uncond): faster, more memory
                    two = lambda x: None if x is None else mx.concatenate([x, x], axis=0)
                    both = predict(
                        batched_text,
                        two(model_input),
                        two(audio),
                        two(visual_token_type_ids),
                        mx.concatenate([timestep, timestep]),
                    )
                    cond = tuple(None if x is None else x[:1] for x in both)
                    uncond = tuple(None if x is None else x[1:] for x in both)
                else:  # two calls, as diffusers runs them
                    cond = predict(text, model_input, audio, visual_token_type_ids, timestep)
                    uncond = predict(negative, model_input, audio, visual_token_type_ids, timestep)
                noise_pred = tuple(None if c is None else u + guidance_scale * (c - u) for c, u in zip(cond, uncond))
            video = scheduler.step(noise_pred[0], timestep, video)
            if sample_audio:
                audio = audio_scheduler.step(noise_pred[1], timestep, audio)
            mx.eval([x for x in (video, audio) if x is not None])
            output.step_seconds.append(time.perf_counter() - step_start)
            if return_step_latents:
                step_audio = None if audio is None else np.array(audio.astype(mx.float32))
                output.step_latents.append((np.array(video.astype(mx.float32)), step_audio))
        output.timings["denoise"] = time.perf_counter() - start
        output.peak_memory_gib["denoise"] = mx.get_peak_memory() / 2**30
        if release_transformer:
            self.transformer = None
            del noise_pred, model_input

        if first_frame is not None:
            video = video[:, :-1]  # drop the conditioning tail frame
        if on_stage is not None:
            on_stage("video_decode")
        if clear_cache:
            mx.clear_cache()
        mx.reset_peak_memory()
        start = time.perf_counter()
        final = video.transpose(0, 4, 1, 2, 3).astype(mx.float32)
        output.latents = np.array(final)
        if on_latents is not None:
            on_latents(output.latents)
        z = final / self.vae.scaling_factor
        with tqdm(
            total=self.vae.num_tiles(z.shape, vae_tiling), desc="video decode", unit="tile", disable=not progress
        ) as bar:
            frames = self.vae.decode(z, tiling=vae_tiling, callback=bar.update)
        frames = mx.clip(frames / 2 + 0.5, 0, 1).transpose(0, 2, 3, 4, 1)
        output.frames = np.array(frames)
        output.timings["video_decode"] = time.perf_counter() - start
        output.peak_memory_gib["video_decode"] = mx.get_peak_memory() / 2**30

        if sample_audio:
            if on_stage is not None:
                on_stage("audio_decode")
            if clear_cache:
                mx.clear_cache()
            mx.reset_peak_memory()
            start = time.perf_counter()
            z = audio.transpose(0, 2, 1).astype(mx.float32) / self.audio_vae.scaling_factor
            waveform = self.vocoder(self.audio_vae.decode(z))[:, 0]
            output.audio = np.array(waveform)
            output.timings["audio_decode"] = time.perf_counter() - start
            output.peak_memory_gib["audio_decode"] = mx.get_peak_memory() / 2**30
        return output
