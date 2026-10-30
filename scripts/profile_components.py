"""Profile where time and memory go in the MLX components on the real Lite-distill weights.

Feeds the `architecture_notes` of the reports. Inputs are random NumPy arrays of the Smoke or Baseline sizes; timings
are medians of three evaluated runs after one warm-up, on the default device (GPU).

    uv run python scripts/profile_components.py --weights weights/lite-distill [--sections dit text_trim vocoder load]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from kandinsky6_mlx.models.dit.model import Kandinsky6Transformer3DModel
from kandinsky6_mlx.models.dit.primitives import apply_gate_sum, apply_rotary, apply_scale_shift

# Latent grids after patchify: (T', H', W') for Smoke (256x432, 25 frames) and Baseline (480x864, 121 frames).
GRIDS = {"smoke": (7, 16, 27), "baseline": (31, 30, 54)}
AUDIO = {"smoke": 45, "baseline": 218}
TEXT, TEXT_VALID = 1024, 200


def timed(fn, repeats: int = 3) -> float:
    mx.eval(fn())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        times.append(time.perf_counter() - start)
    return round(statistics.median(times), 4)


def rand(*shape, dtype=mx.bfloat16) -> mx.array:
    return mx.array(np.random.default_rng(len(shape)).standard_normal(shape).astype(np.float32)).astype(dtype)


def profile_dit(weights: Path, size: str) -> dict:
    model = Kandinsky6Transformer3DModel.from_converted(weights / "transformer", mx.bfloat16)
    block = model.visual_transformer_blocks[0]
    video = block.video_dec_block
    grid = GRIDS[size]
    tokens = int(np.prod(grid))
    vis, aud = rand(1, tokens, 1792), rand(1, AUDIO[size], 896)
    text_v, text_a = rand(1, TEXT, 1792), rand(1, TEXT, 896)
    t_v, t_a = rand(1, 512, dtype=mx.float32), rand(1, 512, dtype=mx.float32)
    mask = mx.array(np.arange(TEXT)[None] < TEXT_VALID)
    vis_rope = model.visual_rope_embeddings(tuple(mx.arange(n) for n in grid), model.scale_factor).reshape(
        tokens, 1, 32, 2, 2
    )
    aud_rope = model.audio_rope_embeddings(mx.arange(AUDIO[size]))
    heads = 1792 // 64
    q = rand(1, heads, tokens, 64)
    q_lhd = rand(1, tokens, heads, 64)
    params = video.visual_modulation(t_v)
    shift, scale, gate = mx.split(mx.split(params, 3, axis=-1)[0], 3, axis=-1)

    result = {
        "video_tokens": tokens,
        "fused_block_total": timed(lambda: block(vis, aud, text_v, text_a, (t_v, t_a), vis_rope, aud_rope, mask)),
        "video_self_attention_module": timed(lambda: video.self_attention(vis, rotary_emb=vis_rope)),
        "sdpa_only_self": timed(lambda: mx.fast.scaled_dot_product_attention(q, q, q, scale=0.125)),
        "apply_rotary_one_tensor": timed(lambda: apply_rotary(q_lhd, vis_rope)),
        "text_cross_attention_1024_keys": timed(
            lambda: video.cross_attention(vis, encoder_hidden_states=text_v, attn_mask=mask)
        ),
        "text_cross_attention_200_keys_trimmed": timed(
            lambda: video.cross_attention(vis, encoder_hidden_states=text_v[:, :TEXT_VALID])
        ),
        "va_cross_attention_video_queries": timed(
            lambda: block.va_cross_attention(vis, encoder_hidden_states=aud, rope_q=vis_rope, rope_kv=aud_rope)
        ),
        "av_cross_attention_audio_queries": timed(
            lambda: block.av_cross_attention(aud, encoder_hidden_states=vis, rope_q=aud_rope, rope_kv=vis_rope)
        ),
        "video_feed_forward": timed(lambda: video.feed_forward(vis)),
        "norm_scale_shift_gate_glue_once": timed(
            lambda: apply_gate_sum(
                vis, apply_scale_shift(video.self_attention_norm(vis.astype(mx.float32)), vis, scale, shift), gate
            )
        ),
        "visual_rope_table_build": timed(
            lambda: model.visual_rope_embeddings(tuple(mx.arange(n) for n in grid), model.scale_factor)
        ),
        "rope_table_mib": round(vis_rope.nbytes / 2**20, 1),
    }
    encoder = model.video_text_transformer_blocks[0]
    text_rope = model.video_text_rope_embeddings(mx.arange(TEXT))
    result["text_encoder_block_1024"] = timed(lambda: encoder(text_v, t_v, text_rope, mask))
    result["text_encoder_block_200_trimmed"] = timed(
        lambda: encoder(text_v[:, :TEXT_VALID], t_v, text_rope[:TEXT_VALID])
    )
    result["projected_full_forward_s"] = round(
        32 * result["fused_block_total"] + 4 * result["text_encoder_block_1024"], 2
    )
    return result


def profile_text_trim(weights: Path) -> dict:
    """Whole model at Smoke size in float32: padded text with mask vs the same text with padding removed."""
    model = Kandinsky6Transformer3DModel.from_converted(weights / "transformer", mx.float32)
    rng = np.random.default_rng(0)
    grid = GRIDS["smoke"]
    latents = rng.standard_normal((1, grid[0], grid[1] * 2, grid[2] * 2, 33)).astype(np.float32)
    text = rng.standard_normal((1, TEXT, 3584)).astype(np.float32)
    common = {
        "hidden_states": mx.array(latents),
        "pooled_projections": mx.array(rng.standard_normal((1, 768)).astype(np.float32)),
        "timestep": mx.array([833.33]),
        "audio_hidden_states": mx.array(rng.standard_normal((1, 45, 40)).astype(np.float32)),
    }
    mask = mx.array(np.arange(TEXT)[None] < TEXT_VALID)
    full = model(**common, encoder_hidden_states=mx.array(text), encoder_attention_mask=mask)
    trimmed = model(**common, encoder_hidden_states=mx.array(text[:, :TEXT_VALID]))
    mx.eval(full, trimmed)
    out = {}
    for name, a, b in zip(("video", "audio"), full, trimmed):
        a, b = np.array(a, np.float64), np.array(b, np.float64)
        out[name] = {"rel_max": float(np.abs(a - b).max() / np.abs(a).max())}
    return out


def profile_vocoder(weights: Path, size: str) -> dict:
    from tests.parity.test_audio_real import load_audio

    vae, vocoder = load_audio(weights)
    mel = vae.decode(rand(1, 40, AUDIO[size], dtype=mx.float32))
    mx.eval(mel)
    result = {"audio_vae_decode": timed(lambda: vae.decode(rand(1, 40, AUDIO[size], dtype=mx.float32)))}
    x = vocoder.conv_pre(mel.transpose(0, 2, 1))
    mx.eval(x)
    stages = []
    for stage, up in enumerate(vocoder.ups):
        y = up[0](x)
        mx.eval(y)
        blocks = vocoder.resblocks[stage * vocoder.num_kernels : (stage + 1) * vocoder.num_kernels]
        act = blocks[0].activations[0]
        stages.append(
            {
                "length": y.shape[1],
                "channels": y.shape[2],
                "upsample": timed(lambda up=up, x=x: up[0](x)),
                "amp_blocks": timed(lambda y=y, blocks=blocks: sum(b(y) for b in blocks)),
                "one_antialiased_activation": timed(lambda act=act, y=y: act(y)),
                "activations_per_stage": sum(len(b.activations) for b in blocks),
            }
        )
        x = sum(b(y) for b in blocks) / vocoder.num_kernels
        mx.eval(x)
    result["stages"] = stages
    result["vocoder_total"] = timed(lambda: vocoder(mel))
    return result


def profile_load(weights: Path) -> dict:
    start = time.perf_counter()
    Kandinsky6Transformer3DModel.from_converted(weights / "transformer", mx.bfloat16)
    return {"transformer_from_converted_bf16_s": round(time.perf_counter() - start, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, default=Path("weights/lite-distill"))
    parser.add_argument("--sections", nargs="+", default=["load", "dit", "text_trim", "vocoder"])
    parser.add_argument("--out", type=Path, default=Path("outputs/profile/components.json"))
    args = parser.parse_args()
    results = {}
    for section in args.sections:
        if section == "dit":
            results["dit"] = {size: profile_dit(args.weights, size) for size in ("smoke", "baseline")}
        elif section == "text_trim":
            results["text_trim_float32_smoke"] = profile_text_trim(args.weights)
        elif section == "vocoder":
            results["audio"] = {size: profile_vocoder(args.weights, size) for size in ("smoke", "baseline")}
        elif section == "load":
            results["load"] = profile_load(args.weights)
        mx.clear_cache()
        print(json.dumps({section: results.get(section) or results}, indent=1)[:4000], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
