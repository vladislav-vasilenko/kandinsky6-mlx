"""Spec 03: MLX DiT primitives against the diffusers 0.42.1 reference."""

import math

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.models.dit import primitives as P
from kandinsky6_mlx.utils.compare import compare
from tests.parity.dit_cases import PRIMITIVE_CLASSES, primitive_results

RESULTS = primitive_results()


def test_every_primitive_class_is_covered():
    assert {result["class"] for result in RESULTS} == set(PRIMITIVE_CLASSES)


def test_attention_paths_are_covered():
    attention = [result["module"] for result in RESULTS if result["class"] == "Kandinsky6Attention"]
    # Text self-attention with the padding mask, video/audio self- and cross-attention, cross-modal attention.
    assert any("text_transformer_blocks" in name for name in attention)
    assert any(name.endswith("va_cross_attention") for name in attention)
    assert any(name.endswith("av_cross_attention") for name in attention)


@pytest.mark.parametrize("result", RESULTS, ids=[r["module"] for r in RESULTS])
def test_recorded_call_matches_reference(result):
    assert result["rel_max"] <= result["tolerance"], result


def test_patchify_element_order():
    torch = pytest.importorskip("torch")
    x = np.arange(2 * 3 * 4 * 6 * 5, dtype=np.float32).reshape(2, 3, 4, 6, 5)
    p_t, p_h, p_w = 1, 2, 2
    ref = (
        torch.from_numpy(x)
        .view(2, 3 // p_t, p_t, 4 // p_h, p_h, 6 // p_w, p_w, 5)
        .permute(0, 1, 3, 5, 2, 4, 6, 7)
        .flatten(4, 7)
        .numpy()
    )
    np.testing.assert_array_equal(np.array(P.patchify(mx.array(x), (p_t, p_h, p_w))), ref)


def test_rope3d_with_repeated_temporal_position():
    torch = pytest.importorskip("torch")
    from diffusers.models.transformers.transformer_kandinsky6 import Kandinsky6RoPE3D

    axes = (16, 24, 24)
    pos = (np.array([0, 1, 2, 0]), np.arange(3), np.arange(5))
    ref = Kandinsky6RoPE3D(axes)(tuple(torch.from_numpy(p) for p in pos), (1.0, 2.0, 2.0))
    got = P.Kandinsky6RoPE3D(axes)(tuple(mx.array(p) for p in pos), (1.0, 2.0, 2.0))
    assert compare(got, ref)["rel_max"] <= 1e-6


def _float64_freqs(dim: int) -> np.ndarray:
    argument = (np.float32(-math.log(10000.0)) * np.arange(dim, dtype=np.float32)) / np.float32(dim)
    return np.exp(argument.astype(np.float64))


@pytest.mark.parametrize(("head_dim", "scaling"), [(64, 1.0), (64, 0.144), (128, 1.0), (128, 0.144)])
def test_rope1d_long_positions_as_accurate_as_reference(head_dim, scaling):
    """Over 2048 positions float32 angle rounding dominates; the port must be no less accurate than the reference."""
    torch = pytest.importorskip("torch")
    from diffusers.models.transformers.transformer_kandinsky6 import Kandinsky6RoPE1D

    pos = np.arange(2048)
    angles = pos.astype(np.float64)[:, None] * (_float64_freqs(head_dim // 2) * np.float32(scaling))[None]
    truth = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)], -1)
    truth = truth.reshape(2048, 1, head_dim // 2, 2, 2)
    ref = Kandinsky6RoPE1D(head_dim, freqs_scaling=scaling)(torch.from_numpy(pos)).numpy()
    got = np.array(P.Kandinsky6RoPE1D(head_dim, freqs_scaling=scaling)(mx.array(pos)))
    ref_error, got_error = np.abs(ref - truth).max(), np.abs(got - truth).max()
    assert got_error <= 1.1 * ref_error + 1e-7, (got_error, ref_error)
    assert np.abs(got - ref).max() <= 2 * ref_error + 1e-7


@pytest.mark.parametrize("dim", [512, 1024])
def test_timestep_embedding_as_accurate_as_reference(dim):
    torch = pytest.importorskip("torch")
    from diffusers.models.embeddings import Timesteps

    t = np.linspace(0, 1000, 37).astype(np.float32)
    angles = t.astype(np.float64)[:, None] * _float64_freqs(dim // 2)[None]
    truth = np.concatenate([np.cos(angles), np.sin(angles)], -1)
    ref = Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)(torch.from_numpy(t)).numpy()
    got = np.array(P.timestep_embedding(mx.array(t), dim))
    assert np.abs(got - truth).max() <= 1.1 * np.abs(ref - truth).max() + 1e-7


@pytest.mark.parametrize("cond_shape", [(2, 8), (2, 1, 8)])
def test_scale_shift_and_gate_in_bf16(cond_shape):
    torch = pytest.importorskip("torch")
    from diffusers.models.transformers.transformer_kandinsky6 import apply_gate_sum, apply_scale_shift

    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 5, 8)).astype(np.float32)
    normed = rng.standard_normal((2, 5, 8)).astype(np.float32)
    scale, shift, gate = (rng.standard_normal(cond_shape).astype(np.float32) for _ in range(3))

    def t(a, dtype=torch.bfloat16):
        return torch.from_numpy(a).to(dtype)

    def m(a, dtype=mx.bfloat16):
        return mx.array(a).astype(dtype)

    ref = apply_scale_shift(t(normed, torch.float32), t(x), t(scale), t(shift))
    got = P.apply_scale_shift(m(normed, mx.float32), m(x), m(scale), m(shift))
    assert got.dtype == mx.bfloat16
    np.testing.assert_array_equal(np.array(got.astype(mx.float32)), ref.float().numpy())

    ref = apply_gate_sum(t(x), t(normed), t(gate))
    got = P.apply_gate_sum(m(x), m(normed), m(gate))
    assert got.dtype == mx.bfloat16
    np.testing.assert_array_equal(np.array(got.astype(mx.float32)), ref.float().numpy())


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_rms_norm_matches_torch(dtype):
    """torch.nn.RMSNorm(eps=None) upcasts bf16 to float32 and uses the float32 epsilon; bf16 output is bit-exact."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = (1e-2 * rng.standard_normal((3, 4, 64))).astype(np.float32)
    w = (1 + 0.3 * rng.standard_normal(64)).astype(np.float32)
    torch_dtype, mx_dtype = getattr(torch, dtype), getattr(mx, dtype)
    ref_norm = torch.nn.RMSNorm(64).to(torch_dtype)
    with torch.no_grad():
        ref_norm.weight.copy_(torch.from_numpy(w))
        ref = ref_norm(torch.from_numpy(x).to(torch_dtype)).float().numpy()
    norm = P.RMSNorm(64)
    norm.weight = mx.array(w).astype(mx_dtype)
    got = norm(mx.array(x).astype(mx_dtype))
    assert got.dtype == mx_dtype
    if dtype == "bfloat16":
        # The float32 reduction order differs between backends; rounding to bf16 absorbs it.
        np.testing.assert_array_equal(np.array(got.astype(mx.float32)), ref)
    else:
        assert compare(got, ref)["rel_max"] <= 1e-6


def _load_torch_weights(module, torch_module):
    module.load_weights([(k, mx.array(v.detach().float().numpy())) for k, v in torch_module.state_dict().items()])


def test_apply_rotary_on_random_qk():
    torch = pytest.importorskip("torch")
    from diffusers.models.transformers.transformer_kandinsky6 import Kandinsky6RoPE1D, apply_rotary

    x = np.random.default_rng(0).standard_normal((2, 7, 3, 8)).astype(np.float32)
    pos = np.arange(7)
    ref = apply_rotary(torch.from_numpy(x), Kandinsky6RoPE1D(8)(torch.from_numpy(pos)))
    got = P.apply_rotary(mx.array(x), P.Kandinsky6RoPE1D(8)(mx.array(pos)))
    assert compare(got, ref)["rel_max"] <= 1e-6


def test_attention_mask_with_batch_of_two():
    """Rows padded differently: a (B, L_kv) mask must hide keys per batch row, not broadcast over queries."""
    torch = pytest.importorskip("torch")
    from diffusers.models.transformers.transformer_kandinsky6 import Kandinsky6Attention, Kandinsky6RoPE1D

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ref_attn = Kandinsky6Attention(32, 8, kv_dim=16).eval()
    attn = P.Kandinsky6Attention(32, 8, 16)
    _load_torch_weights(attn, ref_attn)
    hidden = rng.standard_normal((2, 6, 32)).astype(np.float32)
    context = rng.standard_normal((2, 5, 16)).astype(np.float32)
    mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], bool)
    rope_q, rope_kv = Kandinsky6RoPE1D(8)(torch.arange(6)), Kandinsky6RoPE1D(8)(torch.arange(5))
    with torch.no_grad():
        ref = ref_attn(
            torch.from_numpy(hidden), torch.from_numpy(context), torch.from_numpy(mask), rope_q=rope_q, rope_kv=rope_kv
        )
    got = attn(
        mx.array(hidden),
        mx.array(context),
        mx.array(mask),
        rope_q=mx.array(rope_q.numpy()),
        rope_kv=mx.array(rope_kv.numpy()),
    )
    assert compare(got, ref)["rel_max"] <= 1e-4


def test_bf16_dtype_propagation():
    """Attention and FFN in bf16 return bf16; modulation with float32 weights returns float32 for bf16 input."""
    torch = pytest.importorskip("torch")
    from diffusers.models.attention import FeedForward
    from diffusers.models.transformers.transformer_kandinsky6 import Kandinsky6Attention, Kandinsky6Modulation

    torch.manual_seed(0)
    x = np.random.default_rng(0).standard_normal((1, 6, 32)).astype(np.float32)

    ref_attn = Kandinsky6Attention(32, 8).to(torch.bfloat16)
    attn = P.Kandinsky6Attention(32, 8)
    _load_torch_weights(attn, ref_attn)
    attn.set_dtype(mx.bfloat16)
    with torch.no_grad():
        ref = ref_attn(torch.from_numpy(x).bfloat16())
    got = attn(mx.array(x).astype(mx.bfloat16))
    assert got.dtype == mx.bfloat16 and ref.dtype == torch.bfloat16
    assert compare(got, ref.float())["cosine"] >= 0.999

    ref_ff = FeedForward(32, inner_dim=64, activation_fn="gelu", bias=False).to(torch.bfloat16)
    ff = P.FeedForward(32, 64, bias=False)
    _load_torch_weights(ff, ref_ff)
    ff.set_dtype(mx.bfloat16)
    with torch.no_grad():
        assert ref_ff(torch.from_numpy(x).bfloat16()).dtype == torch.bfloat16
    assert ff(mx.array(x).astype(mx.bfloat16)).dtype == mx.bfloat16

    ref_mod = Kandinsky6Modulation(16, 32, 3)
    mod = P.Kandinsky6Modulation(16, 32, 3)
    _load_torch_weights(mod, ref_mod)
    with torch.no_grad():
        assert ref_mod(torch.ones(1, 16, dtype=torch.bfloat16)).dtype == torch.float32
    assert mod(mx.ones((1, 16), dtype=mx.bfloat16)).dtype == mx.float32
