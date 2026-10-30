"""Spec 04: MLX DiT blocks and output heads against the diffusers 0.42.1 reference."""

import itertools

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.models.dit import blocks as B
from kandinsky6_mlx.models.dit import primitives as P
from kandinsky6_mlx.utils.compare import compare
from kandinsky6_mlx.utils.fixtures import load_fixture, load_module_weights, unflatten
from tests.parity.dit_cases import BLOCK_CLASSES, TINY, block_results, build_block

RESULTS = block_results()


def test_every_block_class_and_flag_combination_is_covered():
    assert {result["class"] for result in RESULTS} == set(BLOCK_CLASSES)
    _, meta = load_fixture(TINY / "dit_variants", backend="numpy")
    combos = {(f["cross_gates"], f["fix_modulation"], f["ca_rope"]) for f in meta["variants"].values()}
    combos.add((True, True, True))  # released flags, recorded in the `dit` fixture
    assert combos == set(itertools.product((False, True), repeat=3))
    assert any(
        r["scenario"] == "video_only" and r["class"] == "Kandinsky6FusedTransformerDecoderBlock" for r in RESULTS
    )


@pytest.mark.parametrize("result", RESULTS, ids=[f"{r['scenario']}-{r['module']}" for r in RESULTS])
def test_recorded_call_matches_reference(result):
    assert result["rel_max"] <= result["tolerance"], result


def _load_torch_weights(module, torch_module):
    module.load_weights([(k, mx.array(v.detach().float().numpy())) for k, v in torch_module.state_dict().items()])


@pytest.mark.parametrize("masked", [False, True])
def test_encoder_block_with_and_without_padding_mask(masked):
    torch = pytest.importorskip("torch")
    from diffusers.models.transformers.transformer_kandinsky6 import (
        Kandinsky6RoPE1D,
        Kandinsky6TransformerEncoderBlock,
    )

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ref_block = Kandinsky6TransformerEncoderBlock(32, 16, 64, 8).eval()
    block = B.Kandinsky6TransformerEncoderBlock(32, 16, 64, 8)
    _load_torch_weights(block, ref_block)
    x = rng.standard_normal((2, 5, 32)).astype(np.float32)
    t = rng.standard_normal((2, 16)).astype(np.float32)
    mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], bool) if masked else None
    rope = Kandinsky6RoPE1D(8)(torch.arange(5))
    with torch.no_grad():
        ref = ref_block(
            torch.from_numpy(x), torch.from_numpy(t), rope, None if mask is None else torch.from_numpy(mask)
        )
    got = block(mx.array(x), mx.array(t), mx.array(rope.numpy()), None if mask is None else mx.array(mask))
    assert compare(got, ref)["rel_max"] <= 1e-4


def _fused_pair(cross_gates: bool, fix_modulation: bool, ca_rope: bool):
    """Reference and MLX fused blocks with equal video/audio widths, so a swapped gate is a numeric, not shape, bug."""
    import torch
    from diffusers.models.transformers.transformer_kandinsky6 import Kandinsky6FusedTransformerDecoderBlock

    torch.manual_seed(0)
    args = (16, 16, 32, 8, 16, 16, 32, 8)
    flags = {"ca_rope": ca_rope, "cross_gates": cross_gates, "fix_modulation": fix_modulation}
    ref_block = Kandinsky6FusedTransformerDecoderBlock(*args, **flags).eval()
    block = B.Kandinsky6FusedTransformerDecoderBlock(*args, **flags)
    _load_torch_weights(block, ref_block)
    return ref_block, block


def _fused_inputs():
    import torch
    from diffusers.models.transformers.transformer_kandinsky6 import Kandinsky6RoPE1D, Kandinsky6RoPE3D

    rng = np.random.default_rng(1)
    arrays = {
        "vis": rng.standard_normal((1, 12, 16)),
        "aud": rng.standard_normal((1, 6, 16)),
        "text_v": rng.standard_normal((1, 5, 16)),
        "text_a": rng.standard_normal((1, 5, 16)),
        "t_v": rng.standard_normal((1, 16)),
        "t_a": rng.standard_normal((1, 16)),
    }
    arrays = {k: v.astype(np.float32) for k, v in arrays.items()}
    vis_rope = Kandinsky6RoPE3D((2, 2, 4))((torch.arange(2), torch.arange(2), torch.arange(3))).reshape(12, 1, 4, 2, 2)
    aud_rope = Kandinsky6RoPE1D(8, freqs_scaling=0.144)(torch.arange(6))
    arrays["vis_rope"], arrays["aud_rope"] = vis_rope.numpy(), aud_rope.numpy()
    arrays["mask"] = np.array([[1, 1, 1, 1, 0]], bool)
    return arrays


def _run(block, arrays, convert, audio=True):
    a = {k: convert(v) for k, v in arrays.items()}
    return block(
        a["vis"],
        a["aud"] if audio else None,
        a["text_v"],
        a["text_a"],
        (a["t_v"], a["t_a"]),
        a["vis_rope"],
        a["aud_rope"] if audio else None,
        a["mask"],
    )


def _fused_rel_max(ref_block, block, audio=True) -> float:
    import torch

    arrays = _fused_inputs()
    with torch.no_grad():
        ref = _run(ref_block, arrays, torch.from_numpy, audio)
    got = _run(block, arrays, mx.array, audio)
    assert (got[1] is None) == (ref[1] is None)
    return max(compare(g, r)["rel_max"] for g, r in zip(got, ref) if r is not None)


FLAGS = list(itertools.product((False, True), repeat=3))


@pytest.mark.parametrize(("cross_gates", "fix_modulation", "ca_rope"), FLAGS)
def test_fused_block_equal_widths(cross_gates, fix_modulation, ca_rope):
    pytest.importorskip("torch")
    ref_block, block = _fused_pair(cross_gates, fix_modulation, ca_rope)
    assert _fused_rel_max(ref_block, block) <= 1e-4


def test_fused_block_video_only_equal_widths():
    pytest.importorskip("torch")
    ref_block, block = _fused_pair(True, True, True)
    assert _fused_rel_max(ref_block, block, audio=False) <= 1e-4


@pytest.mark.parametrize("cross_gates", [False, True])
def test_wrong_gate_swap_is_detected(cross_gates, monkeypatch):
    """Inverting the `cross_gates` gate swap must break parity, so the tests above cover that branch."""
    pytest.importorskip("torch")
    ref_block, block = _fused_pair(cross_gates, True, True)
    assert _fused_rel_max(ref_block, block) <= 1e-4

    def wrong_gates(self, va_gate, av_gate):
        return (va_gate, av_gate) if self.cross_gates else (av_gate, va_gate)

    monkeypatch.setattr(B.Kandinsky6FusedTransformerDecoderBlock, "_cross_modal_gates", wrong_gates)
    assert _fused_rel_max(ref_block, block) > 1e-2


def test_wrong_gate_swap_is_detected_on_recorded_fixture(monkeypatch):
    """On the released tiny config (video width 32, audio 16) a swap cannot even run."""

    def wrong_gates(self, va_gate, av_gate):
        return va_gate, av_gate

    monkeypatch.setattr(B.Kandinsky6FusedTransformerDecoderBlock, "_cross_modal_gates", wrong_gates)
    with pytest.raises(ValueError):
        block_results()


def test_unpatchify_element_order():
    torch = pytest.importorskip("torch")
    p_t, p_h, p_w = 1, 2, 2
    x = np.arange(2 * 3 * 2 * 3 * (5 * p_t * p_h * p_w), dtype=np.float32).reshape(2, 3, 2, 3, -1)
    ref = (
        torch.from_numpy(x)
        .view(2, 3, 2, 3, -1, p_t, p_h, p_w)
        .permute(0, 1, 5, 2, 6, 3, 7, 4)
        .flatten(1, 2)
        .flatten(2, 3)
        .flatten(3, 4)
        .numpy()
    )
    got = np.array(B.unpatchify(mx.array(x), (p_t, p_h, p_w)))
    assert got.shape == (2, 3, 4, 6, 5)
    np.testing.assert_array_equal(got, ref)


def test_unpatchify_is_not_inverse_of_patchify():
    """The head's channel-slowest order differs from patchify's channel-fastest order; guard against "fixing" it."""
    x = mx.array(np.arange(1 * 2 * 4 * 4 * 3, dtype=np.float32).reshape(1, 2, 4, 4, 3))
    roundtrip = B.unpatchify(P.patchify(x, (1, 2, 2)), (1, 2, 2))
    assert not np.array_equal(np.array(roundtrip), np.array(x))


def test_audio_head_double_norm_is_required():
    """Dropping the second LayerNorm of the audio head must break parity with the recorded output."""
    tensors, meta = load_fixture(TINY / "dit", backend="mlx")
    call = next(c for c in meta["calls"]["av"] if c["class"] == "Kandinsky6OutLayerAudio")
    head = build_block(call["class"], call["module"], tensors, meta["config"])
    load_module_weights(head, tensors, f"weights.{call['module']}")
    audio_embed, time_embed = unflatten(call["args"], tensors)
    reference = unflatten(call["output"], tensors)

    assert compare(head(audio_embed, time_embed), reference)["rel_max"] <= 1e-4
    shift, scale = mx.split(head.modulation(time_embed), 2, axis=-1)
    single = head.out_layer(P.apply_scale_shift(head.norm(audio_embed), audio_embed, scale, shift))
    assert compare(single, reference)["rel_max"] > 1e-2
