---
id: 03-dit-primitives
title: DiT building blocks — RoPE, modulation, attention, feed-forward, embeddings
depends_on: [02-weight-conversion]
estimate: 2 days
real_weights: none
---

## Goal

MLX versions of every small component the Kandinsky 6 DiT is assembled from, each matching the reference to float32 precision on tiny fixtures.

## Read first

`diffusers/models/transformers/transformer_kandinsky6.py` (0.42.1):

- `get_freqs` (52–63), `apply_scale_shift` (66–75), `apply_gate_sum` (78–82), `apply_rotary` (85–88).
- `Kandinsky6RoPE1D` (91–113), `Kandinsky6RoPE3D` (116–157).
- `Kandinsky6AttnProcessor` (160–207), `Kandinsky6Attention` (271–317).
- `Kandinsky6TimeEmbeddings` (210–225) and diffusers `Timesteps`, `TimestepEmbedding` (`diffusers/models/embeddings.py`).
- `Kandinsky6TextEmbeddings` (228–237), `Kandinsky6VisualEmbeddings` (240–256), `Kandinsky6Modulation` (259–268).
- `FeedForward(activation_fn="gelu", bias=False)` from `diffusers/models/attention.py`: parameter names (`net.0.proj`, `net.2`) and the exact GELU variant.

## Deliverables

- `src/kandinsky6_mlx/models/dit/primitives.py` with the functions and modules above, same names and parameter paths as the reference.
- `tests/parity/test_dit_primitives.py`.
- `reports/03-dit-primitives.json`.

## Implementation notes

- RoPE tables are built in float32 and returned with shapes `(L, 1, D/2, 2, 2)` (1D) and `(T, H, W, 1, D/2, 2, 2)` (3D). `apply_rotary` multiplies pairs by 2×2 matrices; keep this formulation rather than converting to complex/cos-sin form, so parity is exact.
- `Kandinsky6RoPE3D.forward` divides angles per axis by `scale_factor` (released config `(1.0, 2.0, 2.0)`).
- Attention: projections give `(B, L, H, D)`; `query_norm`/`key_norm` are `RMSNorm(head_dim)` applied per head; RoPE is applied after the norms; `rotary_emb_kv` defaults to `rotary_emb`. `mx.fast.scaled_dot_product_attention` needs `(B, H, L, D)` — transpose in and out. Use scale `head_dim ** -0.5`.
- `attn_mask` for text is a boolean padding mask `(B, L_text)`; implement it as an additive mask for the SDPA call and test with and without padding.
- `apply_scale_shift` and `apply_gate_sum` compute in float32 and cast back to the input dtype; replicate exactly, including the reshape of `(B, C)` conditions to broadcast over token dims.
- `Kandinsky6VisualEmbeddings` patchifies `(B, T, H, W, C)` with patch `(1, 2, 2)` via `view` + `permute(0,1,3,5,2,4,6,7)` + `flatten(4,7)` then a `Linear`. Reproduce the element order exactly; a wrong permute still produces finite numbers.
- `Kandinsky6TimeEmbeddings`: `Timesteps(model_dim, flip_sin_to_cos=True, downscale_freq_shift=0)` then `TimestepEmbedding(..., act_fn="silu")`; float32.

## Tests

For each primitive, using tiny fixtures from spec 01 in float32:

- RoPE1D and RoPE3D tables (`rel_max <= 1e-6`), including non-default `visual_rope_pos` (a temporal index that repeats 0, as used by I2V).
- `apply_rotary` on random q/k.
- Attention: self-attention, cross-attention with `kv_dim != dim`, separate `rope_q`/`rope_kv`, with and without text padding mask.
- FeedForward, Modulation, Time/Text/Visual embeddings.
- `apply_scale_shift`/`apply_gate_sum` with `(B, C)` and `(B, 1, C)` conditions, bf16 inputs (output dtype must be bf16).

## Validation runs

```bash
uv run --group reference pytest tests/parity/test_dit_primitives.py -q
```

Record per-primitive `rel_max` in the report.

## Acceptance criteria

- [ ] Every primitive passes float32 parity at `rel_max <= 1e-5` (attention and FFN may use `<= 1e-4` with the measured value in the report).
- [ ] Visual patchify order verified by a test that patchifies a tensor of distinct integers and compares element positions exactly.
- [ ] bf16 inputs return bf16 outputs where the reference does.

## Out of scope

Blocks and the full model (specs 04, 05).

## Stop conditions

`mx.fast.scaled_dot_product_attention` cannot express the mask semantics → implement the plain softmax path, keep parity, and record the deviation.
