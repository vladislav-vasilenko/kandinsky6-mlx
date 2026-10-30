---
id: 04-dit-blocks
title: DiT blocks — text encoder block, fused video/audio decoder block, output heads
depends_on: [03-dit-primitives]
estimate: 2 days
real_weights: none
---

## Goal

The three block types of the Kandinsky 6 DiT, matching the reference in float32 on tiny fixtures, including every branch of the fused video/audio block.

## Read first

`diffusers/models/transformers/transformer_kandinsky6.py` (0.42.1):

- `Kandinsky6OutLayer` (320–349): modulation → LayerNorm(no affine) → scale/shift → Linear → unpatchify `view(...).permute(0,1,5,2,6,3,7,4).flatten(1,2).flatten(2,3).flatten(3,4)`.
- `Kandinsky6OutLayerAudio` (352–367): note the second `self.norm(x)` after the affine; it is intentional (comment in the source).
- `Kandinsky6TransformerEncoderBlock` (370–402).
- `Kandinsky6TransformerDecoderBlock` (405–431): used only as a container of submodules.
- `Kandinsky6FusedTransformerDecoderBlock` (434–588): the core of the model.

## Deliverables

- `src/kandinsky6_mlx/models/dit/blocks.py`.
- `tests/parity/test_dit_blocks.py`.
- `reports/04-dit-blocks.json`.

## Implementation notes

The fused block forward, in order (follow the source exactly):

1. Video: `visual_modulation(t_v)` split into self-attention, cross-attention and FFN parameter groups; self-attention with `vis_rope`; gated residual; compute `vis_pre_ca` and the text cross-attention output `vis_out_t`, **but do not add it yet** when audio is present.
2. Audio: same with `t_a`, `aud_rope`; the audio text cross-attention **is** added immediately.
3. If both present: `va_modulation`/`av_modulation` use `t_a`/`t_v` swapped when `fix_modulation=True`; with `cross_gates=True` the parameters are split `[model_dim, model_dim, model_dim_a]` and the **gates are swapped** (`vis` uses `av_gate`, `aud` uses `va_gate`). Then add `vis_out_t` to `vis`, compute `va_cross_attention(vis_for_va, aud_pre_ca)` and `av_cross_attention(aud_for_av, vis_pre_ca)` with `rope_q`/`rope_kv` set only when `ca_rope=True`.
4. If audio absent: add `vis_out_t` to `vis` (the `elif` branch).
5. FFN for video and audio.

Note that cross-modal attention uses the **pre-cross-attention** normalized states (`aud_pre_ca`, `vis_pre_ca`) as keys/values, not the updated streams. Every `apply_gate_sum` result is cast back with `.type_as(...)`.

## Tests

Tiny fixtures, float32:

- Encoder block with and without text padding mask.
- Fused block in all four flag combinations that matter: (`cross_gates`, `fix_modulation`) ∈ {(T,T), (T,F), (F,T), (F,F)}, and `ca_rope` on/off. The released checkpoints use all `True`; the others guard against swapped-gate bugs.
- Fused block with `aud=None` (video-only path).
- Output heads: video unpatchify element order with distinct-integer inputs; audio head double norm.

## Validation runs

```bash
uv run --group reference pytest tests/parity/test_dit_blocks.py -q
```

## Acceptance criteria

- [ ] Every block variant passes float32 parity at `rel_max <= 1e-4`.
- [ ] A deliberately wrong gate swap (test-only monkeypatch) is detected by the tests, proving they cover that branch.

## Out of scope

Full model assembly (spec 05).

## Stop conditions

A branch of the reference cannot be reached with tiny configs → record which and why.
