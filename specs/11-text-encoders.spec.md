---
id: 11-text-encoders
title: Qwen2.5-VL hidden states and CLIP-L pooled embedding in MLX
depends_on: [09-pipeline-t2v-mvp]
estimate: 2 days
real_weights: stage-b
---

## Goal

Replace cached text embeddings by MLX text encoding so a prompt alone produces a video.

## Read first

- `pipeline_kandinsky6_ti2va.py`: `_get_prompt_embeds` (263), `encode_prompt` (309), `expand_prompts` (413): prompt template, which hidden-state layer is used, how template tokens are cropped, attention mask, max length, CLIP pooled output.
- K6 `text_encoder/` shards are bit-identical (SHA-256) to `Qwen/Qwen2.5-VL-7B-Instruct`; MLX conversions exist in `mlx-community/Qwen2.5-VL-7B-Instruct-{bf16,8bit,6bit,4bit,3bit}`.
- MLX Qwen2.5-VL implementation to depend on: https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen2_5_vl (MIT). Pin an exact `mlx-vlm` version.
  - `language.py` `Qwen2Model.__call__` returns `self.norm(h)`: this equals transformers `hidden_states[-1]` (the last hidden state after the final RMSNorm), which is what K6 uses. Call the inner model (`language_model.model`), not `LanguageModel.__call__`, which applies `lm_head` and returns logits.
  - Text-only M-RoPE: position ids are the same `arange` on all three axes; check how `LanguageModel` builds them (`get_rope_index`) and pass equivalent ids explicitly.
  - Only the language weights are needed; `vision.py` is not used for T2V.
- K6 prompt path (`_get_prompt_embeds`, 263–307): `_PROMPT_TEMPLATE.format(prompt)`, tokenizer with `padding="max_length"`, `max_length = max_sequence_length + _QWEN_CROP_START`; embeddings `hidden_states[-1][:, _QWEN_CROP_START:]`; mask is `None` when nothing is padded. CLIP: `pooler_output`, `_CLIP_MAX_LENGTH`, `padding="max_length"`.
- CLIP-L: no maintained MLX port is assumed; porting the CLIP text tower (12 layers, ~123M params) here is acceptable and small.

## Deliverables

- `src/kandinsky6_mlx/text/qwen.py`, `src/kandinsky6_mlx/text/clip.py`, `src/kandinsky6_mlx/text/encode.py` (`encode_prompt(prompt) -> dict` in the cache format of spec 09).
- `tests/parity/test_text_encoders.py` (stage B only; tokenizer tests are stage A).
- `reports/11-text-encoders.json`.

## Implementation notes

- Text-only path: the image tower of Qwen2.5-VL is not needed for T2V; I2V conditioning in K6 goes through the VAE (check in spec 12 whether the prompt path ever receives the image).
- Tokenization must equal the reference token ids exactly (same tokenizer files, same template). Use the `tokenizers`/`transformers` tokenizer files from the checkpoint rather than a reimplementation.
- Padding is on the right and attention is causal, so pad tokens do not change real-token states; still pass the mask so padded positions match the reference.
- Load the encoder, encode, then free it before loading the DiT.

## Tests

- Stage A: token ids and masks equal the reference tokenizer for 5 prompts (Latin, Cyrillic, long, empty negative prompt).
- Stage B, bf16 Qwen: `cosine >= 0.999` per token vs reference fp32 hidden states; CLIP pooled `rel_l2 <= 1e-3`.
- Stage B, quantized Qwen (8/6/4 bit): record cosine; then generate Smoke with each and record PSNR vs the bf16-embedding video. Do not mark quantized variants as supported unless they meet a threshold chosen and justified in the report.

## Validation runs

```bash
uv run pytest tests/parity/test_text_encoders.py -q
uv run python scripts/generate.py --weights weights/lite-distill --prompt "..." --preset smoke --seed 42 --out outputs/11/smoke_prompt.mp4
```

## Acceptance criteria

- [ ] Token ids identical.
- [ ] bf16 hidden-state criterion met; prompt-to-video works.
- [ ] Quantized variants measured and a recommendation recorded.

## Out of scope

Prompt rewriting/enhancement models.

## Stop conditions

The hidden-state layer or template cannot be reproduced through `mlx-vlm` → implement the forward to the needed layer directly and record the deviation.
