"""Spec 11: MLX text encoders (Qwen2.5-VL via mlx-vlm, CLIP-L port) against transformers 5.22.1.

Stage A runs on tiny random models and the checkpoint's tokenizers; stage B (marker `real_weights`) compares the
MLX encoders with `reference/record_text_encoders.py` (float32 Qwen layer by layer, float32 CLIP) on real weights.
"""

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.utils.compare import compare

pytest.importorskip("mlx_vlm")

from kandinsky6_mlx.text.clip import CLIPTextModel
from kandinsky6_mlx.text.encode import CLIP_MAX_LENGTH, PROMPT_TEMPLATE, QWEN_CROP_START

ROOT = Path(__file__).resolve().parents[2]
TEXT_ROOT = Path(os.environ.get("K6_TEXT_ROOT", ROOT / "weights" / "lite-distill-text"))
QWEN_WEIGHTS = Path(
    os.environ.get("K6_QWEN_WEIGHTS", Path.home() / "ComfyUI-Shared/models/text_encoders/qwen_2.5_vl_7b.safetensors")
)
TEXT_REFERENCE = ROOT / "fixtures" / "real" / "text_reference"
PROMPTS = (
    "A cat on a sofa.",
    "Кошка на диване, мягкий свет.",
    "x " * 2000,  # truncated to 1024 + 129 tokens
    "",  # the empty negative prompt of the CFG path (spec 13)
    "Mixed: 猫 and <AUDCAP>purring<ENDAUDCAP>",
)


def tiny_qwen_config() -> dict:
    return {
        "model_type": "qwen2_5_vl",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 1000,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 4096,
        "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]},
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "vision_config": {
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
        },
    }


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory):
    """A tiny random Qwen2.5-VL saved with the checkpoint's names (`model.*`, `visual.*`, `lm_head.*`)."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    root = tmp_path_factory.mktemp("tiny_qwen")
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(**tiny_qwen_config())).float().eval()
    weights = {
        k.replace("model.language_model.", "model.").replace("model.visual.", "visual."): v.contiguous()
        for k, v in model.state_dict().items()
    }
    save_file(weights, str(root / "weights.safetensors"))
    (root / "config.json").write_text(json.dumps(tiny_qwen_config()))
    return torch, model, root


def test_streaming_reference_equals_the_pipeline_call(tiny_qwen):
    """record_text_encoders.StreamingQwen on the real tokens equals what the pipeline computes: the full
    `Qwen2_5_VLForConditionalGeneration` on the right-padded input with its mask, `hidden_states[-1]`."""
    torch, model, root = tiny_qwen
    from reference.record_text_encoders import StreamingQwen

    ids = torch.from_numpy(np.random.default_rng(0).integers(3, 1000, (1, 40)))
    padded = torch.cat([ids, torch.full((1, 9), 2)], dim=1)
    mask = torch.cat([torch.ones(1, 40, dtype=torch.long), torch.zeros(1, 9, dtype=torch.long)], dim=1)
    with torch.no_grad():
        out = model(input_ids=padded, attention_mask=mask, output_hidden_states=True, return_dict=True)
    reference = out["hidden_states"][-1][:, :40]
    streamed = StreamingQwen(root, [root / "weights.safetensors"])(ids)
    assert compare(streamed, reference)["rel_max"] <= 1e-5


def test_mlx_qwen_equals_the_pipeline_call(tiny_qwen):
    torch, model, root = tiny_qwen
    from kandinsky6_mlx.text.qwen import QwenTextEncoder

    ids = np.random.default_rng(1).integers(3, 1000, (1, 37))
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(ids), output_hidden_states=True, return_dict=True)
    encoder = QwenTextEncoder.from_files(root / "config.json", root / "weights.safetensors", dtype=mx.float32)
    assert compare(encoder(mx.array(ids)), out["hidden_states"][-1])["rel_max"] <= 1e-5


def tiny_clip(tmp_path, torch):
    from transformers import CLIPTextConfig
    from transformers import CLIPTextModel as Reference

    config = CLIPTextConfig(
        vocab_size=200,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
        hidden_act="quick_gelu",
        eos_token_id=2,
    )
    torch.manual_seed(0)
    reference = Reference(config).float().eval()
    reference.save_pretrained(tmp_path / "clip")
    return reference, CLIPTextModel.from_pretrained(tmp_path / "clip")


def test_clip_pooled_matches_transformers_with_padding(tmp_path):
    """Right padding with the largest id (CLIP pads with <|endoftext|>) and the reference's padding mask: the pooled
    state at the first largest id matches, and so do all states up to it."""
    torch = pytest.importorskip("torch")
    reference, model = tiny_clip(tmp_path, torch)
    ids = torch.tensor([[0, 5, 7, 9, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199, 199]])
    mask = (torch.arange(16) <= 4).long()[None]
    with torch.no_grad():
        out = reference(input_ids=ids, attention_mask=mask)
    hidden, pooled = model(mx.array(ids.numpy()))
    assert compare(pooled, out["pooler_output"])["rel_max"] <= 1e-5
    assert compare(hidden[:, :5], out["last_hidden_state"][:, :5])["rel_max"] <= 1e-5


def test_template_constants_match_diffusers():
    pytest.importorskip("diffusers")
    from diffusers.pipelines.kandinsky6 import pipeline_kandinsky6_ti2va as reference

    assert PROMPT_TEMPLATE == reference._PROMPT_TEMPLATE
    assert QWEN_CROP_START == reference._QWEN_CROP_START
    assert CLIP_MAX_LENGTH == reference._CLIP_MAX_LENGTH


@pytest.mark.parametrize("prompt", PROMPTS, ids=["latin", "cyrillic", "long-truncated", "empty", "mixed"])
def test_token_ids_equal_the_reference_tokenizers(prompt):
    """Qwen: AutoTokenizer equals the pipeline's Qwen2_5_VLProcessor (ids and mask). CLIP: same tokenizer call."""
    pytest.importorskip("torchvision")
    from transformers import AutoTokenizer, CLIPTokenizer, Qwen2_5_VLProcessor

    from kandinsky6_mlx.text.encode import clip_token_ids, qwen_token_ids

    if not (TEXT_ROOT / "tokenizer").is_dir():
        pytest.skip(f"needs {TEXT_ROOT / 'tokenizer'}")
    processor = Qwen2_5_VLProcessor.from_pretrained(TEXT_ROOT / "tokenizer")
    ids, valid = qwen_token_ids(AutoTokenizer.from_pretrained(TEXT_ROOT / "tokenizer"), prompt)
    reference = processor(
        text=[PROMPT_TEMPLATE.format(prompt)],
        images=None,
        videos=None,
        max_length=1024 + QWEN_CROP_START,
        truncation=True,
        return_tensors="np",
        padding="max_length",
    )
    np.testing.assert_array_equal(ids, reference["input_ids"])
    assert valid == int(reference["attention_mask"].sum())
    clip_tokenizer = CLIPTokenizer.from_pretrained(TEXT_ROOT / "tokenizer_2")
    expected = clip_tokenizer(
        [prompt], max_length=77, truncation=True, add_special_tokens=True, padding="max_length", return_tensors="np"
    )
    np.testing.assert_array_equal(clip_token_ids(clip_tokenizer, prompt)["input_ids"], expected["input_ids"])


# Stage B thresholds, derived in report 11 from the five reference prompts:
# - float32 MLX Qwen (the port check): worst rel_l2 2.3e-5 -> 10x; worst token cosine 1.0 -> 0.9999.
# - bf16: AGENTS.md (rel_l2 <= 2x the torch bf16 drift per prompt); per-token cosine >= 0.998, because the spec's
#   0.999 is missed by torch bf16 itself (worst token 0.99844).
# - CLIP pooled: the spec's rel_l2 <= 1e-3.
FP32_MAX_REL_L2 = 2.3e-4
FP32_MIN_TOKEN_COSINE = 0.9999
BF16_MIN_TOKEN_COSINE = 0.998


def token_cosines(actual, reference) -> np.ndarray:
    a, r = np.asarray(actual, np.float64)[0], np.asarray(reference, np.float64)[0]
    return (a * r).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(r, axis=-1))


def stage_b_metrics(dtype=mx.bfloat16, bits=None) -> list[dict]:
    from transformers import AutoTokenizer

    from kandinsky6_mlx.text.encode import encode_qwen
    from kandinsky6_mlx.text.qwen import QwenTextEncoder
    from kandinsky6_mlx.utils.fixtures import load_fixture

    reference, meta = load_fixture(TEXT_REFERENCE, backend="numpy")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ROOT / "tokenizer")
    encoder = QwenTextEncoder.from_files(TEXT_ROOT / "text_encoder" / "config.json", QWEN_WEIGHTS, dtype, bits=bits)
    results = []
    for i, prompt in enumerate(meta["prompts"]):
        got = encode_qwen(encoder, tokenizer, prompt)
        ref = reference[f"outputs.prompt_{i}.qwen"]
        entry = {"rel_l2": compare(got, ref)["rel_l2"], "min_token_cosine": float(token_cosines(got, ref).min())}
        drift = reference.get(f"outputs.prompt_{i}.qwen_torch_bf16")
        if drift is not None:
            entry["torch_bf16_rel_l2"] = compare(drift, ref)["rel_l2"]
            entry["torch_bf16_min_token_cosine"] = float(token_cosines(drift, ref).min())
        results.append(entry)
    return results


def _require_real():
    files = [
        QWEN_WEIGHTS,
        TEXT_REFERENCE.with_suffix(".safetensors"),
        TEXT_ROOT / "text_encoder_2" / "model.safetensors",
    ]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        pytest.skip(f"needs {missing}")


@pytest.mark.real_weights
@pytest.mark.slow
def test_real_qwen_float32_matches_the_reference():
    _require_real()
    for i, m in enumerate(stage_b_metrics(mx.float32)):
        assert m["rel_l2"] <= FP32_MAX_REL_L2 and m["min_token_cosine"] >= FP32_MIN_TOKEN_COSINE, (i, m)


@pytest.mark.real_weights
@pytest.mark.slow
def test_real_qwen_bf16_within_torch_bf16_drift():
    _require_real()
    for i, m in enumerate(stage_b_metrics(mx.bfloat16)):
        assert m["rel_l2"] <= 2 * m["torch_bf16_rel_l2"], (i, m)
        assert m["min_token_cosine"] >= BF16_MIN_TOKEN_COSINE, (i, m)


@pytest.mark.real_weights
def test_real_clip_pooled_matches_the_reference():
    _require_real()
    from transformers import CLIPTokenizer

    from kandinsky6_mlx.text.encode import clip_pooled
    from kandinsky6_mlx.utils.fixtures import load_fixture

    reference, meta = load_fixture(TEXT_REFERENCE, backend="numpy")
    model = CLIPTextModel.from_pretrained(TEXT_ROOT / "text_encoder_2")
    tokenizer = CLIPTokenizer.from_pretrained(TEXT_ROOT / "tokenizer_2")
    for i, prompt in enumerate(meta["prompts"]):
        got = clip_pooled(model, tokenizer, prompt)
        assert compare(got, reference[f"outputs.prompt_{i}.pooled"])["rel_l2"] <= 1e-3, i
