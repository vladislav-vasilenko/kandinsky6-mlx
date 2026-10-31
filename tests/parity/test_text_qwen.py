"""MLX Qwen2.5-VL text path (via mlx-vlm) against transformers, and the prompt template against diffusers."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.utils.compare import compare

pytest.importorskip("mlx_vlm")

from kandinsky6_mlx.text.encode import CLIP_MAX_LENGTH, PROMPT_TEMPLATE, QWEN_CROP_START
from kandinsky6_mlx.text.qwen import QwenTextEncoder

ROOT = Path(__file__).resolve().parents[2]


def test_template_constants_match_diffusers():
    pytest.importorskip("diffusers")
    from diffusers.pipelines.kandinsky6 import pipeline_kandinsky6_ti2va as reference

    assert PROMPT_TEMPLATE == reference._PROMPT_TEMPLATE
    assert QWEN_CROP_START == reference._QWEN_CROP_START
    assert CLIP_MAX_LENGTH == reference._CLIP_MAX_LENGTH


def test_last_hidden_state_matches_transformers(tmp_path):
    """A tiny random Qwen2.5-VL saved with the single-file names (`model.*`, `visual.*`, `lm_head`): the MLX encoder's
    output equals transformers' `hidden_states[-1]`, the state Kandinsky 6 uses."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    config = {
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
        "vision_config": {
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
        },
    }
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(**config)).float().eval()
    # transformers 5 nests the decoder under model.language_model; checkpoints store it as model.*
    weights = {
        k.replace("model.language_model.", "model.").replace("model.visual.", "visual."): v.contiguous()
        for k, v in model.state_dict().items()
    }
    save_file(weights, str(tmp_path / "weights.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(config))

    ids = np.random.default_rng(0).integers(0, 1000, (1, 37))
    with torch.no_grad():
        reference = model(input_ids=torch.from_numpy(ids), output_hidden_states=True, return_dict=True).hidden_states[
            -1
        ]
    encoder = QwenTextEncoder.from_files(tmp_path / "config.json", tmp_path / "weights.safetensors", dtype=mx.float32)
    assert compare(encoder(mx.array(ids)), reference)["rel_max"] <= 1e-5


def test_tokenizer_matches_the_reference_processor():
    """For text alone the Qwen2.5-VL processor forwards to its tokenizer; the processor needs torchvision."""
    pytest.importorskip("torchvision")
    from transformers import AutoTokenizer, Qwen2_5_VLProcessor

    from kandinsky6_mlx.text.encode import qwen_token_ids

    root = ROOT / "weights" / "lite-distill-text" / "tokenizer"
    if not root.is_dir():
        pytest.skip(f"needs {root}")
    processor, tokenizer = Qwen2_5_VLProcessor.from_pretrained(root), AutoTokenizer.from_pretrained(root)
    for prompt in ("A cat on a sofa.", "Кошка на диване.", "x " * 2000, ""):
        ids, valid = qwen_token_ids(tokenizer, prompt)
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
