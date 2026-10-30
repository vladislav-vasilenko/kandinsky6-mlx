"""Spec 05 stage B: the MLX DiT in bf16 on the real Lite-distill weights against the float32 reference recording.

The recording comes from `reference/record_dit_real.py` (run on Kaggle) and lives in `fixtures/real/dit_smoke`; the
converted transformer in `weights/lite-distill/transformer` (override with `K6_LITE_TRANSFORMER`). Criterion from
AGENTS.md for every block (run on the float32 inputs) and for the final outputs of the whole model:
`rel_l2 <= 2 * rel_l2(torch bf16 vs torch fp32)` and `cosine >= 0.999`.

Print the metrics for the report with:

    uv run python -m tests.parity.test_dit_real
"""

import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

from kandinsky6_mlx.models.dit.model import Kandinsky6Transformer3DModel
from kandinsky6_mlx.utils.compare import compare
from kandinsky6_mlx.utils.fixtures import load_fixture, select, unflatten

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = Path(os.environ.get("K6_LITE_TRANSFORMER", ROOT / "weights" / "lite-distill" / "transformer"))
FIXTURE = ROOT / "fixtures" / "real" / "dit_smoke"
METRICS = ("rel_l2", "cosine", "rel_max")


def _metrics(actual, reference) -> dict:
    """Same structure as the recorder: one entry per present modality of a fused block, `text` for encoder blocks."""
    if isinstance(reference, tuple):
        return {
            name: {k: compare(a, r)[k] for k in METRICS}
            for name, a, r in zip(("video", "audio"), actual, reference)
            if r is not None
        }
    return {"text": {k: compare(actual, reference)[k] for k in METRICS}}


class _Recording:
    """Stands in for a block of the model, evaluates its output and keeps it."""

    def __init__(self, block, name: str, sink: dict):
        self.block, self.name, self.sink = block, name, sink

    def __call__(self, *args):
        output = self.block(*args)
        mx.eval(output)
        self.sink[self.name] = output
        return output


def stage_b_metrics(weights: Path, fixture: Path, dtype: mx.Dtype = mx.bfloat16) -> dict:
    """MLX metrics against the float32 recording: every block on the recorded float32 inputs, every block inside the
    full forward pass, and the final outputs; plus the torch bf16 metrics stored in the recording."""
    model = Kandinsky6Transformer3DModel.from_converted(weights, dtype)
    tensors, meta = load_fixture(fixture, backend="mlx")
    calls = meta["calls"]["ref"]

    per_block = {}
    for call in calls:
        chain, index = call["module"].rsplit(".", 1)
        args = list(unflatten(call["args"], tensors))
        for i in call["args_to_model_dtype"]:
            if args[i] is not None:
                args[i] = args[i].astype(dtype)
        output = getattr(model, chain)[int(index)](*args)
        mx.eval(output)
        per_block[call["module"]] = _metrics(output, unflatten(call["output"], tensors))

    sink: dict = {}
    for attr in ("video_text_transformer_blocks", "audio_text_transformer_blocks", "visual_transformer_blocks"):
        blocks = getattr(model, attr)
        for i, block in enumerate(blocks):
            blocks[i] = _Recording(block, f"{attr}.{i}", sink)
    inputs = {k: v for k, v in select(tensors, "inputs.ref").items() if "@" not in k}
    outputs = model(**inputs)
    mx.eval(outputs)
    chained = {call["module"]: _metrics(sink[call["module"]], unflatten(call["output"], tensors)) for call in calls}
    references = [tensors[f"outputs.ref.out.{i}"] for i in range(len(outputs))]
    final = {
        name: {k: compare(o, r)[k] for k in METRICS} for name, o, r in zip(("video", "audio"), outputs, references)
    }
    return {"mlx": {"per_block": per_block, "chained": chained, "final": final}, "torch_bf16": meta["torch_bf16"]}


def violations(metrics: dict) -> list[dict]:
    """Entries failing `rel_l2 <= 2 * torch rel_l2` or `cosine >= 0.999`, in forward order (the first is the first
    block exceeding the criterion)."""
    failed = []
    for scope in ("per_block", "final"):
        mlx_scope, torch_scope = metrics["mlx"][scope], metrics["torch_bf16"][scope]
        entries = mlx_scope.items() if scope == "per_block" else [("model", mlx_scope)]
        for module, modalities in entries:
            torch_modalities = torch_scope[module] if scope == "per_block" else torch_scope
            for modality, got in modalities.items():
                limit = 2 * torch_modalities[modality]["rel_l2"]
                if got["rel_l2"] > limit or got["cosine"] < 0.999:
                    failed.append({"scope": scope, "module": module, "modality": modality, "limit": limit, **got})
    return failed


@pytest.mark.real_weights
@pytest.mark.slow
def test_real_lite_distill_bf16_blocks_and_outputs():
    if not (WEIGHTS / "model.safetensors").is_file() or not FIXTURE.with_suffix(".safetensors").is_file():
        pytest.skip(f"needs converted weights in {WEIGHTS} and the recording {FIXTURE}.safetensors")
    failed = violations(stage_b_metrics(WEIGHTS, FIXTURE))
    assert not failed, f"first entry over the criterion: {failed[0]}"


def test_stage_b_harness_on_a_tiny_recording(tmp_path):
    """The whole stage B path on a tiny model: torch recording at Smoke size, conversion, MLX replay.

    In float32 every block and the final outputs must match at `rel_max <= 1e-4`, which checks that the chained block
    inputs, the argument casts and the metric layout of the recorder and of `stage_b_metrics` agree.
    """
    torch = pytest.importorskip("torch")
    from diffusers import Kandinsky6Transformer3DModel as Reference

    from kandinsky6_mlx.convert.convert import convert_component
    from kandinsky6_mlx.utils.fixtures import save_fixture
    from reference import record_dit_real, tiny_configs
    from reference.make_tiny_fixtures import build

    build(Reference, tiny_configs.DIT).save_pretrained(tmp_path / "ckpt" / "transformer")
    convert_component(tmp_path / "ckpt" / "transformer", "transformer", tmp_path / "converted")
    generator = torch.Generator().manual_seed(0)
    text = {
        "inputs.encoder_hidden_states": torch.randn(1, 5, tiny_configs.DIT["in_text_dim"], generator=generator),
        "inputs.pooled_projections": torch.randn(1, tiny_configs.DIT["in_text_dim2"], generator=generator),
        "inputs.encoder_attention_mask": torch.tensor([[True, True, True, True, False]]),
    }
    save_fixture(tmp_path / "text", text, {"fixture": "text"})
    out = tmp_path / "dit_smoke"
    args = ["--checkpoint", str(tmp_path / "ckpt"), "--text-embeds", str(tmp_path / "text"), "--out", str(out)]
    assert record_dit_real.main(args) == 0

    _, meta = load_fixture(out, backend="numpy")
    calls = meta["calls"]["ref"]
    num_blocks = 2 * tiny_configs.DIT["num_text_blocks"] + tiny_configs.DIT["num_visual_blocks"]
    assert len(calls) == num_blocks == len(meta["torch_bf16"]["per_block"])
    second = calls[-1]["args"]["tuple"]
    assert second[0]["tensor"].startswith("outputs.ref.visual_transformer_blocks.0@0")

    metrics = stage_b_metrics(tmp_path / "converted", out, dtype=mx.float32)
    for scope in ("per_block", "chained"):
        for module, modalities in metrics["mlx"][scope].items():
            for modality, values in modalities.items():
                assert values["rel_max"] <= 1e-4, (scope, module, modality, values)
    for modality, values in metrics["mlx"]["final"].items():
        assert values["rel_max"] <= 1e-4, (modality, values)
    assert violations(metrics) == []  # float32 MLX is far inside twice the torch bf16 drift


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    result = stage_b_metrics(WEIGHTS, FIXTURE)
    result["violations"] = violations(result)
    print(json.dumps(result, indent=2))
