"""Record the float32 reference text embeddings of Kandinsky 6 prompts for spec 11 stage B.

Qwen2.5-VL-7B in float32 needs about 30 GB, which a 48 GB Mac cannot hold safely, so the reference runs it layer by
layer: the modules of `transformers.Qwen2_5_VLTextModel` (5.22.1) are built on the meta device and each decoder layer
is materialized on the CPU, loaded (bf16 weights upcast to float32), applied and freed in turn. The loop repeats
`Qwen2_5_VLTextModel.forward` for unpacked text (`arange` positions on the three M-RoPE axes, `create_causal_mask`, the
rotary embedding, the layers, the final norm), and `tests/parity/test_text_encoders.py` checks it against the full
`Qwen2_5_VLForConditionalGeneration` call the pipeline makes (`hidden_states[-1]`, padded input with its mask).

Each prompt is tokenized exactly as `Kandinsky6TI2VAPipeline._get_prompt_embeds` does (the checkpoint's
`Qwen2_5_VLProcessor`, template, `padding="max_length"` to 1024 + 129). Only the real tokens are run: right padding
cannot change them under causal attention. CLIP-L runs whole in float32 (`pooler_output`).

Output fixture, per prompt `i`: `inputs.prompt_{i}.input_ids` (padded, as tokenized), `outputs.prompt_{i}.qwen` (the
real tokens after the 129 template tokens, `(1, n, 3584)`), `outputs.prompt_{i}.pooled` (`(1, 768)`), and in the
metadata the prompts and real-token counts.

    uv run --group reference python reference/record_text_encoders.py \\
        --qwen-weights ~/ComfyUI-Shared/models/text_encoders/qwen_2.5_vl_7b.safetensors \\
        --text-root weights/lite-distill-text --out fixtures/real/text_reference
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kandinsky6_mlx.text.encode import CLIP_MAX_LENGTH, MAX_SEQUENCE_LENGTH, PROMPT_TEMPLATE, QWEN_CROP_START
from kandinsky6_mlx.utils.fixtures import save_fixture

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = [
    (ROOT / "fixtures" / "real_inputs" / "smoke_prompt.txt").read_text(),
    (ROOT / "fixtures" / "real_inputs" / "baseline_prompt.txt").read_text(),
    "A red fox runs through fresh snow at dawn, slow motion. <AUDCAP>Crunching snow, distant crows.<ENDAUDCAP>",
    "Рыжая лиса бежит по свежему снегу на рассвете, замедленная съёмка.",
    "",
]


class StreamingQwen:
    """`Qwen2_5_VLTextModel.forward` with one decoder layer resident at a time, in `dtype` (float32 for the reference;
    bfloat16 gives the torch bf16 drift as `from_pretrained(dtype=torch.bfloat16)` computes it)."""

    def __init__(
        self, config_dir: Path, weight_files: list[Path], dtype: torch.dtype = torch.float32, device: str = "cpu"
    ):
        self.dtype, self.device = dtype, device
        from safetensors import safe_open
        from transformers import Qwen2_5_VLConfig
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLRotaryEmbedding,
            Qwen2_5_VLTextModel,
        )

        self.config = Qwen2_5_VLConfig.from_pretrained(config_dir).text_config
        self.config._attn_implementation = "sdpa"  # from_pretrained's default, which the pipeline uses
        if set(self.config.layer_types) != {"full_attention"}:
            raise NotImplementedError(f"layer types {set(self.config.layer_types)}")
        with torch.device("meta"):
            self.model = Qwen2_5_VLTextModel(self.config)
        # inv_freq stays float32, as in from_pretrained
        self.rotary = Qwen2_5_VLRotaryEmbedding(config=self.config).to(device)
        self.files = [safe_open(str(f), framework="pt") for f in weight_files]
        self.where = {name: handle for handle in self.files for name in handle.keys()}  # noqa: SIM118 - safe_open is not a dict
        self._materialize(self.model.embed_tokens, "model.embed_tokens.")
        self._materialize(self.model.norm, "model.norm.")

    def _materialize(self, module: torch.nn.Module, prefix: str) -> None:
        module.to_empty(device=self.device).to(self.dtype)
        state = {}
        for name in module.state_dict():
            key = prefix + name
            state[name] = self.where[key].get_tensor(key).to(self.dtype).to(self.device)
        module.load_state_dict(state, strict=True)

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """Last hidden states for one `(1, L)` id tensor or a list of them; each layer is loaded once for all."""
        from transformers.masking_utils import create_causal_mask

        single = isinstance(input_ids, torch.Tensor)
        batch = [input_ids] if single else list(input_ids)
        hidden = [self.model.embed_tokens(ids.to(self.device)) for ids in batch]
        masks, rope = [], []
        for h in hidden:
            positions = torch.arange(h.shape[1], device=self.device).view(1, 1, -1).expand(3, h.shape[0], -1)
            masks.append(
                create_causal_mask(
                    config=self.config, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=None
                )
            )
            rope.append(self.rotary(h, positions))
        for i, layer in enumerate(self.model.layers):
            self._materialize(layer, f"model.layers.{i}.")
            for k in range(len(hidden)):
                out = layer(hidden[k], attention_mask=masks[k], position_embeddings=rope[k], position_ids=None)
                hidden[k] = out[0] if isinstance(out, tuple) else out
            layer.to("meta")
            gc.collect()
        out = [self.model.norm(h).cpu() for h in hidden]
        return out[0] if single else out


def tokenize(processor, prompt: str) -> tuple[torch.Tensor, int]:
    inputs = processor(
        text=[PROMPT_TEMPLATE.format(prompt)],
        images=None,
        videos=None,
        max_length=MAX_SEQUENCE_LENGTH + QWEN_CROP_START,
        truncation=True,
        return_tensors="pt",
        padding="max_length",
    )
    return inputs["input_ids"], int(inputs["attention_mask"].sum())


def main(argv: list[str] | None = None) -> int:
    from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLProcessor

    from reference.recorder import library_versions

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qwen-weights", type=Path, nargs="+", required=True, help="safetensors with model.* names")
    parser.add_argument("--text-root", type=Path, required=True, help="text_encoder/, tokenizer/, *_2/ folders")
    parser.add_argument("--prompts-file", type=Path, help="one prompt per line (default: the built-in set)")
    parser.add_argument(
        "--drift-device", help="also record torch bf16 Qwen states on this device (mps or cuda): the tolerance basis"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    prompts = args.prompts_file.read_text().split("\n") if args.prompts_file else DEFAULT_PROMPTS
    processor = Qwen2_5_VLProcessor.from_pretrained(args.text_root / "tokenizer")
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.text_root / "tokenizer_2")
    clip = CLIPTextModel.from_pretrained(args.text_root / "text_encoder_2", dtype=torch.float32).eval()
    qwen = StreamingQwen(args.text_root / "text_encoder", args.qwen_weights)

    tensors, counts = {}, []
    started = time.time()
    tokenized = [tokenize(processor, prompt) for prompt in prompts]
    hidden_states = qwen([ids[:, :valid] for ids, valid in tokenized])
    print(f"Qwen float32, {len(prompts)} prompts: {time.time() - started:.0f} s", flush=True)
    drift = None
    if args.drift_device:
        # bf16 on a CPU is not representative (see report 11); the drift is recorded on a GPU.
        bf16 = StreamingQwen(args.text_root / "text_encoder", args.qwen_weights, torch.bfloat16, args.drift_device)
        drift = [h.float() for h in bf16([ids[:, :valid] for ids, valid in tokenized])]
        del bf16
    for i, (prompt, (ids, valid), hidden) in enumerate(zip(prompts, tokenized, hidden_states)):
        clip_ids = clip_tokenizer(
            [prompt],
            max_length=CLIP_MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            pooled = clip(**clip_ids)["pooler_output"]
        tensors[f"inputs.prompt_{i}.input_ids"] = ids
        tensors[f"outputs.prompt_{i}.qwen"] = hidden[:, QWEN_CROP_START:]
        tensors[f"outputs.prompt_{i}.pooled"] = pooled
        if drift is not None:
            tensors[f"outputs.prompt_{i}.qwen_torch_bf16"] = drift[i][:, QWEN_CROP_START:]
        counts.append(valid)
    save_fixture(
        args.out,
        tensors,
        {
            "fixture": args.out.name,
            "reference": "transformers Qwen2_5_VLTextModel (layer by layer, float32) + CLIPTextModel (float32)",
            "qwen_weights": [str(p) for p in args.qwen_weights],
            "prompts": prompts,
            "real_tokens": counts,
            "seconds": round(time.time() - started, 1),
            "torch_bf16_drift_device": args.drift_device,
            "attention": "sdpa (from_pretrained default, as the pipeline loads Qwen)",
            "versions": library_versions() | {"torch": torch.__version__},
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
