"""Encode a prompt into Kandinsky 6 text embeddings on the Mac: Qwen2.5-VL in MLX, CLIP-L pooled output in torch.

Writes the cache format of the MLX pipeline (`inputs.encoder_hidden_states` of the real tokens, no mask, and
`inputs.pooled_projections`). CLIP still runs in torch (CPU, float32; about 123M parameters) until spec 11 ports it, so
this script needs the `reference` group and the `text` extra:

    uv run --extra text --group reference python scripts/encode_prompt.py \\
        --qwen-weights ~/ComfyUI-Shared/models/text_encoders/qwen_2.5_vl_7b.safetensors \\
        --text-root weights/lite-distill-text --prompt-file fixtures/real_inputs/smoke_prompt.txt \\
        --out fixtures/real/smoke_text

`--text-root` holds the checkpoint's `text_encoder/config.json`, `tokenizer/`, `text_encoder_2/` and `tokenizer_2/`.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from kandinsky6_mlx.text.encode import clip_token_ids, encode_qwen
from kandinsky6_mlx.text.qwen import QwenTextEncoder


def clip_pooled(text_root: Path, prompt: str) -> np.ndarray:
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(text_root / "tokenizer_2")
    model = CLIPTextModel.from_pretrained(text_root / "text_encoder_2", dtype=torch.float32).eval()
    ids = clip_token_ids(tokenizer, prompt)
    with torch.no_grad():
        pooled = model(**{k: torch.from_numpy(v) for k, v in ids.items()})["pooler_output"]
    return pooled.float().numpy()


def main(argv: list[str] | None = None) -> int:
    from safetensors.numpy import save_file
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qwen-weights", type=Path, nargs="+", required=True, help="safetensors with model.* names")
    parser.add_argument("--text-root", type=Path, required=True)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    text = args.prompt if args.prompt is not None else args.prompt_file.read_text()
    started = time.perf_counter()
    encoder = QwenTextEncoder.from_files(args.text_root / "text_encoder" / "config.json", args.qwen_weights)
    load_seconds = time.perf_counter() - started
    tokenizer = AutoTokenizer.from_pretrained(args.text_root / "tokenizer")
    mx.reset_peak_memory()
    started = time.perf_counter()
    embeds = encode_qwen(encoder, tokenizer, text)
    mx.eval(embeds)
    qwen_seconds = time.perf_counter() - started
    peak = mx.get_peak_memory() / 2**30
    del encoder
    mx.clear_cache()
    started = time.perf_counter()
    pooled = clip_pooled(args.text_root, text)
    clip_seconds = time.perf_counter() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"inputs.encoder_hidden_states": np.array(embeds), "inputs.pooled_projections": pooled},
        str(args.out.with_suffix(".safetensors")),
    )
    meta = {
        "fixture": args.out.name,
        "prompt": text,
        "qwen": {"implementation": "mlx-vlm Qwen2Model, bf16", "weights": [str(p) for p in args.qwen_weights]},
        "clip": "transformers CLIPTextModel, torch CPU float32",
        "tokens": int(embeds.shape[1]),
        "seconds": {
            "qwen_load": round(load_seconds, 1),
            "qwen": round(qwen_seconds, 2),
            "clip": round(clip_seconds, 2),
        },
        "qwen_peak_memory_gib": round(peak, 2),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "prompt"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
