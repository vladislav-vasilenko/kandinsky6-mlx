"""Encode a prompt into Kandinsky 6 text embeddings on the Mac (Qwen2.5-VL via mlx-vlm and CLIP-L, both in MLX).

Writes the cache format of the MLX pipeline (`inputs.encoder_hidden_states` of the real tokens, no mask, and
`inputs.pooled_projections`), for reuse across runs. `scripts/generate.py --prompt` encodes on the fly instead.

    uv run --extra text python scripts/encode_prompt.py --text-root weights/lite-distill-text \\
        --prompt-file fixtures/real_inputs/smoke_prompt.txt --out fixtures/real/smoke_text

`--text-root` holds the checkpoint's `text_encoder/` (config and shards), `tokenizer/`, `text_encoder_2/` and
`tokenizer_2/`; `--qwen-weights` points at other Qwen safetensors with the same names (e.g. a single-file copy).
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from kandinsky6_mlx.text.encode import TextEncoders

QWEN_DTYPES = {"bfloat16": mx.bfloat16, "float32": mx.float32}


def add_text_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text-root", type=Path, default=Path("weights/lite-distill-text"))
    parser.add_argument("--qwen-weights", type=Path, nargs="+", help="Qwen safetensors (default: text_encoder/*)")
    parser.add_argument("--qwen-dtype", choices=sorted(QWEN_DTYPES), default="bfloat16")
    parser.add_argument("--qwen-bits", type=int, choices=(8, 6, 4), help="quantize Qwen (see report 11)")


def load_encoders(args: argparse.Namespace) -> TextEncoders:
    return TextEncoders.from_files(args.text_root, args.qwen_weights, QWEN_DTYPES[args.qwen_dtype], args.qwen_bits)


def main(argv: list[str] | None = None) -> int:
    from safetensors.numpy import save_file

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_text_arguments(parser)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    text = args.prompt if args.prompt is not None else args.prompt_file.read_text()
    started = time.perf_counter()
    encoders = load_encoders(args)
    load_seconds = time.perf_counter() - started
    mx.reset_peak_memory()
    started = time.perf_counter()
    embeds = encoders.encode_prompt(text)
    encode_seconds = time.perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # safetensors.numpy writes the raw buffer: arrays must be C-contiguous.
    save_file(
        {f"inputs.{k}": np.ascontiguousarray(np.array(v)) for k, v in embeds.items()},
        str(args.out.with_suffix(".safetensors")),
    )
    meta = {
        "fixture": args.out.name,
        "prompt": text,
        "qwen": {
            "implementation": "mlx-vlm Qwen2Model",
            "dtype": args.qwen_dtype,
            "bits": args.qwen_bits,
            "weights": [str(p) for p in args.qwen_weights] if args.qwen_weights else "text_encoder/*",
        },
        "clip": "kandinsky6_mlx.text.clip.CLIPTextModel, float32",
        "tokens": int(embeds["encoder_hidden_states"].shape[1]),
        "seconds": {"load": round(load_seconds, 1), "encode": round(encode_seconds, 2)},
        "encode_peak_memory_gib": round(mx.get_peak_memory() / 2**30, 2),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "prompt"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
