"""Check a Hub checkpoint's tensor names against the converter, reading only safetensors headers over HTTP.

For each component: every tensor must be either dropped by the converter or a parameter/persistent buffer of the
reference class built from the checkpoint's own config (on the `meta` device, no weights), and every convolution
rule must hit tensors of the expected rank. Nothing is downloaded beyond headers and config files.

    uv run --group reference python scripts/check_checkpoint_keys.py \
        --repo kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers --revision 4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81
"""

import argparse
import json
import struct
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kandinsky6_mlx.convert.layouts import COMPONENTS, conv_rule, is_dropped
from kandinsky6_mlx.utils.dtype_policy import DtypePolicy


def fetch(url: str, byte_range: str | None = None) -> bytes:
    headers = {"Range": f"bytes={byte_range}"} if byte_range else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        return response.read()


def remote_header(base: str) -> dict:
    (length,) = struct.unpack("<Q", fetch(base, "0-7"))
    header = json.loads(fetch(base, f"8-{8 + length - 1}"))
    header.pop("__metadata__", None)
    return header


def reference_keys(component: str, config: dict) -> set[str]:
    import torch
    from diffusers import AutoencoderKLHunyuanVideo, Kandinsky6Transformer3DModel
    from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE
    from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder

    cls = {
        "transformer": Kandinsky6Transformer3DModel,
        "vae": AutoencoderKLHunyuanVideo,
        "audio_vae": MMAudioVAE,
        "vocoder": MMAudioVocoder,
    }[component]
    kwargs = {k: v for k, v in config.items() if not k.startswith("_")}
    allowed = set(cls.extract_init_dict(kwargs)[0])
    with torch.device("meta"):
        model = cls(**{k: v for k, v in kwargs.items() if k in allowed})
    return set(model.state_dict())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    root = f"https://huggingface.co/{args.repo}/resolve/{args.revision}"
    policy = DtypePolicy()
    ok = True
    summary = {}
    for component in COMPONENTS:
        config = json.loads(fetch(f"{root}/{component}/config.json"))
        header = remote_header(f"{root}/{component}/diffusion_pytorch_model.safetensors")
        expected = reference_keys(component, config)
        kept = {name for name in header if not is_dropped(component, name)}
        dropped = len(header) - len(kept)
        unknown = sorted(kept - expected)
        missing = sorted(expected - kept)
        bad_rank = [
            name for name in kept if (rule := conv_rule(component, name)) and len(header[name]["shape"]) != rule.rank
        ]
        transposed = sum(1 for name in kept if conv_rule(component, name))
        targets = Counter(policy.target(component, name, header[name]["dtype"]) for name in kept)
        stored = Counter(header[name]["dtype"] for name in kept)
        summary[component] = {
            "tensors": len(header),
            "kept": len(kept),
            "dropped": dropped,
            "transposed": transposed,
            "stored_dtypes": dict(stored),
            "converted_dtypes": dict(targets),
            "unknown": len(unknown),
            "missing": len(missing),
            "bad_rank": len(bad_rank),
        }
        print(f"{component}: {json.dumps(summary[component])}")
        for label, names in (("unknown", unknown), ("missing", missing), ("bad rank", bad_rank)):
            if names:
                ok = False
                print(f"  {label}: {names[:10]}{' ...' if len(names) > 10 else ''}")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
