"""Convert diffusers component directories into MLX-ready safetensors files.

Needs only the standard library and NumPy: safetensors files are read and written directly (8-byte header length,
JSON header, raw little-endian data), so BF16 tensors pass through as raw 16-bit patterns and the converter runs on
any CPU machine, including Kaggle. Tensors are streamed one at a time, so memory stays near the largest tensor.
"""

import argparse
import hashlib
import json
import re
import shutil
import struct
import time
from pathlib import Path

import numpy as np

from kandinsky6_mlx.convert.layouts import COMPONENTS, conv_rule, is_dropped
from kandinsky6_mlx.utils.dtype_policy import DtypePolicy

NUMPY_DTYPES = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),  # raw bit patterns
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}
OUTPUT_NAME = "model.safetensors"


def read_header(path: Path) -> tuple[dict, int]:
    """Return the safetensors header (without `__metadata__`) and the byte offset where data starts."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


def read_tensor(path: Path, info: dict, data_start: int) -> np.ndarray:
    start, end = info["data_offsets"]
    dtype = NUMPY_DTYPES[info["dtype"]]
    array = np.fromfile(path, dtype=dtype, count=(end - start) // dtype.itemsize, offset=data_start + start)
    return array.reshape(info["shape"])


def cast(array: np.ndarray, source: str, target: str) -> np.ndarray:
    if source == target:
        return array
    if source == "BF16":
        array = (array.astype(np.uint32) << 16).view(np.float32)
        source = "F32"
    if target == "F32":
        return array.astype(np.float32)
    raise ValueError(f"unsupported cast {source} -> {target}")


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def source_files(component_dir: Path) -> list[Path]:
    files = sorted(component_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {component_dir}")
    return files


def infer_revision(checkpoint: Path) -> str | None:
    manifest = checkpoint / "weights-manifest.json"
    if manifest.is_file():
        return json.loads(manifest.read_text()).get("revision")
    for part in checkpoint.resolve().parts[::-1]:
        if re.fullmatch(r"[0-9a-f]{40}", part):
            return part
    return None


def convert_component(
    src_dir: str | Path,
    component: str,
    dst_dir: str | Path,
    dtype_policy: DtypePolicy | None = None,
    revision: str | None = None,
) -> dict:
    """Convert one component directory. Writes `model.safetensors`, `config.json` and `conversion.json`."""
    if component not in COMPONENTS:
        raise ValueError(f"unknown component {component!r}; expected one of {COMPONENTS}")
    policy = dtype_policy or DtypePolicy()
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    plan = []  # (name, path, info, data_start, target dtype, rule)
    dropped = []
    for path in source_files(src_dir):
        header, data_start = read_header(path)
        for name in sorted(header):
            info = header[name]
            if is_dropped(component, name):
                dropped.append(name)
                continue
            rule = conv_rule(component, name)
            if rule is not None and len(info["shape"]) != rule.rank:
                raise ValueError(f"{component}/{name}: expected rank {rule.rank}, got shape {info['shape']}")
            plan.append((name, path, info, data_start, policy.target(component, name, info["dtype"]), rule))
    names = [entry[0] for entry in plan]
    if len(set(names)) != len(names):
        raise ValueError(f"{component}: duplicate tensor names across shards")
    plan.sort(key=lambda entry: entry[0])

    header, offset = {}, 0
    for name, _, info, _, target, rule in plan:
        shape = list(info["shape"])
        if rule is not None:
            shape = [shape[axis] for axis in rule.axes()]
        size = int(np.prod(shape, dtype=np.int64)) * NUMPY_DTYPES[target].itemsize
        header[name] = {"dtype": target, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)

    out_path = dst_dir / OUTPUT_NAME
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for name, path, info, data_start, target, rule in plan:
            array = cast(read_tensor(path, info, data_start), info["dtype"], target)
            if rule is not None:
                array = array.transpose(rule.axes())
            data = np.ascontiguousarray(array).tobytes()
            expected = header[name]["data_offsets"]
            if len(data) != expected[1] - expected[0]:
                raise AssertionError(f"{name}: wrote {len(data)} bytes, expected {expected[1] - expected[0]}")
            out.write(data)

    config = src_dir / "config.json"
    if config.is_file():
        shutil.copyfile(config, dst_dir / "config.json")

    dtypes: dict[str, int] = {}
    for entry in header.values():
        dtypes[entry["dtype"]] = dtypes.get(entry["dtype"], 0) + 1
    report = {
        "component": component,
        "source_dir": str(src_dir),
        "source_revision": revision,
        "source_files": {path.name: sha256(path) for path in source_files(src_dir)},
        "output_file": OUTPUT_NAME,
        "output_sha256": sha256(out_path),
        "output_bytes": out_path.stat().st_size,
        "dtype_policy": policy.mode,
        "tensor_count": len(header),
        "dtype_counts": dict(sorted(dtypes.items())),
        "dropped_keys": len(dropped),
        "dropped_prefixes": sorted({name.split(".", 1)[0] + "." for name in dropped}),
        "transposed_keys": [name for name, *_, rule in plan if rule is not None],
        "seconds": round(time.time() - started, 2),
    }
    (dst_dir / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def convert_checkpoint(
    checkpoint: str | Path,
    out: str | Path,
    components: tuple[str, ...] = COMPONENTS,
    dtype_policy: DtypePolicy | None = None,
    revision: str | None = None,
) -> dict[str, dict]:
    checkpoint, out = Path(checkpoint), Path(out)
    revision = revision or infer_revision(checkpoint)
    reports = {}
    for component in components:
        reports[component] = convert_component(
            checkpoint / component, component, out / component, dtype_policy, revision
        )
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a Kandinsky 6 diffusers checkpoint into MLX weights.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="directory with transformer/, vae/, ...")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=list(COMPONENTS))
    parser.add_argument("--dtype-policy", choices=("checkpoint", "float32"), default="checkpoint")
    parser.add_argument("--revision", default=None, help="source revision; inferred when possible")
    args = parser.parse_args(argv)
    reports = convert_checkpoint(
        args.checkpoint, args.out, tuple(args.components), DtypePolicy(args.dtype_policy), args.revision
    )
    for component, report in reports.items():
        print(
            f"{component}: {report['tensor_count']} tensors {report['dtype_counts']}, "
            f"dropped {report['dropped_keys']}, transposed {len(report['transposed_keys'])}, "
            f"{report['output_bytes'] / 2**30:.2f} GiB, {report['seconds']} s"
        )
    return 0
