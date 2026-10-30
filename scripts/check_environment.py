"""Print the environment the port runs in: chip, OS, library versions and free disk space."""

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["mlx", "numpy", "safetensors", "huggingface_hub", "av", "pillow"]
REFERENCE_PACKAGES = ["torch", "diffusers", "transformers", "accelerate"]


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sysctl(key: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    memsize = sysctl("hw.memsize")
    memory_gib = f"{int(memsize) / 2**30:.0f} GiB" if memsize.isdigit() else "unknown"
    print(f"chip: {sysctl('machdep.cpu.brand_string')}")
    print(f"memory: {memory_gib}")
    print(f"macos: {platform.mac_ver()[0] or 'n/a'}")
    print(f"python: {sys.version.split()[0]}")
    for name in PACKAGES:
        print(f"{name}: {version(name) or 'missing'}")
    for name in REFERENCE_PACKAGES:
        print(f"{name}: {version(name) or 'not installed (reference group)'}")

    import mlx.core as mx

    print(f"mlx default device: {mx.default_device()}")
    print(f"free disk: {shutil.disk_usage(ROOT).free / 2**30:.1f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
