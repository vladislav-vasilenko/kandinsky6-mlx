"""Record the float32 reference of the Lite-distill video VAE decode at Smoke size for spec 08 stage B (Kaggle CPU).

Input attached to the kernel: the output of `applehawk/k6-lite-diffusers-weights-media` (the released `vae/`, revision
4c487e8b). Internet is needed to clone github.com/vladislav-vasilenko/kandinsky6-mlx at COMMIT and install the
reference pins. torch on CPU peaks near 25 GB of RAM here (too much for the Mac's free disk as swap); a Kaggle CPU
session has 30 GB.

Output in /kaggle/working: `video_smoke.{safetensors,json}` (about 34 MB: the NumPy seed-42 latents and the 25 decoded
frames); copy it to `fixtures/real/` on the Mac.
"""

import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/vladislav-vasilenko/kandinsky6-mlx"
COMMIT = "7fa2f5d8dac4d0413e5ef293253f8ebc8c3348d9"
PINS = ["diffusers==0.42.1", "transformers==5.22.1", "accelerate==1.13.0"]
SRC = Path("/tmp/kandinsky6-mlx")
OUT = Path("/kaggle/working")


def run(*args: str, cwd: Path | None = None) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd=cwd)


def main() -> None:
    started = time.time()
    run("git", "clone", "--quiet", REPO, str(SRC))
    run("git", "checkout", "--quiet", COMMIT, cwd=SRC)
    run(sys.executable, "-m", "pip", "install", "--quiet", *PINS)
    run(
        sys.executable,
        str(SRC / "reference" / "record_video_real.py"),
        "--checkpoint",
        "/kaggle/input",
        "--out",
        str(OUT / "video_smoke"),
        cwd=SRC,
    )
    sizes = {p.name: p.stat().st_size for p in sorted(OUT.glob("*.safetensors"))}
    print(f"done in {time.time() - started:.0f} s: {sizes}", flush=True)


if __name__ == "__main__":
    main()
