# Hardware, disk and environment requirements

Apple Silicon has unified memory: there is no separate VRAM. "Memory" below means the Mac's RAM, shared by CPU and GPU. Numbers are estimates from checkpoint file sizes; spec 14 replaces them with measurements.

## Software

| Item | Version |
|---|---|
| macOS | 14+ on Apple Silicon (M1 or newer; M2+ for native bf16) |
| Python | 3.11+ via `uv` |
| MLX | `mlx>=0.25` |
| Reference group (optional) | `torch>=2.4`, `diffusers==0.42.1`, `transformers==5.22.1`, `accelerate==1.13.0` |
| Text encoder (spec 11) | `mlx-vlm` (pinned version) |
| Video export | `av` (PyAV, bundles FFmpeg) |

## By phase

| Phase | Specs | Free SSD | Memory | Notes |
|---|---|---|---|---|
| Start, tiny models | 00–08 stage A | **~5 GB** | 16 GB | venv with MLX ~0.3 GB, reference group with PyTorch ~2–3 GB, tiny fixtures < 20 MB. No downloads of weights |
| Weight conversion, Lite-distill | 02 stage B | **0 on the Mac** (Kaggle); ~20 GB if run locally | 16 GB, CPU only | Runs on a Kaggle CPU kernel from the existing weight snapshot (9.4 GiB source + ~8.5 GiB output in `/kaggle/working`). The converter needs only NumPy and safetensors (no MLX, PyTorch or GPU) |
| Download of converted Lite weights | 05–09 stage B | **~8.5 GB** | — | From the private Hugging Face repository the Kaggle kernel uploads to |
| Real-weight parity (DiT, VAEs) | 05–08 stage B | +1–3 GB | 32 GB | Recorded reference activations (Smoke size). Reference fp32 runs are done on Kaggle, not on the Mac |
| First MP4, Lite-distill Smoke | 09 | ~20 GB total | 32 GB recommended | DiT bf16 ~6 GB + activations; text embeddings from cache |
| Prompt → video | 11 | **+16 GB** (bf16 Qwen) or +5 GB (4-bit) | 32 GB (bf16 Qwen) | Qwen loaded, used and freed before the DiT; CLIP 0.46 GB |
| Baseline 512×768, 121 frames | 10 | same | 32 GB+ | Tiled VAE decode |
| Lite base (CFG) | 13 | +~8 GB | 32 GB | Separate DiT weights of the same size |
| Pro-distill | 14 | +~60 GB bf16 (~16 GB 4-bit) | 128 GB bf16, ~48 GB 4-bit | DiT ~30B parameters; feasibility decided in spec 14 |

Recommended total to go through the MVP with prompt input (Lite-distill, converted weights, bf16 Qwen, reference recordings, outputs): **about 45–50 GB free SSD**. Deleting the source safetensors after conversion saves ~9 GB.

## Where each kind of work runs

| Work | Where |
|---|---|
| Tiny-model tests (stage A) | CI on GitHub and the Mac |
| Reference activations on real weights (PyTorch) | Kaggle |
| Weight conversion | Kaggle CPU kernel (or the Mac if disk allows) |
| MLX parity on real weights, generation | Mac only |

## Reference runs

PyTorch fp32 reference with real weights needs ~12 GB for the Lite DiT and ~31 GB for Qwen in fp32. Record reference activations on Kaggle (CPU RAM or 2×T4) with `reference/record_*.py`, then download the fixtures. The Mac only runs MLX and compares.

## This machine (checked 2026-10-31)

Apple M3 Max, 48 GB unified memory: enough for Lite (all stages) and probably Pro-distill 4-bit. Free SSD is **13 GB**: enough for stage A (specs 00–08 on tiny models) and for downloading converted Lite weights (~8.5 GB) once conversion runs on Kaggle; not enough for local conversion (~20 GB) or the MVP with bf16 Qwen (~25 GB of weights in total).
