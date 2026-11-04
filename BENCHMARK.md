# Running the benchmark

How to run `scripts/benchmark.py` from a terminal, on your own Mac. It compares the MLX pipeline with the reference
diffusers pipeline on Apple MPS: time and memory per stage, cold and warm.

## Requirements

- Apple Silicon Mac, macOS, [uv](https://docs.astral.sh/uv/).
- Free disk: about 35 GB for all the weights, and at least 8 GB free while it runs (the benchmark stops below that).
- Memory: Smoke needs about 40 GB of the Mac's RAM for the `diffusers` cells. The `mlx` cells with a cache limit fit in
  10 GB, so on a smaller Mac run only those (see [Choosing cells](#choosing-cells)).

## 1. Install

```bash
git clone https://github.com/vladislav-vasilenko/kandinsky6-mlx
cd kandinsky6-mlx
uv sync --extra text --group reference
```

## 2. Get the weights

The MLX weights are in `VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX`. While that repository is private, ask the owner
for access or convert them yourself (second command block).

```bash
# Option A: converted Lite-distill weights (8.5 GB)
uv run hf download VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX --local-dir weights/lite-distill

# Option B: convert from the original release
uv run hf download kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers \
    --revision 4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81 \
    --include "transformer/*" "vae/*" "audio_vae/*" "vocoder/*" --local-dir weights/lite-distill-diffusers
uv run python -m kandinsky6_mlx.convert --checkpoint weights/lite-distill-diffusers --out weights/lite-distill \
    --revision 4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81
```

Text encoders (Qwen2.5-VL 15.5 GB, CLIP-L, tokenizers):

```bash
uv run hf download kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers \
    --revision 4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81 \
    --include "text_encoder/*" "tokenizer/*" "text_encoder_2/*" "tokenizer_2/*" --local-dir weights/lite-distill-text
```

The `diffusers` cells also need the original transformer of each model, in `weights/lite-distill-diffusers/transformer/`
(`config.json` and `diffusion_pytorch_model.safetensors`; downloaded by option B).

For the `base` cells (Lite-5s, 50 steps with guidance, 7.5 GB more):

```bash
uv run hf download kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers \
    --revision 8ec017692475fd8e7908d8e75442bba0737649af \
    --include "transformer/*" "scheduler/*" "model_index.json" --local-dir weights/lite-base-diffusers
uv run python -m kandinsky6_mlx.convert --checkpoint weights/lite-base-diffusers --out weights/lite-base \
    --components transformer --revision 8ec017692475fd8e7908d8e75442bba0737649af

# The media components are shared with Lite-distill; only the audio VAE's scaling factor differs.
ln -sfn ../lite-distill/vae weights/lite-base/vae
ln -sfn ../lite-distill/vocoder weights/lite-base/vocoder
mkdir -p weights/lite-base/audio_vae weights/lite-base/scheduler
ln -sf ../../lite-distill/audio_vae/model.safetensors weights/lite-base/audio_vae/model.safetensors
uv run hf download kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers audio_vae/config.json \
    --revision 8ec017692475fd8e7908d8e75442bba0737649af --local-dir weights/lite-base-config
cp weights/lite-base-config/audio_vae/config.json weights/lite-base/audio_vae/config.json
cp weights/lite-base-diffusers/scheduler/scheduler_config.json weights/lite-base/scheduler/
```

## 3. Prepare the text embeddings

Every cell uses the same cached embeddings:

```bash
QWEN="weights/lite-distill-text/text_encoder/*.safetensors"
uv run --extra text python scripts/encode_prompt.py --qwen-weights $QWEN \
    --prompt-file fixtures/real_inputs/smoke_prompt.txt --out fixtures/real/smoke_text
uv run --extra text python scripts/encode_prompt.py --qwen-weights $QWEN \
    --prompt-file fixtures/real_inputs/baseline_prompt.txt --out fixtures/real/baseline_text
uv run --extra text python scripts/encode_prompt.py --qwen-weights $QWEN --out fixtures/real/negative_text \
    --prompt "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"
```

## 4. Before you run

For numbers worth comparing: plug in the power adapter, close other apps, do not use the Mac heavily during a run, and
keep it awake by prefixing the commands below with `caffeinate -dims` (the whole matrix takes about 18 hours). The chip, memory, macOS, library versions and power source are recorded in `machine.json`.

## 5. Run

Name the output folder after your machine. Groups run in the order given:

```bash
OUT=benchmarks/results/m3max-48gb
RUN="caffeinate -dims uv run --extra text --group reference python scripts/benchmark.py"

$RUN --list                        # all cells
$RUN --matrix text --out $OUT               # text encoding, about 10 min
$RUN --matrix smoke --out $OUT              # main comparison, 10 rounds: about 6 h
$RUN --matrix tradeoffs --out $OUT          # what memory buys, about 1.5 h
$RUN --matrix profiles --out $OUT           # 16/24/32/48 GB budgets at Smoke, about 40 min
$RUN --matrix baseline --out $OUT           # 5 s at 480x864, 3 rounds: about 3 h
$RUN --matrix profiles-baseline --out $OUT  # budgets at Baseline, about 1.5 h
$RUN --matrix baseline-cfg --out $OUT       # Lite-5s at 480x864, about 2.5 h: leave it overnight
```

Each cell prints one line, for example:

```text
round 0: smoke/distill/mlx-bf16: ok, cold 27.7 s, warm 26.9 s, footprint 35.85 GiB, swap +0 MB
```

### Stopping and continuing

Press `Ctrl-C` at any time. Run the same command again: (cell, round) pairs already in `results.jsonl` are skipped, and
the cell that was interrupted runs again. Use `--rerun` to measure everything again.

### Choosing cells

```bash
$RUN --only 'smoke/distill/mlx-bf16$' --out $OUT    # one cell (a regular expression over cell ids)
$RUN --only '^smoke/distill/mlx' --out $OUT         # all MLX cells of a group
$RUN --only 'profiles/distill/smoke-16gb' --out $OUT
$RUN --only 'smoke/distill/mlx-bf16$' --rounds 1 --repeat 1 --out $OUT  # a quick look, not for publishing
```

## 6. Results

| File | Content |
|---|---|
| `$OUT/results.jsonl` | one record per cell: settings, seconds per stage per run, both memory measures, swap, status |
| `$OUT/machine.json` | chip, GPU cores, memory, macOS, power source, library versions |
| `$OUT/logs/*.log` | the worker's stderr for each cell; look here when a status is `exit 1` |

A quick look at a results file:

```bash
python3 - "$OUT/results.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    print(r["cell"]["id"], "|", r["status"], "| footprint", r["footprint_peak_gib"], "GiB")
    for run in r["runs"]:
        print("  ", run["run"], {k: round(v, 1) for k, v in run["seconds"].items()})
PY
```

## 7. Build the report

Statistics, tables and charts, by the method below (it works on partial results too):

```bash
uv run --with matplotlib python scripts/benchmark_report.py $OUT --out docs/benchmarks
```

| File | Content |
|---|---|
| `docs/benchmarks/tables.md` | primary comparisons with intervals and p values; a table for each group of cells |
| `docs/benchmarks/stats.json` | the same comparisons as data, the failed or stopped cells, and the machine |
| `docs/benchmarks/summary.csv` | one row per cell: median, min, max of each measure |
| `docs/benchmarks/*.png`, `*.svg` | effect sizes with intervals, time per stage, cold vs warm, memory per stage, memory vs speed, memory budgets, text encoding |

## What the numbers mean

- **Cold**: the first run in a fresh process, including reading weights and compiling kernels. Each cell has its own
  process, but the OS file cache stays warm between cells. To include a cold disk, run `sudo purge` first.
- **Warm**: later runs in the same process. With `release_dit` (the default, as the study notebook frees the DiT before
  decoding) a warm run reloads the DiT.
- **Footprint**: the process's physical memory as the OS accounts it (`ri_phys_footprint`, the Memory column of Activity
  Monitor, GPU buffers included), sampled every 0.2 s from outside the process. Compare MLX and torch by this.
- **Framework peak**: `mx.get_peak_memory` or `torch.mps.driver_allocated_memory`. MLX's counts active buffers only and
  excludes its cache, so it can be several times lower than the footprint.

## Statistical method

A benchmark number is a measurement with noise, so published comparisons come with their uncertainty. The method was
fixed before the measurements.

**Unit of analysis.** One independent process. Warm runs inside one process are correlated and are not counted as
separate samples: each process contributes one cold total and one warm value, the median of its warm runs.

**Sample size.** The exact paired test used below cannot give p below `2 / 2^n`, so n = 5 can never reach 0.05.

| Cells | Rounds | Used for |
|---|---|---|
| Primary comparisons (`smoke/*/mlx-bf16`, `diffusers-mps-bf16`, and their `-i2v` cells) | 10 | tests and confidence intervals |
| Other Smoke, tradeoff, text and profile cells | 5 | medians and confidence intervals, no tests |
| Baseline | 3 | descriptive: median and range |
| Baseline per budget, Baseline with guidance, cells expected not to fit | 1 | a single measurement, labelled as such, with per-step statistics where possible |

**Controlling bias.**

- Block design: each round runs every cell once, in a random order from a fixed seed, so slow drift and heating do not
  favour one implementation.
- A 20 s pause between cells; a fresh process for each.
- Pairing: all cells of a round use the same noise seed, and seeds differ between rounds. MLX and diffusers are compared
  within a round.
- No outlier removal. Every run stays in `results.jsonl`. Failed and stopped cells are reported, not retried silently.
- Each record has the power source, macOS's CPU speed limit (below 100 means throttling), the swap growth and the load
  average before the run. Rounds on battery, throttled, or with swap growing by more than 500 MB are flagged in the report
  and analysed with and without them. (The swap condition was added during the run, after swap was seen in many Smoke cells
  and before any test was computed.)
- Guard stops are results and are not run again, with one exception, on the M3 Max run: from 22:19 to 22:25 on 2026-11-03
  something outside the benchmark filled the disk, and the guard stopped 14 Smoke rounds. Those rounds ran again at the end,
  with their original seeds but outside their random order (`--retry-stopped-between`). The first records stay in the results.
- The first step of a run is excluded from per-step time, since it includes kernel compilation.

**Primary endpoints.** Fixed in advance: the total time without loading and text encoding, MLX vs diffusers on MPS, cold
and warm, for four configurations (Lite-distill and Lite-5s, text-to-video and image-to-video): 8 tests. Peak OS
footprint is reported for the same cells with the same method. Stage breakdowns and all other cells are exploratory.

**Analysis.**

- Centre and spread: median, min and max over processes.
- Effect: the ratio diffusers / MLX, as the geometric mean of the per-round ratios, with a 95% confidence interval from
  the t distribution on the log ratios. A ratio of 1.25 means diffusers takes 25% longer.
- Test: exact two-sided sign-flip permutation test on the log ratios, with the Holm correction over the 8 primary tests.
- Claim: "faster" or "uses less memory" only when the corrected p is below 0.05, the interval excludes 1, and the
  difference is at least 5%, smaller differences being of no practical use. Otherwise the result is reported as no
  demonstrated difference.

**Limits of the conclusions.** All of this generalizes over repeated runs and noise seeds on this machine and these
prompts. It says nothing about other chips or about other prompt lengths, since one prompt per size is used and DiT time
depends on the token count. One machine is one sample of hardware; results from other Macs are what would widen the
claim. Memory-profile cells emulate a smaller Mac's budget, not its speed.

## Safety stops

A cell is killed and recorded with a `guard:` status, and the benchmark moves on, when its footprint passes 90% of RAM,
swap grows by more than 4 GB, or free disk falls below 8 GB. On macOS swap can fill a nearly full disk within seconds.
A guard status is a result: that configuration does not fit this machine.

Expected example: `smoke/distill/diffusers-mps-bf16-default-watermark`. torch's default
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.7` lets MPS run past the GPU working set, so it is stopped on a 48 GB Mac. The other
`diffusers` cells set the ratio to 1.0.

## Memory profiles on a smaller Mac

The `profiles` cells emulate 16, 24, 32 and 48 GB on one machine with `mx.set_cache_limit` and `mx.set_memory_limit`.
This shows what fits and what saving memory costs, not the speed of a smaller Mac, whose chip usually has fewer GPU
cores and less memory bandwidth. Results from real machines are welcome: run the cells with `--out
benchmarks/results/<your-chip>-<your-memory>`.
