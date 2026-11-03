"""Statistics, tables and charts from `scripts/benchmark.py` results, by the method in BENCHMARK.md.

    uv run --with matplotlib python scripts/benchmark_report.py benchmarks/results/m3max-48gb --out docs/benchmarks

Writes `summary.csv` (one row per cell), `stats.json` (the primary comparisons), `tables.md` and the charts (PNG and SVG).
The process is the unit of analysis. A process gives one cold total and one warm value, the median of its warm runs;
totals exclude loading, which is reported on its own. Effects are geometric means of per-round ratios with t intervals
on the log ratios; tests are exact two-sided sign-flip permutation tests with the Holm correction.
"""

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

import numpy as np

# 97.5% points of Student's t by degrees of freedom; no SciPy dependency for a dozen numbers.
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
ALPHA = 0.05
MIN_EFFECT = 1.05
PRIMARY = {
    "Lite-distill T2V": ("smoke/distill/mlx-bf16", "smoke/distill/diffusers-mps-bf16"),
    "Lite-distill I2V": ("smoke/distill/mlx-bf16-i2v", "smoke/distill/diffusers-mps-bf16-i2v"),
    "Lite-5s T2V": ("smoke/base/mlx-bf16", "smoke/base/diffusers-mps-bf16"),
    "Lite-5s I2V": ("smoke/base/mlx-bf16-i2v", "smoke/base/diffusers-mps-bf16-i2v"),
}
STAGES = ("load", "image_encode", "denoise", "video_decode", "audio_decode")
STAGE_LABELS = {
    "load": "Load",
    "image_encode": "Image encode",
    "denoise": "Denoise",
    "video_decode": "Video decode",
    "audio_decode": "Audio decode",
    "encode": "Encode",
}
# Validated categorical slots; color follows the entity everywhere.
IMPL_COLOR = {"mlx": "#2a78d6", "diffusers": "#eb6834", "torch": "#eb6834"}
IMPL_LABEL = {"mlx": "MLX", "diffusers": "diffusers (MPS)", "torch": "torch (MPS)"}
STAGE_COLOR = {
    "load": "#2a78d6",
    "image_encode": "#eb6834",
    "denoise": "#1baf7a",
    "video_decode": "#eda100",
    "audio_decode": "#e87ba4",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


# -------------------------------------------------------------------------------------------------------- statistics


def sign_flip_p(log_ratios) -> float:
    """Exact two-sided sign-flip permutation test of a zero mean. Its minimum is 2 / 2**n."""
    values = np.asarray(log_ratios, float)
    observed = abs(values.mean())
    count = sum(
        abs((values * np.array(signs)).mean()) >= observed - 1e-12
        for signs in itertools.product((1, -1), repeat=len(values))
    )
    return count / 2 ** len(values)


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p values, in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def ratio_effect(numerator: list[float], denominator: list[float]) -> dict:
    """Geometric mean of the paired ratios with a 95% t interval on the log scale."""
    logs = np.log(np.asarray(numerator, float) / np.asarray(denominator, float))
    n, mean = len(logs), float(logs.mean())
    half = T_975[n - 1] * float(logs.std(ddof=1)) / math.sqrt(n) if n >= 2 else math.nan
    return {
        "n": n,
        "ratio": math.exp(mean),
        "ci_low": math.exp(mean - half),
        "ci_high": math.exp(mean + half),
        "p": sign_flip_p(logs) if n >= 2 else math.nan,
    }


def verdict(effect: dict) -> str:
    """The claim BENCHMARK.md allows: significant after correction, interval excludes 1, and at least 5%."""
    if not effect["p_holm"] < ALPHA:
        return "no demonstrated difference"
    if effect["ci_low"] > 1 and effect["ratio"] >= MIN_EFFECT:
        return "MLX lower"
    if effect["ci_high"] < 1 and effect["ratio"] <= 1 / MIN_EFFECT:
        return "diffusers lower"
    return "no practically relevant difference"


# ------------------------------------------------------------------------------------------------------ loading


def load_records(path: Path) -> list[dict]:
    lines = filter(str.strip, (path / "results.jsonl").read_text().splitlines())
    return [json.loads(line) for line in lines]


def process_values(record: dict) -> dict | None:
    """One process: cold and warm totals (without loading), stages, and whether conditions were clean."""
    runs = record["runs"]
    if record["status"] != "ok" or not runs:
        return None

    def total(run):
        return sum(v for k, v in run["seconds"].items() if k not in ("load", "load_media"))

    warm = runs[1:]
    steps = []
    for run in runs:
        steps.extend(run.get("step_seconds", [])[1:])  # the first step includes kernel compilation
    return {
        "round": record["round"],
        "cold": total(runs[0]),
        "warm": statistics.median(total(r) for r in warm) if warm else None,
        "load": runs[0]["seconds"].get("load", 0.0) + runs[0]["seconds"].get("load_media", 0.0),
        "cold_stages": dict(runs[0]["seconds"]),
        "warm_stages": {
            stage: statistics.median(r["seconds"].get(stage, 0.0) for r in warm)
            for stage in {s for r in warm for s in r["seconds"]}
        },
        "footprint": record["footprint_peak_gib"],
        "footprint_stages": {
            stage: max(r.get("footprint_peak_gib", {}).get(stage, 0.0) for r in runs)
            for stage in {s for r in runs for s in r.get("footprint_peak_gib", {})}
        },
        "framework": max((v for r in runs for v in r["framework_peak_gib"].values()), default=None),
        "step": statistics.median(steps) if steps else None,
        "step_spread": (min(steps), max(steps)) if steps else None,
        "clean": record.get("power") in (None, "AC Power")
        and record.get("speed_limit") in (None, 100)
        and record["max_swap_delta_mb"] <= 500,
        "swap": record["max_swap_delta_mb"],
    }


def group_cells(records: list[dict]) -> dict[str, dict]:
    cells: dict[str, dict] = {}
    for record in records:
        entry = cells.setdefault(record["cell"]["id"], {"cell": record["cell"], "ok": [], "failed": []})
        values = process_values(record)
        if values is None:
            entry["failed"].append(
                {"round": record.get("round"), "status": record["status"], "footprint": record["footprint_peak_gib"]}
            )
        else:
            entry["ok"].append(values)
    return cells


def spread(values: list[float]) -> dict:
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(cells: dict[str, dict]) -> list[dict]:
    rows = []
    for cell_id, entry in cells.items():
        row = {"id": cell_id, "n": len(entry["ok"]), "failed": len(entry["failed"])}
        row["status"] = entry["failed"][0]["status"] if entry["failed"] else "ok"
        for metric in ("cold", "warm", "load", "footprint", "framework", "step"):
            values = [p[metric] for p in entry["ok"] if p[metric] is not None]
            if values:
                s = spread(values)
                row |= {f"{metric}_median": s["median"], f"{metric}_min": s["min"], f"{metric}_max": s["max"]}
        rows.append(row)
    return rows


def primary_stats(cells: dict[str, dict], clean_only: bool = False) -> dict:
    """The 8 primary time tests (and the 4 footprint comparisons as their own family), diffusers / MLX."""
    times, memory = [], []
    for label, (mlx_id, diffusers_id) in PRIMARY.items():
        if mlx_id not in cells or diffusers_id not in cells:
            continue

        def by_round(cell_id):
            return {p["round"]: p for p in cells[cell_id]["ok"] if p["clean"] or not clean_only}

        mlx, diffusers = by_round(mlx_id), by_round(diffusers_id)
        rounds = sorted(set(mlx) & set(diffusers))
        for metric, family in (("cold", times), ("warm", times), ("footprint", memory)):
            pairs = [(diffusers[r][metric], mlx[r][metric]) for r in rounds if mlx[r][metric] and diffusers[r][metric]]
            if len(pairs) >= 2:
                effect = ratio_effect(*zip(*pairs))
                family.append({"config": label, "metric": metric} | effect)
    for family in (times, memory):
        for item, adjusted in zip(family, holm([x["p"] for x in family])):
            item["p_holm"] = adjusted
            item["verdict"] = verdict(item)
    return {"time": times, "memory": memory}


# -------------------------------------------------------------------------------------------------------- tables


def fmt(value, digits=1) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def cell_table(rows: list[dict], prefix: str) -> list[str]:
    lines = [
        "| Cell | n | Cold, s (min–max) | Warm, s (min–max) | Load, s | Step, s | OS footprint, GiB (max) | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if not r["id"].startswith(prefix):
            continue
        cold = (
            f"{fmt(r.get('cold_median'))} ({fmt(r.get('cold_min'))}–{fmt(r.get('cold_max'))})"
            if "cold_median" in r
            else "–"
        )
        warm = (
            f"{fmt(r.get('warm_median'))} ({fmt(r.get('warm_min'))}–{fmt(r.get('warm_max'))})"
            if "warm_median" in r
            else "–"
        )
        memory = f"{fmt(r.get('footprint_median'))} ({fmt(r.get('footprint_max'))})" if "footprint_median" in r else "–"
        name = r["id"].split("/", 1)[1]
        lines.append(
            f"| `{name}` | {r['n']} | {cold} | {warm} | {fmt(r.get('load_median'))} | "
            f"{fmt(r.get('step_median'), 2)} | {memory} | {r['status']} |"
        )
    return lines


def write_tables(rows: list[dict], stats: dict, sensitivity: dict, path: Path) -> None:
    def effects(items):
        lines = [
            "| Configuration | Metric | n | diffusers / MLX | 95% CI | p (Holm) | Conclusion |",
            "|---|---|---|---|---|---|---|",
        ]
        for x in items:
            lines.append(
                f"| {x['config']} | {x['metric']} | {x['n']} | {x['ratio']:.2f} | {x['ci_low']:.2f}–{x['ci_high']:.2f} | "
                f"{x['p_holm']:.3f} | {x['verdict']} |"
            )
        return lines

    out = ["## Primary comparisons: time", "", *effects(stats["time"]), "", "## Primary comparisons: OS footprint", ""]
    out += [*effects(stats["memory"]), ""]
    if sensitivity != stats:
        out += ["## Sensitivity: rounds on AC power without throttling only", "", *effects(sensitivity["time"]), ""]
    for title, prefix in (
        ("Smoke cells", "smoke/"),
        ("Text encoding", "text/"),
        ("Memory-for-speed settings", "tradeoffs/"),
        ("Memory budgets, Smoke", "profiles/"),
        ("Baseline", "baseline/"),
        ("Memory budgets, Baseline", "profiles-baseline/"),
        ("Baseline with guidance", "baseline-cfg/"),
    ):
        if any(r["id"].startswith(prefix) for r in rows):
            out += [f"## {title}", "", *cell_table(rows, prefix), ""]
    path.write_text("\n".join(out) + "\n")


# -------------------------------------------------------------------------------------------------------- charts


def setup_style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "ytick.left": False,
            "xtick.bottom": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def save(fig, out: Path, name: str) -> None:
    for extension in ("png", "svg"):
        fig.savefig(out / f"{name}.{extension}", dpi=180 if extension == "png" else None, bbox_inches="tight")


def chart_forest(plt, stats: dict, out: Path) -> None:
    """Effect sizes with confidence intervals: the chart behind the claims."""
    items = stats["time"]
    if not items:
        return
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(items) + 1.5))
    labels = [f"{x['config']}, {x['metric']}" for x in items]
    for y, x in enumerate(items):
        ax.plot([x["ci_low"], x["ci_high"]], [y, y], color=IMPL_COLOR["mlx"], linewidth=2, solid_capstyle="round")
        ax.plot(x["ratio"], y, "o", color=IMPL_COLOR["mlx"], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(x["ci_high"], y, f"  {x['ratio']:.2f}", va="center", color=INK)
    ax.axvline(1, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(items)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("diffusers on MPS time / MLX time (above 1: MLX faster), with 95% CI")
    ax.set_title("MLX vs diffusers on MPS, Smoke, total without loading", loc="left", fontsize=12)
    save(fig, out, "ratio_ci")
    plt.close(fig)


def chart_stages(plt, cells: dict, out: Path) -> None:
    """Median warm time per stage, stacked, per primary cell: where the time goes."""
    entries = []
    for label, pair in PRIMARY.items():
        for cell_id in pair:
            if cell_id in cells and cells[cell_id]["ok"]:
                entries.append((label, cell_id, cells[cell_id]))
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(entries) + 1.6))
    names = []
    for y, (label, _, entry) in enumerate(entries):
        impl = entry["cell"]["impl"]
        names.append(f"{label}, {IMPL_LABEL[impl]}")
        left = 0.0
        for stage in STAGES:
            values = [
                p["cold_stages"].get(stage) if not p["warm_stages"] else p["warm_stages"].get(stage)
                for p in entry["ok"]
            ]
            values = [v for v in values if v]
            if not values:
                continue
            width = statistics.median(values)
            ax.barh(y, width, left=left, height=0.62, color=STAGE_COLOR[stage], edgecolor=SURFACE, linewidth=2)
            left += width
        ax.text(left, y, f"  {left:.1f} s", va="center", color=INK)
    ax.set_yticks(range(len(entries)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds, median of warm runs per process (diffusers reloads its DiT in warm runs, like MLX)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLOR[s]) for s in STAGES]
    ax.legend(handles, [STAGE_LABELS[s] for s in STAGES], ncol=5, loc="upper center", bbox_to_anchor=(0.45, -0.14))
    ax.set_title("Time per stage, Smoke (256x432, 25 frames)", loc="left", fontsize=12)
    save(fig, out, "time_per_stage")
    plt.close(fig)


def chart_cold_warm(plt, cells: dict, out: Path) -> None:
    """Cold vs warm totals (loading included for cold), median with min-max whiskers."""
    entries = [
        (label, cell_id)
        for label, pair in PRIMARY.items()
        for cell_id in pair
        if cell_id in cells and cells[cell_id]["ok"]
    ]
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(entries) + 1.5))
    names = []
    for y, (label, cell_id) in enumerate(entries):
        entry = cells[cell_id]
        names.append(f"{label}, {IMPL_LABEL[entry['cell']['impl']]}")
        color = IMPL_COLOR[entry["cell"]["impl"]]
        cold = [p["cold"] + (p["load"] or 0) for p in entry["ok"]]
        warm = [p["warm"] + p["warm_stages"].get("load", 0) for p in entry["ok"] if p["warm"]]
        for values, offset, alpha, tag in ((cold, -0.19, 1.0, "cold"), (warm, 0.19, 0.55, "warm")):
            if not values:
                continue
            middle = statistics.median(values)
            ax.barh(y + offset, middle, height=0.34, color=color, alpha=alpha)
            ax.plot([min(values), max(values)], [y + offset] * 2, color=INK, linewidth=1)
            ax.text(max(values), y + offset, f"  {tag} {middle:.1f} s", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(entries)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds including loading; bar: median over processes; line: min to max")
    ax.set_title("Before and after warm-up, Smoke", loc="left", fontsize=12)
    save(fig, out, "cold_vs_warm")
    plt.close(fig)


def chart_memory(plt, cells: dict, out: Path) -> None:
    """Peak OS footprint per stage for the Lite-distill T2V pair, MLX vs diffusers."""
    pair = PRIMARY["Lite-distill T2V"]
    if not all(p in cells and cells[p]["ok"] for p in pair):
        return
    stages = [s for s in STAGES if s != "image_encode"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for offset, cell_id in zip((-0.2, 0.2), pair):
        entry = cells[cell_id]
        impl = entry["cell"]["impl"]
        for y, stage in enumerate(stages):
            values = [p["footprint_stages"].get(stage, 0.0) for p in entry["ok"]]
            middle = statistics.median(values)
            ax.barh(y + offset, middle, height=0.36, color=IMPL_COLOR[impl], label=IMPL_LABEL[impl] if y == 0 else None)
            ax.text(middle, y + offset, f"  {middle:.1f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(stages)), [STAGE_LABELS[s] for s in stages])
    ax.invert_yaxis()
    ax.set_xlabel("Peak physical footprint, GiB (median over processes)")
    ax.legend(loc="lower right")
    ax.set_title("Memory per stage, Lite-distill, Smoke", loc="left", fontsize=12)
    save(fig, out, "memory_per_stage")
    plt.close(fig)


def chart_tradeoffs(plt, cells: dict, out: Path) -> None:
    """What memory buys: warm time against OS footprint for each setting."""
    entries = [(k.split("/")[-1], v) for k, v in cells.items() if k.startswith("tradeoffs/") and v["ok"]]
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name, entry in entries:
        x = statistics.median(p["footprint"] for p in entry["ok"])
        y = statistics.median((p["warm"] or p["cold"]) for p in entry["ok"])
        ax.plot(x, y, "o", color=IMPL_COLOR["mlx"], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(name, (x, y), xytext=(7, 4), textcoords="offset points", fontsize=9, color=INK)
    ax.grid(True, axis="both")
    ax.set_xlabel("Peak physical footprint, GiB")
    ax.set_ylabel("Warm total without loading, s")
    ax.tick_params(axis="y", colors=MUTED)
    ax.set_xlim(left=0)
    ax.set_title("What memory buys: MLX settings, Lite-distill, Smoke", loc="left", fontsize=12)
    save(fig, out, "memory_vs_speed")
    plt.close(fig)


def chart_profiles(plt, cells: dict, out: Path) -> None:
    """Footprint and time per memory budget, as two panels on their own scales."""
    entries = [v for k, v in cells.items() if k.startswith("profiles/") and v["ok"]]
    if not entries:
        return
    entries.sort(key=lambda e: e["cell"]["profile_ram_gb"])
    labels = [f"{e['cell']['profile_ram_gb']} GB Mac" for e in entries]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
    for y, entry in enumerate(entries):
        memory = statistics.median(p["footprint"] for p in entry["ok"])
        seconds = statistics.median((p["warm"] or p["cold"]) for p in entry["ok"])
        left.barh(y, memory, height=0.6, color=IMPL_COLOR["mlx"])
        left.text(memory, y, f"  {memory:.1f}", va="center", fontsize=9)
        left.plot(entry["cell"]["memory_limit_gib"], y, "|", color=INK, markersize=16, markeredgewidth=2)
        right.barh(y, seconds, height=0.6, color=IMPL_COLOR["mlx"])
        right.text(seconds, y, f"  {seconds:.1f}", va="center", fontsize=9)
    left.set_yticks(range(len(entries)), labels)
    left.invert_yaxis()
    left.set_xlabel("Peak footprint, GiB (tick: the budget)")
    right.set_xlabel("Warm total without loading, s")
    left.set_title("Emulated memory budgets, Smoke (same chip)", loc="left", fontsize=12)
    save(fig, out, "memory_budgets")
    plt.close(fig)


def chart_text(plt, cells: dict, out: Path) -> None:
    entries = [(k.split("/")[-1], v) for k, v in cells.items() if k.startswith("text/") and v["ok"]]
    if not entries:
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 2.4), sharey=True)
    for y, (name, entry) in enumerate(entries):
        color = IMPL_COLOR[entry["cell"]["impl"]]
        seconds = statistics.median(p["warm"] for p in entry["ok"] if p["warm"])
        memory = statistics.median(p["footprint"] for p in entry["ok"])
        left.barh(y, seconds, height=0.6, color=color)
        left.text(seconds, y, f"  {seconds:.2f}", va="center", fontsize=9)
        right.barh(y, memory, height=0.6, color=color)
        right.text(memory, y, f"  {memory:.1f}", va="center", fontsize=9)
    left.set_yticks(range(len(entries)), [n for n, _ in entries])
    left.invert_yaxis()
    left.set_xlabel("Warm encode of the Smoke prompt, s")
    right.set_xlabel("Peak footprint, GiB")
    left.set_title("Text encoding (Qwen2.5-VL + CLIP-L)", loc="left", fontsize=12)
    save(fig, out, "text_encoding")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="folder with results.jsonl")
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks"))
    args = parser.parse_args(argv)

    records = load_records(args.results)
    cells = group_cells(records)
    rows = summarize(cells)
    stats, sensitivity = primary_stats(cells), primary_stats(cells, clean_only=True)
    args.out.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row}, key=lambda k: (k != "id", k))
    with (args.out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)
    failures = {k: v["failed"] for k, v in cells.items() if v["failed"]}
    machine = args.results / "machine.json"
    (args.out / "stats.json").write_text(
        json.dumps(
            {
                "primary": stats,
                "sensitivity_clean_rounds": sensitivity,
                "failed_or_stopped": failures,
                "machine": json.loads(machine.read_text()) if machine.is_file() else None,
            },
            indent=2,
        )
        + "\n"
    )
    write_tables(rows, stats, sensitivity, args.out / "tables.md")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_style(plt)
    chart_forest(plt, stats, args.out)
    chart_stages(plt, cells, args.out)
    chart_cold_warm(plt, cells, args.out)
    chart_memory(plt, cells, args.out)
    chart_tradeoffs(plt, cells, args.out)
    chart_profiles(plt, cells, args.out)
    chart_text(plt, cells, args.out)
    print(f"{len(records)} records, {len(cells)} cells -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
