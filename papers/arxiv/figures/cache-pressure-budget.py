#!/usr/bin/env python3
"""CPU-only cache-pressure figure; verify public typed records before plotting."""
import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent / "experiments/storage_demand/pressure_fault_v1/publication_pressure_v1"


def plot(bundle=BUNDLE, output=HERE):
    spec = importlib.util.spec_from_file_location("pressure_public_verify", bundle / "verify.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    result = verifier.derive(verifier.load_bundle(bundle))
    # Keep font caches outside both the repository and the user's default cache.
    with tempfile.TemporaryDirectory(prefix="cache-pressure-mpl-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from matplotlib.ticker import MultipleLocator

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.titlesize": 11, "axes.labelsize": 10,
                             "legend.fontsize": 8.5, "pdf.fonttype": 42,
                             "ps.fonttype": 42, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.axisbelow": True})
        blue, orange, green, purple = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
        points = result["points"]
        x = list(range(3))
        labels = [f'{p["point"]}\n{p["requested_entries"]} entries\n{p["requested_cache_bytes"] / 2**20:g} MiB'
                  for p in points]
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
        fig.subplots_adjust(left=0.085, right=0.975, top=0.845, bottom=0.18, wspace=0.28, hspace=0.68)
        fig.suptitle("More retained cache, unchanged reader traffic", fontsize=16, weight="bold", y=0.975)
        fig.text(0.5, 0.934, "Coupled entry / byte-budget sweep · blocking only · one fresh lifecycle per point",
                 ha="center", fontsize=10.5, color="#444444")
        for ax in axes.flat:
            ax.set_xticks(x, labels)
            ax.set_xlim(-0.55, 2.55)
            ax.tick_params(axis="x", length=0, pad=7, labelsize=9)
            ax.grid(axis="y", color="#E3E6E8", linewidth=0.7)

        ax = axes[0, 0]
        ending = [p["final_owner_cache"]["charged_bytes"] / 2**20 for p in points]
        peak = [p["final_owner_cache"]["peak_bytes"] / 2**20 for p in points]
        caps = [p["requested_cache_bytes"] / 2**20 for p in points]
        ax.bar([i - .17 for i in x], ending, width=.30, color=blue, label="Ending owner charge")
        ax.bar([i + .17 for i in x], peak, width=.30, color=orange, label="Owner logical peak")
        ax.plot(x, caps, "s--", color="#555555", linewidth=1.3, markersize=4, label="Requested ceiling")
        for i, end, maximum, cap in zip(x, ending, peak, caps):
            ax.annotate(f"{end:.1f}", (i - .17, end), xytext=(-2, 5), textcoords="offset points",
                        ha="right", fontsize=8, color=blue,
                        bbox=dict(facecolor="white", edgecolor="none", pad=.5))
            ax.annotate(f"{maximum:.1f}", (i + .17, maximum), xytext=(2, 5), textcoords="offset points",
                        ha="left", fontsize=8, color="#926100")
            ax.annotate(f"{cap:g}", (i, cap), xytext=(0, 13 if i == 0 else 7), textcoords="offset points",
                        ha="center", fontsize=8.5, color="#444444")
        ax.set_ylim(0, 780)
        ax.yaxis.set_major_locator(MultipleLocator(200))
        ax.set_ylabel("Cache logical charge / budget (MiB)")
        ax.set_title("A  Requested limits vs. observed charge", loc="left", pad=12, weight="bold")
        handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend(handles, legend_labels, loc="upper left", frameon=False)

        ax = axes[0, 1]
        gpu = [p["memory_sample_bounds"]["device_gpu"]["sampled_max_bytes"] / 2**20 for p in points]
        ax.bar(x, gpu, width=.52, color=purple)
        for i, value in zip(x, gpu):
            ax.annotate(f"{value:,.0f}", (i, value), xytext=(0, 7), textcoords="offset points", ha="center", weight="bold")
        ax.set_ylim(0, 5600)
        ax.yaxis.set_major_locator(MultipleLocator(1000))
        ax.set_ylabel("Sampled GPU memory maximum (MiB)")
        ax.set_title("B  GPU memory: 36 samples per point", loc="left", pad=12, weight="bold")
        ax.text(.5, .96, "Not exact lifetime peaks or proven upper bounds", transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, color="#555555")

        ax = axes[1, 0]
        reads = [p["invocation_counter_totals"]["checkpoint_read_bytes"] / 2**30 for p in points]
        ax.plot(x, reads, "o-", color=green, linewidth=2, markersize=7)
        for i, p in zip(x, points):
            ax.annotate(f'{p["invocation_counter_totals"]["expert_hits"]} expert hits', (i, reads[i]),
                        xytext=(0, -21), textcoords="offset points", ha="center", fontsize=9, color="#006C50")
        ax.set_ylim(0, 43)
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.set_ylabel("Logical checkpoint reader bytes (GiB)")
        ax.set_title("C  Reader traffic is unchanged", loc="left", pad=12, weight="bold")
        ax.text(.5, .93, "36,240,064,128 bytes at every point", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, weight="bold", color="#006C50")
        ax.text(.5, .16, "69,114 read calls · 11,519 verified expert reads\nper point; not physical NAS backend bytes",
                transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555", linespacing=1.5)

        ax = axes[1, 1]
        names = ("Hello2", "QA", "Arithmetic", "Chinese")
        colors = (blue, orange, green, purple)
        markers = ("o", "s", "^", "D")
        for index, (name, color, marker) in enumerate(zip(names, colors, markers)):
            y = [p["requests"][index]["http_e2e_seconds"] for p in points]
            ax.scatter([i + (index - 1.5) * .12 for i in x], y, s=42, marker=marker,
                       color=color, label=name, edgecolors="white", linewidths=.4, zorder=3)
        ax.set_ylim(0, 9.5)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.set_ylabel("Individual HTTP E2E (s)")
        ax.set_title("D  Four prompts, n=1 per prompt / point", loc="left", pad=12, weight="bold")
        ax.legend(loc="upper center", ncol=4, frameon=False, columnspacing=.8, handletextpad=.3,
                  fontsize=8)
        ax.text(.5, .42, "Dots are requests, not statistical replicates\nNo CI, pooled latency, or causal speedup",
                transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555", linespacing=1.5)
        fig.text(.5, .063, "12/12 HTTP requests · 9/9 eight-ID canaries · 78 tokens. Limits, gauges, and sampled GPU memory are not additive.",
                 ha="center", fontsize=9, color="#333333")
        fig.text(.5, .034, "Fixed order, n=1: descriptive coupled-budget sensitivity, not an isolated LRU effect or physical capacity frontier.",
                 ha="center", fontsize=9, color="#333333")
        output = Path(output)
        if not output.is_dir():
            raise ValueError("output directory must already exist")
        created = datetime(2026, 10, 15, tzinfo=timezone.utc)
        fig.savefig(output / "cache-pressure-budget.pdf", metadata={"Title": "Coupled cache-budget pressure observations",
                    "Author": "Ferrule", "CreationDate": created, "ModDate": created})
        fig.savefig(output / "cache-pressure-budget.png", dpi=200, metadata={"Software": "Ferrule cache-pressure-budget.py"})
        plt.close(fig)
    print("Verified public records; wrote cache-pressure-budget.pdf and cache-pressure-budget.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    plot(args.bundle, args.output)
