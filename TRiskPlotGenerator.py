import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from TRiskEvaluation import TRisk

# ==========================================
# Configuration
# ==========================================
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
OUT_DIR = os.environ.get("TRISK_FIG_DIR", "figures")   # where the .eps files are placed
CRITERION = "meaningbert"

DATASETS = [("asset", "ASSET"), ("turkcorpus", "TurkCorpus")]
# 2x2 grid: the four combined-reward (SARI+MB) models
MODELS = [
    ("llama1b_grpo_sari-mb", "Llama-1B-GRPO"),
    ("llama1b_rloo_sari-mb", "Llama-1B-RLOO"),
    ("llama3b_grpo_sari-mb", "Llama-3B-GRPO"),
    ("llama3b_rloo_sari-mb", "Llama-3B-RLOO"),
]
BASELINES = [("Hybrid", "Hybrid"), ("Dress", "DRESS"), ("Dress-Ls", "DRESS-Ls"), ("ACCESS", "ACCESS")]
COLORS = {"Hybrid": "#1f77b4", "DRESS": "#ff7f0e", "DRESS-Ls": "#2ca02c", "ACCESS": "#d62728"}
# markers give every baseline a pattern in addition to its colour (colour-blind readers, B/W print)
MARKERS = {"Hybrid": "o", "DRESS": "s", "DRESS-Ls": "^", "ACCESS": "D"}
ALPHAS = np.linspace(0, 5, 50)

plt.rcParams.update({"font.size": 11})


def load(stem):
    return pd.read_csv(f"{RESULTS_DIR}/{stem}-{CRITERION}.csv", header=None).iloc[:, 0].values


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating T-Risk 2x2 plots...")
    for ds, ds_label in DATASETS:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        axes = axes.ravel()
        handles_labels = None

        for ax, (mkey, mlab) in zip(axes, MODELS):
            try:
                tgt = load(f"{ds}-{mkey}")
            except Exception as e:
                print(f"  [skip] {ds}-{mkey}: {e}")
                continue
            for bkey, blab in BASELINES:
                try:
                    base = load(f"{ds}-{bkey}")
                except Exception:
                    continue
                if len(base) != len(tgt):
                    continue
                curve = [TRisk(tgt, base, a)[0] for a in ALPHAS]
                ax.plot(ALPHAS, curve, label=blab, color=COLORS[blab], linewidth=1.6,
                        marker=MARKERS[blab], markevery=7, markersize=5)

            ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.7)
            ax.axhline(2, color="gray", ls=":", lw=1.0)
            ax.axhline(-2, color="gray", ls=":", lw=1.0)
            ax.set_title(mlab, fontsize=12)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
            ax.grid(True, ls=":", alpha=0.5)
            if handles_labels is None:
                handles_labels = ax.get_legend_handles_labels()

        for ax in axes[2:]:
            ax.set_xlabel(r"$\alpha$ (risk aversion)")
        for ax in (axes[0], axes[2]):
            ax.set_ylabel(r"$T_{Risk}$")

        if handles_labels:
            fig.legend(*handles_labels, title="Baseline", loc="lower center",
                       ncol=4, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        out = os.path.join(OUT_DIR, f"TRisk_Plot_{ds}.eps")
        fig.savefig(out, format="eps", bbox_inches="tight")
        print(f"  saved {out}")
        plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
