# -*- coding: utf-8 -*-
"""Generate the publication figures from the final result tables.

Data source: TABLES_PUBLICATION_v2.md (accuracies, deltas, bootstrap CIs,
BH-adjusted McNemar p-values) and transitions.json (paired transitions).

Outputs (PDF, vector) into ./figures/:
  fig1_baseline_vs_delta.pdf  - scatter: baseline accuracy vs correction delta
  fig2_forest_deltas.pdf      - forest plot of deltas with 95% CI, both phases
  fig3_heatmap_deltas.pdf     - heatmap of deltas, model x (dataset, phase)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

MODELS = ["qwen2.5:7b", "gemma3:4b", "phi3:mini", "mistral:7b",
          "llama3:8b", "nemotron-mini:4b"]

# baseline accuracy (%) per dataset
BASE = {
    "A": {"qwen2.5:7b": 74.0, "gemma3:4b": 78.0, "phi3:mini": 62.0,
          "mistral:7b": 56.0, "llama3:8b": 82.0, "nemotron-mini:4b": 62.0},
    "B": {"qwen2.5:7b": 78.3, "gemma3:4b": 71.7, "phi3:mini": 70.0,
          "mistral:7b": 71.7, "llama3:8b": 76.7, "nemotron-mini:4b": 56.7},
    "C": {"qwen2.5:7b": 98.0, "gemma3:4b": 80.0, "phi3:mini": 72.0,
          "mistral:7b": 78.0, "llama3:8b": 74.0, "nemotron-mini:4b": 50.0},
}

# (delta, ci_low, ci_high, p_adj) per phase/dataset/model
EFFECTS = {
    ("1.1", "A"): {
        "mistral:7b": (24.0, 12.0, 36.0, 0.003),
        "qwen2.5:7b": (12.0, 4.0, 22.0, 0.047),
        "phi3:mini": (12.0, 4.0, 22.0, 0.047),
        "gemma3:4b": (-2.0, -12.0, 8.0, 1.000),
        "llama3:8b": (-14.0, -28.0, 0.0, 0.111),
        "nemotron-mini:4b": (-14.0, -24.0, -6.0, 0.047),
    },
    ("1.1", "B"): {
        "qwen2.5:7b": (5.0, -3.3, 13.3, 0.929),
        "phi3:mini": (3.3, -8.3, 15.0, 0.929),
        "nemotron-mini:4b": (3.3, -3.3, 10.0, 0.929),
        "gemma3:4b": (1.7, -6.7, 10.0, 1.000),
        "mistral:7b": (-6.7, -20.0, 6.7, 0.929),
        "llama3:8b": (-15.0, -28.3, -1.7, 0.381),
    },
    ("1.1", "C"): {
        "gemma3:4b": (6.0, -4.0, 16.0, 0.453),
        "nemotron-mini:4b": (-6.0, -14.0, 2.0, 0.450),
        "qwen2.5:7b": (-6.0, -14.0, 0.0, 0.375),
        "mistral:7b": (-12.0, -26.0, 2.0, 0.292),
        "phi3:mini": (-12.0, -24.0, 0.0, 0.292),
        "llama3:8b": (-14.0, -30.0, 2.0, 0.292),
    },
    ("1.2", "A"): {
        "qwen2.5:7b": (10.0, -2.0, 22.0, 0.359),
        "phi3:mini": (2.0, -8.0, 12.0, 1.000),
        "gemma3:4b": (0.0, -12.0, 10.0, 1.000),
        "mistral:7b": (-2.0, -18.0, 14.0, 1.000),
        "nemotron-mini:4b": (-14.0, -24.0, -6.0, 0.047),
        "llama3:8b": (-22.0, -36.0, -8.0, 0.047),
    },
    ("1.2", "B"): {
        "gemma3:4b": (3.3, -6.7, 13.3, 0.754),
        "qwen2.5:7b": (3.3, -6.7, 13.3, 0.754),
        "nemotron-mini:4b": (-13.3, -25.0, -3.3, 0.058),
        "phi3:mini": (-18.3, -33.3, -3.3, 0.058),
        "llama3:8b": (-20.0, -33.3, -6.7, 0.035),
        "mistral:7b": (-23.3, -36.7, -10.0, 0.026),
    },
    ("1.2", "C"): {
        "qwen2.5:7b": (-6.0, -14.0, 0.0, 0.375),
        "llama3:8b": (-10.0, -26.0, 6.0, 0.399),
        "gemma3:4b": (-2.0, -14.0, 10.0, 1.000),
        "nemotron-mini:4b": (-18.0, -30.0, -6.0, 0.026),
        "phi3:mini": (-20.0, -34.0, -6.0, 0.026),
        "mistral:7b": (-32.0, -48.0, -16.0, 0.005),
    },
}

COLORS = {
    "qwen2.5:7b": "#1f77b4",
    "gemma3:4b": "#2ca02c",
    "phi3:mini": "#9467bd",
    "mistral:7b": "#ff7f0e",
    "llama3:8b": "#d62728",
    "nemotron-mini:4b": "#7f7f7f",
}
MARKERS = {"1.1": "o", "1.2": "^"}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
})


def fig1_scatter():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for (phase, ds), effects in EFFECTS.items():
        for m, (delta, lo, hi, p) in effects.items():
            ax.scatter(BASE[ds][m], delta, c=COLORS[m], marker=MARKERS[phase],
                       s=42 if p < 0.05 else 26,
                       edgecolors="black" if p < 0.05 else "none",
                       linewidths=0.8, alpha=0.9, zorder=3)
    ax.axhline(0, color="black", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("Baseline accuracy, Phase 1.0 (%)")
    ax.set_ylabel("Accuracy delta under review (points)")
    ax.set_title("Baseline accuracy does not predict correction response")
    # legends: models (colors) and phases (markers)
    model_handles = [plt.Line2D([], [], marker="s", ls="", color=COLORS[m],
                                label=m) for m in MODELS]
    phase_handles = [
        plt.Line2D([], [], marker="o", ls="", color="black", label="Phase 1.1 (simple)"),
        plt.Line2D([], [], marker="^", ls="", color="black", label="Phase 1.2 (directed)"),
        plt.Line2D([], [], marker="o", ls="", color="white", markeredgecolor="black",
                   label="significant ($p_{adj}<0.05$)"),
    ]
    leg1 = ax.legend(handles=model_handles, loc="lower left", frameon=False, ncol=2)
    ax.add_artist(leg1)
    ax.legend(handles=phase_handles, loc="upper right", frameon=False)
    ax.set_ylim(-40, 32)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_baseline_vs_delta.pdf"))
    plt.close(fig)


def fig2_forest():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.4), sharey=True)
    datasets = ["A", "B", "C"]
    y_labels, y_pos = [], []
    for pi, phase in enumerate(["1.1", "1.2"]):
        ax = axes[pi]
        y = 0
        yticks, ylabels = [], []
        for m in MODELS:
            for ds in datasets:
                delta, lo, hi, p = EFFECTS[(phase, ds)][m]
                sig = p < 0.05
                ax.errorbar(delta, y, xerr=[[delta - lo], [hi - delta]],
                            fmt="o", color=COLORS[m], markersize=4.5,
                            elinewidth=1.4 if sig else 0.8,
                            capsize=2.5,
                            markeredgecolor="black" if sig else "none",
                            markeredgewidth=0.8)
                yticks.append(y)
                ylabels.append(f"{m}  ({ds})")
                y += 1
            y += 0.6  # gap between models
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_title(f"Phase {phase} ({'simple' if phase == '1.1' else 'directed'} review)")
        ax.set_xlabel("Δ accuracy vs. baseline (points)")
        ax.set_xlim(-50, 40)
        if pi == 0:
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels, fontsize=6.5)
        ax.invert_yaxis()
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.suptitle("Correction effects with 95% bootstrap CIs (bold outline: $p_{adj}<0.05$)",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "fig2_forest_deltas.pdf"))
    plt.close(fig)


def fig3_heatmap():
    cols = [("1.1", "A"), ("1.1", "B"), ("1.1", "C"),
            ("1.2", "A"), ("1.2", "B"), ("1.2", "C")]
    data = np.array([[EFFECTS[c][m][0] for c in cols] for m in MODELS])
    sig = np.array([[EFFECTS[c][m][3] < 0.05 for c in cols] for m in MODELS])
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    vmax = 32
    im = ax.imshow(data, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{p}\n{d}" for p, d in cols])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    for i in range(len(MODELS)):
        for j in range(len(cols)):
            label = f"{data[i, j]:+.1f}".rstrip("0").rstrip(".")
            if sig[i, j]:
                label += "*"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.5,
                    color="white" if abs(data[i, j]) > 18 else "black",
                    fontweight="bold" if sig[i, j] else "normal")
    ax.set_title("Accuracy deltas by model, phase, and dataset (* $p_{adj}<0.05$)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Δ accuracy (points)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_heatmap_deltas.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig1_scatter()
    fig2_forest()
    fig3_heatmap()
    print("Figures written to", OUT)
