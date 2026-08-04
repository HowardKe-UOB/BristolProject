"""Paper-ready figures, ENGLISH-ONLY labels (academic convention per user rule).

Same data and validated palette as make_figures.py; outputs figures/*_en_v1.png.

    python make_figures_en.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
RED = "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
ORDINAL = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "Arial", "Helvetica"],
    "axes.unicode_minus": False,
    "figure.dpi": 200, "savefig.dpi": 200,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK2,
})
OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def style_ax(ax, ygrid=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left"]].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ================= Fig 1: the label-free journey ================= #
stages = [
    ("Old protocol (256-d projection)", 0.503),
    ("768-d backbone feature", 0.620),
    ("+ camera centering + re-rank", 0.663),
    ("+ reciprocal rank fusion (RRF)", 0.706),
    ("+ horizontal-flip TTA", 0.718),
    ("CAP 5-seed ensemble", 0.749),
    ("Ensemble-distilled trio (k=2)", 0.883),
]
fig, ax = plt.subplots(figsize=(8.4, 4.2))
y = np.arange(len(stages))[::-1]
vals = [v for _, v in stages]
ax.barh(y, vals, height=0.62, color=ORDINAL)
for yi, v in zip(y, vals):
    ax.text(v + 0.008, yi, f"{v:.3f}", va="center", ha="left", fontsize=10,
            color=INK, fontweight="bold")
ax.axvline(0.963, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
ax.text(0.960, len(stages) - 0.35, "supervised 0.963 ", color=INK2, fontsize=9,
        ha="right", va="bottom")
ax.set_yticks(y)
ax.set_yticklabels([s for s, _ in stages], fontsize=9.5)
ax.set_xlim(0, 1.02)
ax.set_xlabel("rank-1 (leave-out 66.130 cross-view protocol)")
ax.set_title("The label-free journey: rank-1 0.503 → 0.883",
             fontsize=12.5, loc="left", pad=12)
style_ax(ax, ygrid=False)
ax.grid(axis="x", color=GRID, linewidth=0.8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_journey_en_v1.png", facecolor="white")
plt.close(fig)

# ================= Fig 2: ensemble-size curve ================= #
k = np.array([1, 2, 3, 4, 5])
mean = np.array([0.674, 0.715, 0.728, 0.739, 0.742])
std = np.array([0.108, 0.063, 0.051, 0.035, 0.0])
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.fill_between(k, mean - std, mean + std, color=BLUE, alpha=0.15, linewidth=0)
ax.plot(k, mean, color=BLUE, linewidth=2, marker="o", markersize=7,
        markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=1.5)
for xi, m, s in zip(k, mean, std):
    ax.annotate(f"{m:.3f}±{s:.3f}", (xi, m), textcoords="offset points",
                xytext=(0, 11), ha="center", fontsize=9, color=INK2)
ax.set_xticks(k)
ax.set_xlabel("number of CAP seeds k in the ensemble")
ax.set_ylabel("rank-1")
ax.set_ylim(0.5, 0.85)
ax.set_title("Seed-ensemble size curve: mean rises, variance shrinks\n"
             "(all C(5,k) subsets, mean ± std)", fontsize=12, loc="left", pad=12)
style_ax(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_ensemble_curve_en_v1.png", facecolor="white")
plt.close(fig)

# ================= Fig 3: per-camera protocol sweep ================= #
cams = ["66.1", "66.128", "66.130*", "66.133", "66.139", "66.3", "66.33",
        "FULL\ntransductive"]
unsup_hold = [0.482, 0.544, 0.877, 0.542, 0.520, 0.418, 0.557, 0.516]
unsup_dep = [0.554, 0.625, 0.883, 0.551, 0.528, 0.398, 0.557, 0.549]
supv = [0.964, 0.978, 0.963, 0.915, 0.912, 0.979, 0.877, 0.956]
x = np.arange(len(cams))
w = 0.26
fig, ax = plt.subplots(figsize=(9.2, 4.4))
for off, vals, col, lab in [(-w, unsup_hold, BLUE, "unsupervised (holdout mode)"),
                            (0, unsup_dep, AQUA, "unsupervised (deployment mode)"),
                            (w, supv, YELLOW, "supervised")]:
    ax.bar(x + off, vals, width=w - 0.02, color=col, label=lab)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.012, f"{v:.2f}".lstrip("0"), ha="center", fontsize=7,
                color=INK2)
ax.set_xticks(x)
ax.set_xticklabels(cams, fontsize=9)
ax.set_ylabel("rank-1")
ax.set_ylim(0, 1.12)
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax.set_title("Per-camera query sweep (* 66.130 = oblique camera, unseen in training)",
             fontsize=12, loc="left", pad=12)
style_ax(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_percamera_en_v1.png", facecolor="white")
plt.close(fig)

# ================= Fig 4: label-efficiency (legacy) ================= #
frac = [0, 10, 25, 50, 100]
r1 = [0.476, 0.559, 0.694, 0.835, 0.850]
r5 = [0.715, 0.767, 0.853, 0.932, 0.924]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(frac, r5, color=AQUA, linewidth=2, marker="o", markersize=7,
        markerfacecolor=AQUA, markeredgecolor="white", markeredgewidth=1.5, label="rank-5")
ax.plot(frac, r1, color=BLUE, linewidth=2, marker="o", markersize=7,
        markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=1.5, label="rank-1")
ax.annotate("rank-5", (frac[-1], r5[-1]), textcoords="offset points", xytext=(8, 0),
            fontsize=9, color=INK2, va="center")
ax.annotate("rank-1", (frac[-1], r1[-1]), textcoords="offset points", xytext=(8, 0),
            fontsize=9, color=INK2, va="center")
for xi, v in zip(frac, r1):
    ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=8, color=INK2)
ax.set_xlabel("fraction of identities enrolled (%)")
ax.set_ylabel("accuracy")
ax.set_xlim(-4, 116)
ax.set_ylim(0.4, 1.0)
ax.legend(loc="lower right", fontsize=9, frameon=False)
ax.set_title("Semi-supervised label-efficiency curve\n(legacy ViT-S backbone, 2 seeds)",
             fontsize=12, loc="left", pad=12)
style_ax(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_label_efficiency_en_v1.png", facecolor="white")
plt.close(fig)

# ================= Fig 5: method families, wins vs losses ================= #
methods = [
    ("Ensemble distillation", 13.4),
    ("768-d backbone read-out", 11.7),
    ("Reciprocal rank fusion (RRF)", 5.0),
    ("Camera centering (CC)", 3.6),
    ("CAP + seed ensemble", 3.1),
    ("Horizontal-flip TTA", 1.2),
    ("Frozen ViT-S fusion", -2.5),
    ("Query expansion (AQE)", -8.0),
    ("Continued training (over-training)", -9.8),
    ("Bootstrap in CC space", -14.2),
    ("Part matching (patch tokens)", -15.4),
    ("GeM + BNNeck architecture swap", -17.8),
]
fig, ax = plt.subplots(figsize=(8.2, 5.0))
y = np.arange(len(methods))[::-1]
vals = [v for _, v in methods]
cols = [BLUE if v > 0 else RED for v in vals]
ax.barh(y, vals, height=0.6, color=cols)
for yi, v in zip(y, vals):
    ax.text(v + (0.5 if v > 0 else -0.5), yi, f"{v:+.1f}", va="center",
            ha="left" if v > 0 else "right", fontsize=9, color=INK, fontweight="bold")
ax.axvline(0, color=BASE, linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels([m for m, _ in methods], fontsize=9.5)
ax.set_xlabel("Δ rank-1 (points) vs. then-best label-free result")
ax.set_xlim(-22, 18)
ax.set_title("Twelve method families: what worked vs. what didn't",
             fontsize=12.5, loc="left", pad=12)
style_ax(ax, ygrid=False)
ax.grid(axis="x", color=GRID, linewidth=0.8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_methods_en_v1.png", facecolor="white")
plt.close(fig)

print("saved:", ", ".join(sorted(f for f in os.listdir(OUT) if "_en_" in f)))
