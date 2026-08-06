"""Methods/pipeline schematic figures (English-only, paper-ready).

fig6: overall method pipeline, input -> CAP self-training -> ensemble teacher ->
      distilled students -> inference stack, with key stats annotated.
fig7: three mechanism demonstrations: (a) camera-aware proxies contain merge
      errors; (b) pairwise links vs transitive merges; (c) sparse-overlap
      topology signals.

    python make_methods_figs_en.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

BLUE, AQUA, YELLOW, RED = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE_L, AQUA_L, YELLOW_L, RED_L = "#e5efFA", "#e2f5ee", "#fdf3dd", "#fbe7e7"

plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "Arial", "Helvetica"],
    "axes.unicode_minus": False,
    "figure.dpi": 200, "savefig.dpi": 200,
    "text.color": INK,
})
OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def rbox(ax, x, y, w, h, fc, ec, lw=1.0, r=1.2, z=2):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(b)
    return b


def subbox(ax, x, y, w, h, title, lines, accent, fs_t=9.0, fs_b=7.8):
    rbox(ax, x, y, w, h, "white", accent, lw=1.2, r=1.0, z=3)
    ax.plot([x + 1.2, x + 1.2], [y + 1.0, y + h - 1.0], color=accent, lw=3,
            solid_capstyle="round", zorder=4)
    ax.text(x + 2.6, y + h - 2.2, title, fontsize=fs_t, fontweight="bold",
            color=INK, va="top", zorder=5)
    body = "\n".join(lines)
    ax.text(x + 2.6, y + h - 4.9, body, fontsize=fs_b, color=INK2, va="top",
            linespacing=1.45, zorder=5)


def harrow(ax, x0, x1, y, color=MUTED, lw=1.6, z=4):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=lw, zorder=z))


def varrow(ax, x, y0, y1, label=None, color=INK2, lw=2.2):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=16, color=color, lw=lw, zorder=6))
    if label:
        ax.text(x + 1.4, (y0 + y1) / 2, label, fontsize=8.6, color=INK2,
                va="center", ha="left", fontstyle="italic", zorder=6)


# ===================================================================== #
# FIG 6 -- overall pipeline
# ===================================================================== #
fig, ax = plt.subplots(figsize=(11.4, 13.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 138)
ax.axis("off")

ax.text(1, 136.5, "Label-free cross-camera cattle re-identification: method overview",
        fontsize=14, fontweight="bold", color=INK)
ax.text(1, 133.8, "no identity labels anywhere - only per-camera tracklets, timestamps, "
                  "and the camera-overlap topology", fontsize=9, color=MUTED)

# ---------------- band 1: input ---------------- #
B1Y, B1H = 112, 19
rbox(ax, 0.5, B1Y, 99, B1H, "#f7f6f3", BASE, lw=1.0, r=1.6, z=1)
ax.text(2.2, B1Y + B1H - 2.4, "STAGE 0 - INPUT STRUCTURE (free supervision only)",
        fontsize=9.5, fontweight="bold", color=MUTED, va="top")
subbox(ax, 3, B1Y + 1.8, 29, 12.5, "7-camera barn, one 2.2 h session",
       ["98 cows, ~124k crops, ~1 fps", "sparse-overlap topology:",
        "only ~7 of 21 camera pairs overlap"], BLUE)
subbox(ax, 35, B1Y + 1.8, 29, 12.5, "Per-camera tracklets",
       ["single-camera tracker output:", "contiguous same-cow runs",
        "(997 tracklets - no identity labels)"], BLUE)
subbox(ax, 67, B1Y + 1.8, 29.5, 12.5, "Temporal topology signal",
       ["same instant + non-overlapping FOVs", "=> cannot be the same cow",
        "cannot-link constraints (99.9% reliable)"], BLUE)
harrow(ax, 32.4, 34.6, B1Y + 8)
harrow(ax, 64.4, 66.6, B1Y + 8)

varrow(ax, 50, B1Y - 0.4, B1Y - 5.2)

# ---------------- band 2: rung 1 CAP ---------------- #
B2Y, B2H = 79.5, 27.5
rbox(ax, 0.5, B2Y, 99, B2H, BLUE_L, BLUE, lw=1.2, r=1.6, z=1)
ax.text(2.2, B2Y + B2H - 2.4, "STAGE 1 - SELF-TRAINING WITH CAMERA-AWARE PROXIES "
                              "(CAP)   x5 random seeds", fontsize=9.5,
        fontweight="bold", color=BLUE, va="top")
subbox(ax, 3, B2Y + 11.4, 22, 11.5, "Backbone",
       ["DINOv2 ViT-B/14", "fine-tune last 4 blocks", "tracklet clip -> 768-d"], BLUE)
subbox(ax, 27.5, B2Y + 11.4, 22, 11.5, "Intra-camera clustering",
       ["per camera, CA-Jaccard", "+ DBSCAN, cannot-link aware", "-> per-camera pseudo-IDs"], BLUE)
subbox(ax, 52, B2Y + 11.4, 22, 11.5, "Cross-camera mining",
       ["crop-level optimal transport", "on overlapping pairs",
        "~15% precision (the bottleneck)"], BLUE)
subbox(ax, 76.5, B2Y + 11.4, 20.5, 11.5, "CAP proxies",
       ["one proxy per", "(cluster, camera) pair,", "momentum-updated"], BLUE)
harrow(ax, 25.0, 27.3, B2Y + 17.1); harrow(ax, 49.5, 51.8, B2Y + 17.1)
harrow(ax, 74.0, 76.3, B2Y + 17.1)
subbox(ax, 3, B2Y + 1.2, 45, 9.6, "Losses",
       ["intra-camera softmax CE over same-camera proxies",
        "+ inter-camera soft pull to same-cluster, other-camera proxies",
        "+ cannot-link repulsion  (a wrong merge corrupts ONE proxy)"],
       BLUE, fs_b=7.0)
subbox(ax, 51, B2Y + 1.2, 45.5, 9.6, "Output: 5 diverse models",
       ["single-seed rank-1 = 0.67 +/- 0.11; 1 of 5 seeds collapses",
        "and no label-free detector exists -> ensemble all seeds"], BLUE, fs_b=7.0)

varrow(ax, 50, B2Y - 0.4, B2Y - 3.4)

# ---------------- band 3: teacher + rung 2 ---------------- #
B3Y, B3H = 45.5, 30
rbox(ax, 0.5, B3Y, 99, B3H, AQUA_L, AQUA, lw=1.2, r=1.6, z=1)
ax.text(2.2, B3Y + B3H - 2.4, "STAGE 2 - ENSEMBLE TEACHER  ->  DISTILLED STUDENTS "
                              "(the decisive step)", fontsize=9.5,
        fontweight="bold", color="#0e7a54", va="top")
subbox(ax, 3, B3Y + 14, 29, 12, "Seed-ensemble teacher space",
       ["mean of 5 seeds' distances", "rank-1 0.749 (from 0.67 +/- 0.11)",
        "cleaner neighbourhoods than any seed"], AQUA)
subbox(ax, 35, B3Y + 14, 29, 12, "Teacher-space mining",
       ["intra-camera clusters: P = 0.879", "cross-camera mutual 2-NN links:",
        "287 links @ 52% (vs 15% before)"], AQUA)
subbox(ax, 67, B3Y + 14, 29.5, 12, "Fixed-label students  x3 seeds",
       ["labels FROZEN (no drift/collapse)", "pairwise link loss - NEVER merge",
        "from scratch, 1000 steps, early stop"], AQUA)
harrow(ax, 32.4, 34.6, B3Y + 20); harrow(ax, 64.4, 66.6, B3Y + 20)
subbox(ax, 3, B3Y + 1.8, 45, 9.4, "Why pairwise, not transitive",
       ["union-find merging collapses 52-60% link precision", "to ~17% pair precision "
        "(measured); a pairwise pull", "bounds the damage of every wrong link"],
       AQUA, fs_b=7.2)
subbox(ax, 51, B3Y + 1.8, 45.5, 9.4, "Output: 3 low-variance students",
       ["single-model rank-1 = 0.81 +/- 0.03 (each ALONE > any", "previous ensemble); "
        "distillation ladder converges after", "one rung (re-mining gains nothing)"],
       AQUA, fs_b=7.2)

varrow(ax, 50, B3Y - 0.4, B3Y - 5.2)

# ---------------- band 4: inference ---------------- #
B4Y, B4H = 15, 24.5
rbox(ax, 0.5, B4Y, 99, B4H, YELLOW_L, YELLOW, lw=1.2, r=1.6, z=1)
ax.text(2.2, B4Y + B4H - 2.4, "STAGE 3 - INFERENCE STACK (all label-free, "
                              "per student model)", fontsize=9.5,
        fontweight="bold", color="#9a6a00", va="top")
steps = [
    ("768-d read-out", ["backbone feature,", "not projection head"]),
    ("flip TTA", ["average normal +", "mirrored embedding"]),
    ("camera centering", ["subtract per-camera", "mean, renormalise"]),
    ("PCA whitening", ["fit on gallery,", "256 dims"]),
    ("CA-Jaccard re-rank", ["camera-aware", "k-reciprocal"]),
    ("RRF fusion", ["fuse the 3 views'", "rankings (k=20)"]),
]
w = 14.6
for i, (t, ls) in enumerate(steps):
    x = 3 + i * (w + 1.4)
    subbox(ax, x, B4Y + 9.5, w, 10, t, ls, YELLOW, fs_t=8.0, fs_b=6.9)
    if i:
        harrow(ax, x - 1.5, x - 0.1, B4Y + 14.5, lw=1.2)
subbox(ax, 3, B4Y + 0.9, 62, 8.2, "Cross-student fusion",
       ["mean of the 3 students' fused distance matrices",
        "(+ optional spatio-temporal impossibility mask)"], YELLOW, fs_b=7.2)
rbox(ax, 68, B4Y + 1.4, 28.5, 7.2, INK, INK, r=1.0, z=3)
ax.text(82.2, B4Y + 6.4, "rank-1 0.883   rank-5 0.957", fontsize=10.5,
        fontweight="bold", color="white", ha="center", zorder=5)
ax.text(82.2, B4Y + 3.4, "mAP 0.536  (supervised: 0.963)", fontsize=8.5,
        color="#d9d8d3", ha="center", zorder=5)

ax.text(1, 11.2, "Protocol: leave-out camera 66.130 (oblique view, never seen in training); "
                 "query = 66.130 tracklets (n=163), gallery = 6 other cameras (n=834).",
        fontsize=8.2, color=MUTED)
ax.text(1, 8.8, "Full-transductive protocol: 0.549 with deployment-mode training "
                "(links mined across all 7 cameras).", fontsize=8.2, color=MUTED)

fig.tight_layout()
fig.savefig(f"{OUT}/fig6_pipeline_en_v3_regen_scratch.png", facecolor="white",
            bbox_inches="tight")
plt.close(fig)

# ===================================================================== #
# FIG 7 -- mechanism demonstrations
# ===================================================================== #
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
for a in axes:
    a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")

# ----- (a) CAP proxies contain merge errors ----- #
a = axes[0]
a.text(0.1, 9.6, "(a) Camera-aware proxies contain merge errors", fontsize=10.5,
       fontweight="bold", color=INK)
# left: cluster-contrast
a.text(2.3, 8.55, "Cluster-Contrast", fontsize=8.5, color=INK2, ha="center")
cowA = [(1.2, 6.6), (1.9, 7.3), (2.6, 6.5)]
cowB = [(2.0, 5.3), (2.9, 5.6)]
for x, y in cowA:
    a.add_patch(Circle((x, y), 0.22, color=BLUE, zorder=3))
for x, y in cowB:
    a.add_patch(Circle((x, y), 0.22, color=RED, zorder=3))
a.add_patch(Circle((2.1, 6.3), 0.34, facecolor="white", edgecolor=INK, lw=1.6, zorder=4))
a.text(2.1, 6.3, "C", fontsize=8, ha="center", va="center", zorder=5, fontweight="bold")
for x, y in cowA + cowB:
    a.add_patch(FancyArrowPatch((x, y), (2.1, 6.3), arrowstyle="-|>", mutation_scale=7,
                                color=MUTED, lw=0.9, zorder=2,
                                shrinkA=5, shrinkB=9))
a.text(2.2, 4.2, "one wrong merge -> two cows\ncollapse into ONE centroid",
       fontsize=7.6, color=RED, ha="center", va="top", linespacing=1.4)
# right: CAP
a.text(7.4, 8.55, "CAP (ours)", fontsize=8.5, color=INK2, ha="center")
prox = {"cam1": (6.2, 6.9), "cam2": (8.4, 6.9), "cam3": (7.3, 5.4)}
cols = {"cam1": BLUE, "cam2": BLUE, "cam3": RED}
for k, (x, y) in prox.items():
    a.add_patch(Circle((x, y), 0.30, facecolor="white", edgecolor=cols[k], lw=1.8, zorder=4))
    a.text(x, y, "P", fontsize=7.5, ha="center", va="center", zorder=5, color=cols[k],
           fontweight="bold")
    dy = -0.78 if k == "cam3" else 0.62
    a.text(x, y + dy, k.replace("cam", "camera "), fontsize=6.6, ha="center", color=MUTED)
a.add_patch(FancyArrowPatch(prox["cam1"], prox["cam2"], arrowstyle="<|-|>",
                            mutation_scale=8, color=AQUA, lw=1.4, zorder=3,
                            shrinkA=8, shrinkB=8))
a.add_patch(FancyArrowPatch(prox["cam1"], prox["cam3"], arrowstyle="<|-|>",
                            mutation_scale=8, color=RED, lw=1.4, linestyle=(0, (3, 2)),
                            zorder=3, shrinkA=8, shrinkB=8))
a.text(7.4, 4.2, "soft pulls between per-camera proxies;\na wrong link corrupts "
                 "ONE proxy pair only", fontsize=7.6, color=INK2, ha="center",
       va="top", linespacing=1.4)
a.text(5.0, 1.3, "single-seed collapse rate: 1/5 (Cluster-Contrast bootstrap failed at "
                 "-14 pts);\nCAP seeds ensemble to 0.749", fontsize=7.2, color=MUTED,
       ha="center", linespacing=1.4)

# ----- (b) pairwise vs transitive ----- #
b = axes[1]
b.text(0.1, 9.6, "(b) Pairwise links - never transitive merges", fontsize=10.5,
       fontweight="bold", color=INK)
nodes = {"A1": (1.4, 7.4), "A2": (1.4, 5.6), "B1": (4.0, 7.4), "B2": (4.0, 5.6)}
for k, (x, y) in nodes.items():
    col = BLUE if k in ("A1", "B1") else RED
    b.add_patch(Circle((x, y), 0.32, facecolor="white", edgecolor=col, lw=1.8, zorder=4))
    b.text(x, y, k, fontsize=7, ha="center", va="center", zorder=5, color=col)
b.add_patch(FancyArrowPatch(nodes["A1"], nodes["B1"], arrowstyle="-", color=AQUA,
                            lw=1.6, zorder=3, shrinkA=8, shrinkB=8))
b.add_patch(FancyArrowPatch(nodes["A2"], nodes["B1"], arrowstyle="-", color=RED,
                            lw=1.6, linestyle=(0, (3, 2)), zorder=3, shrinkA=8, shrinkB=8))
b.text(2.7, 8.3, "correct link (52-60%)", fontsize=6.8, color=AQUA, ha="center")
b.text(2.0, 6.4, "wrong link", fontsize=6.8, color=RED, ha="center", rotation=35)
b.add_patch(FancyBboxPatch((0.6, 4.6), 4.2, 3.8, boxstyle="round,pad=0.15",
                           facecolor="none", edgecolor=RED, lw=1.4,
                           linestyle=(0, (4, 3)), zorder=2))
b.text(2.5, 4.0, "transitive union-find:\none wrong edge fuses all;\npair precision "
                 "60% -> 17%", fontsize=7.4, color=RED, ha="center",
       va="top", linespacing=1.4)
nodesR = {"A1": (6.6, 7.4), "A2": (6.6, 5.6), "B1": (9.2, 7.4), "B2": (9.2, 5.6)}
for k, (x, y) in nodesR.items():
    col = BLUE if k in ("A1", "B1") else RED
    b.add_patch(Circle((x, y), 0.32, facecolor="white", edgecolor=col, lw=1.8, zorder=4))
    b.text(x, y, k, fontsize=7, ha="center", va="center", zorder=5, color=col)
b.add_patch(FancyArrowPatch(nodesR["A1"], nodesR["B1"], arrowstyle="<|-|>",
                            mutation_scale=8, color=AQUA, lw=1.6, zorder=3,
                            shrinkA=8, shrinkB=8))
b.add_patch(FancyArrowPatch(nodesR["A2"], nodesR["B1"], arrowstyle="<|-|>",
                            mutation_scale=8, color=RED, lw=1.2,
                            linestyle=(0, (3, 2)), zorder=3, shrinkA=8, shrinkB=8))
b.text(7.9, 4.0, "pairwise pull:\neach link acts alone,\nconfidence-weighted ->\n"
                 "keeps full precision", fontsize=7.4, color=INK2, ha="center",
       va="top", linespacing=1.4)
b.text(5.0, 1.3, "k=2 links: 287 @ 52% -> students 0.81 +/- 0.03;\ntransitive variant "
                 "regressed to 0.56", fontsize=7.2, color=MUTED, ha="center",
       linespacing=1.4)

# ----- (c) topology signals ----- #
c = axes[2]
c.text(0.1, 9.6, "(c) Sparse-overlap topology as free supervision", fontsize=10.5,
       fontweight="bold", color=INK)
pos = {"66.1": (1.6, 7.6), "66.3": (3.4, 8.3), "66.33": (5.2, 7.6),
       "66.128": (2.2, 5.8), "66.133": (4.6, 5.8), "66.139": (3.4, 4.6),
       "66.130": (6.6, 6.3)}
# measured overlapping pairs (same-instant same-cow rate >= 2%; see ch3)
overlap_edges = [("66.1", "66.133"), ("66.1", "66.139"), ("66.1", "66.3"),
                 ("66.128", "66.133"), ("66.130", "66.139"),
                 ("66.133", "66.139"), ("66.3", "66.33")]
from itertools import combinations as _comb
_ov = {frozenset(e) for e in overlap_edges}
nonoverlap_edges = [tuple(sorted(p)) for p in _comb(sorted(pos), 2)
                    if frozenset(p) not in _ov]
assert len(overlap_edges) == 7 and len(nonoverlap_edges) == 14
for a1, a2 in nonoverlap_edges:
    c.add_patch(FancyArrowPatch(pos[a1], pos[a2], arrowstyle="-", color=RED,
                                lw=0.8, alpha=0.45, linestyle=(0, (3, 2)),
                                zorder=1, shrinkA=7, shrinkB=7))
for a1, a2 in overlap_edges:
    c.add_patch(FancyArrowPatch(pos[a1], pos[a2], arrowstyle="-", color=AQUA,
                                lw=1.8, zorder=2, shrinkA=7, shrinkB=7))
for k, (x, y) in pos.items():
    col = YELLOW if k == "66.130" else BLUE
    c.add_patch(Circle((x, y), 0.34, facecolor="white", edgecolor=col, lw=1.8, zorder=4))
    c.text(x, y - 0.62, k, fontsize=6.2, ha="center", color=MUTED, zorder=5)
c.text(6.6, 7.2, "oblique,\nheld out", fontsize=6.4, color="#9a6a00", ha="center",
       va="bottom", linespacing=1.2)
c.text(1.2, 2.9, "solid: the 7 of 21 pairs that overlap (measured):\n"
                 "co-occurring crops -> mining candidates",
       fontsize=7.4, color=AQUA, ha="left", va="top", linespacing=1.4)
c.text(1.2, 1.5, "dashed: all 14 never co-observing pairs: same instant\n"
                 "=> different cows (cannot-link, 99.9% reliable)",
       fontsize=7.4, color=RED, ha="left", va="top", linespacing=1.4)

fig.tight_layout()
fig.savefig(f"{OUT}/fig7_mechanisms_en_v4.png", facecolor="white",
            bbox_inches="tight")
plt.close(fig)

print("saved: fig6_pipeline_en_v3.png, fig7_mechanisms_en_v3.png")
