#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 6: the two-stage pipeline (what the model actually does)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"

fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.2)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color="#1a1a1a")


def arrow(x1, y1, x2, y2, txt="", col="#333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=17,
                                 lw=1.8, color=col))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.22, txt, ha="center",
                fontsize=8.5, color=col, style="italic")


# ---- STAGE 1 banner
ax.add_patch(FancyBboxPatch((0.3, 3.4), 6.1, 2.55,
             boxstyle="round,pad=0.05,rounding_size=0.1",
             fc="#eef3f8", ec="#2b5d8a", lw=1.4, ls="--"))
ax.text(3.35, 5.72, "STAGE 1  —  brain age model   (REGRESSION)",
        ha="center", fontsize=10.5, fontweight="bold", color="#2b5d8a")
ax.text(3.35, 5.42, "trained on HEALTHY people only", ha="center",
        fontsize=8.5, color="#2b5d8a", style="italic")

box(0.6, 4.35, 2.0, 0.85,
    "INPUT\nbrain volumes\n+ ICV + sex", "#ffffff", "#2b5d8a", 8.5)
box(3.15, 4.35, 1.7, 0.85, "linear\nregression", "#cfe0ef", "#2b5d8a", 9, True)
box(5.15, 4.35, 1.05, 0.85, "predicted\nage", "#ffffff", "#2b5d8a", 8.5)
arrow(2.6, 4.77, 3.15, 4.77)
arrow(4.85, 4.77, 5.15, 4.77)
ax.text(3.5, 3.72, "TARGET = chronological age", ha="center",
        fontsize=8.8, color="#8c3b38", fontweight="bold")

# ---- gap computation
box(6.7, 4.35, 1.9, 0.85, "GAP =\npredicted − real\nage", "#fff3e0", "#d98f3f", 8.5, True)
arrow(6.2, 4.77, 6.7, 4.77)

# ---- STAGE 2 banner
ax.add_patch(FancyBboxPatch((6.5, 0.35), 6.15, 2.5,
             boxstyle="round,pad=0.05,rounding_size=0.1",
             fc="#faf0ef", ec="#c0504d", lw=1.4, ls="--"))
ax.text(9.55, 2.62, "STAGE 2  —  disease model   (CLASSIFICATION)",
        ha="center", fontsize=10.5, fontweight="bold", color="#c0504d")
ax.text(9.55, 2.32, "trained on patients + controls", ha="center",
        fontsize=8.5, color="#c0504d", style="italic")

box(6.75, 1.15, 2.0, 0.85,
    "INPUT\ngap  /  regional\nprofile  /  volumes", "#ffffff", "#c0504d", 8.3)
box(9.25, 1.15, 1.7, 0.85, "logistic\nregression", "#f2d2cf", "#c0504d", 9, True)
box(11.15, 1.15, 1.25, 0.85, "P(disease)\n0 … 1", "#ffffff", "#c0504d", 8.5)
arrow(8.75, 1.57, 9.25, 1.57)
arrow(10.95, 1.57, 11.15, 1.57)
ax.text(9.6, 0.62, "TARGET = patient / control  (or  converter / stable)",
        ha="center", fontsize=8.8, color="#8c3b38", fontweight="bold")

# gap feeds stage 2
arrow(7.65, 4.35, 7.65, 2.0, "the brain-age gap\nbecomes an input", "#d98f3f")

plt.tight_layout()
plt.savefig(f"{OUT}/fig6_architecture.png", dpi=170, bbox_inches="tight")
print("fig6 saved")
