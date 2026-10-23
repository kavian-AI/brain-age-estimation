#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
z = np.load(f"{OUT}/exp8_rate.npz")
df = pd.read_csv(f"{OUT}/exp8_rate.csv")
p7 = np.load(f"{OUT}/exp7_progression.npz")

fig = plt.figure(figsize=(15, 4.9))

# ---- Panel A: the C-index ladder (exp8)
ax = fig.add_subplot(1, 3, 1)
names = ["direct\nsnapshot", "baseline\ngap", "brain-aging\nRATE", "direct\n+ rate"]
vals = [float(z["c_d"]), float(z["c_b"]), float(z["c_r"]), float(z["c_dr"])]
cols = ["#2b5d8a", "#8a8a8a", "#c0504d", "#5a9367"]
ax.bar(range(4), vals, color=cols, width=.66)
ax.axhline(vals[0], color="#2b5d8a", ls="--", lw=1.2, alpha=.8)
ax.set_ylim(0.72, 0.84)
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("C-index (predict conversion)")
for i, v in enumerate(vals):
    ax.text(i, v + .002, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.annotate(f"+{vals[3]-vals[0]:.3f}", xy=(3, vals[3]), xytext=(2.4, 0.832),
            fontsize=10, color="#3d6b48", fontweight="bold")
ax.set_title("A. Rate of brain aging ADDS to\nthe direct snapshot", fontsize=11)

# ---- Panel B: forest of hazard ratios
ax = fig.add_subplot(1, 3, 2)
rows = [("direct snapshot", 1.51, 1.22, 1.87, "#2b5d8a"),
        ("baseline gap", 1.36, 1.09, 1.69, "#8a8a8a"),
        ("brain-aging RATE", 1.84, 1.62, 2.08, "#c0504d")]
for i, (lab, hr, lo, hi, c) in enumerate(rows):
    ax.plot([lo, hi], [i, i], color=c, lw=2.4)
    ax.plot(hr, i, "o", color=c, ms=9)
    ax.text(hi + 0.04, i, f"HR={hr:.2f}", va="center", fontsize=9)
ax.axvline(1, color="k", ls=":", lw=1)
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
ax.set_xlim(0.9, 2.4); ax.set_xlabel("hazard ratio (per SD)")
ax.set_title("B. Rate is the STRONGEST predictor\n(all in one model)", fontsize=11)

# ---- Panel C: rate distribution, converters vs not
ax = fig.add_subplot(1, 3, 3)
bins = np.linspace(-1, 2.5, 30)
ax.hist(df[df.event == 0].rate, bins=bins, color="#4a7fb0", alpha=.7,
        density=True, label=f"non-converters (n={(df.event==0).sum()})")
ax.hist(df[df.event == 1].rate, bins=bins, color="#c0504d", alpha=.7,
        density=True, label=f"converters (n={(df.event==1).sum()})")
ax.axvline(df[df.event == 0].rate.mean(), color="#2b5d8a", lw=1.5, ls="--")
ax.axvline(df[df.event == 1].rate.mean(), color="#8c3b38", lw=1.5, ls="--")
ax.set_xlabel("brain-aging rate (gap-years per year)")
ax.set_ylabel("density")
ax.set_title("C. Converters age ~2x faster\n(0.97 vs 0.51 gap-yr/yr)", fontsize=11)
ax.legend(fontsize=8.5)

fig.suptitle("The thing the snapshot model cannot do: the RATE of brain aging  "
             "(ADNI, n=1100 with ≥3 scans, 160 later conversions)",
             fontsize=12.5, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{OUT}/fig10_rate.png", dpi=165)
print("saved")
