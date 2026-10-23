#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
z = np.load(f"{OUT}/exp5_agreement.npz")
direct, gap, y, flag = z["direct"], z["gap"], z["y"], z["flagged"].astype(bool)

fig = plt.figure(figsize=(14, 5.2))

# -------- Panel A: the requested histogram
ax = fig.add_subplot(1, 3, 1)
bins = np.linspace(-15, 30, 34)
ax.hist(gap[flag], bins=bins, color="#c0504d", alpha=.75,
        label=f"flagged DISEASED (n={flag.sum()})")
ax.hist(gap[~flag], bins=bins, color="#4a7fb0", alpha=.65,
        label=f"flagged HEALTHY (n={(~flag).sum()})")
ax.axvline(0, color="k", lw=.9, ls=":")
ax.set_xlabel("brain-age gap (years)"); ax.set_ylabel("count")
ax.set_title("A. Brain-age gap of people the\ndirect model flags", fontsize=11)
ax.legend(fontsize=8.5)

# -------- Panel B: among flagged-diseased, correct vs false alarm
ax = fig.add_subplot(1, 3, 2)
tp = flag & (y == 1)
fp = flag & (y == 0)
ax.hist(gap[tp], bins=bins, color="#8c3b38", alpha=.8,
        label=f"correct: true AD (n={tp.sum()})")
ax.hist(gap[fp], bins=bins, color="#e0a458", alpha=.85,
        label=f"false alarm: true CN (n={fp.sum()})")
ax.axvline(0, color="k", lw=.9, ls=":")
thr = np.percentile(gap[y == 0], 90)
ax.axvline(thr, color="#2b5d8a", lw=1.4, ls="--")
ax.text(thr + .4, ax.get_ylim()[1]*.9, "CN 90th pct", color="#2b5d8a", fontsize=8)
ax.set_xlabel("brain-age gap (years)"); ax.set_ylabel("count")
ax.set_title("B. Among flagged-diseased:\ndo brains look old?", fontsize=11)
ax.legend(fontsize=8.5)

# -------- Panel C: scatter, the two models against each other
ax = fig.add_subplot(1, 3, 3)
ax.scatter(direct[y == 0], gap[y == 0], s=10, alpha=.4, color="#4a7fb0", label="true CN")
ax.scatter(direct[y == 1], gap[y == 1], s=10, alpha=.5, color="#c0504d", label="true AD")
ax.axvline(0.5, color="k", lw=.9, ls=":")
ax.axhline(0, color="k", lw=.9, ls=":")
r = np.corrcoef(direct, gap)[0, 1]
ax.set_xlabel("direct model  P(disease)")
ax.set_ylabel("brain-age gap (years)")
ax.set_title(f"C. Do the two models agree?\nr = {r:.2f}", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left")
# highlight the interesting quadrant: flagged but age-normal
ax.axhspan(-15, thr, xmin=0.5, xmax=1.0, color="#5a9367", alpha=.10)
ax.text(0.62, -12, "flagged but\nage-normal\n(21%)", fontsize=8, color="#3d6b48")

fig.suptitle("Direct disease model vs brain-age gap  —  ADNI, AD vs CN, out-of-fold",
             fontsize=13, y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f"{OUT}/fig8_agreement.png", dpi=170)
print("saved")
