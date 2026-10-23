#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Explanatory figures: what AUC means, what the gap is, scalar vs profile,
the landmark design, and the site-confound trap."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from sklearn.linear_model import LinearRegression

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
SHORT = ["Hipp", "Ento", "Fusi", "MidT", "Vent", "Whole"]
rng = np.random.default_rng(3)

# ============================== FIGURE A: what AUC means
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, (auc_target, title) in zip(axes, [
        (0.50, "AUC = 0.50\nuseless (coin flip)"),
        (0.70, "AUC = 0.70\nlike our brain-age gap"),
        (0.94, "AUC = 0.94\nlike volumes+age for AD")]):
    sep = {0.50: 0.0, 0.70: 0.74, 0.94: 2.2}[auc_target]
    h = rng.normal(0, 1, 4000)
    p = rng.normal(sep, 1, 4000)
    bins = np.linspace(-4, 6, 60)
    ax.hist(h, bins=bins, alpha=.6, color="#4a7fb0", density=True, label="healthy")
    ax.hist(p, bins=bins, alpha=.6, color="#c0504d", density=True, label="patient")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("model score"); ax.set_yticks([])
    ax.legend(fontsize=8.5)
axes[0].set_ylabel("how many people")
fig.suptitle("FIGURE A — What AUC means: how much the two groups overlap",
             fontsize=13, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f"{OUT}/figA_auc.png", dpi=160)
plt.close()

# ============================== FIGURE B: how the gap is built
d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
a = d.dropna(subset=V + ["ICV", "AGE", "DX", "PTGENDER"]).sort_values("Years_bl")
a = a.groupby("RID").first().reset_index()
cn = a[a.DX == "CN"]; ad = a[a.DX == "Dementia"]


def feats(df):
    return np.column_stack([df[V].values, df.ICV.values,
                            (df.PTGENDER == "Male").astype(int).values])


m = LinearRegression().fit(feats(cn), cn.AGE.values)
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

ax = axes[0]
p_cn = m.predict(feats(cn))
ax.scatter(cn.AGE, p_cn, s=8, alpha=.35, color="#4a7fb0")
lim = [55, 95]
ax.plot(lim, lim, "k--", lw=1.2)
ax.set(xlim=lim, ylim=lim, xlabel="real age", ylabel="predicted age",
       title="1. Train on HEALTHY only\n(model learns normal ageing)")

ax = axes[1]
p_ad = m.predict(feats(ad))
ax.scatter(cn.AGE, p_cn, s=8, alpha=.2, color="#4a7fb0", label="healthy")
ax.scatter(ad.AGE, p_ad, s=10, alpha=.5, color="#c0504d", label="Alzheimer's")
ax.plot(lim, lim, "k--", lw=1.2)
ax.set(xlim=lim, ylim=lim, xlabel="real age", ylabel="predicted age",
       title="2. Apply to PATIENTS\n(they sit ABOVE the line)")
ax.legend(fontsize=8.5)
i = np.argmax(p_ad - ad.AGE.values)
ax.annotate("", xy=(ad.AGE.values[i], p_ad[i]), xytext=(ad.AGE.values[i], ad.AGE.values[i]),
            arrowprops=dict(arrowstyle="<->", color="#8c3b38", lw=2))
ax.text(ad.AGE.values[i] + 1, (p_ad[i] + ad.AGE.values[i]) / 2, "GAP",
        color="#8c3b38", fontweight="bold", fontsize=10)

ax = axes[2]
for nm, df, c in [("CN", cn, "#4a7fb0"), ("MCI", a[a.DX == "MCI"], "#e0a458"),
                  ("AD", ad, "#c0504d")]:
    g = m.predict(feats(df)) - df.AGE.values
    ax.hist(g, bins=np.linspace(-25, 35, 45), alpha=.6, color=c, density=True,
            label=f"{nm} (mean {g.mean():+.1f}y)")
ax.axvline(0, color="k", ls=":", lw=1)
ax.set(xlabel="brain age gap (years)", title="3. The gap grows with disease")
ax.set_yticks([]); ax.legend(fontsize=8.5)
fig.suptitle("FIGURE B — How the brain-age gap is produced (real ADNI data)",
             fontsize=13, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{OUT}/figB_gap.png", dpi=160)
plt.close()

# ============================== FIGURE C: scalar vs profile
G = np.zeros((len(a), len(V)))
for r, c in enumerate(V):
    Xh = np.column_stack([cn[c].values, cn.ICV.values,
                          (cn.PTGENDER == "Male").astype(int).values])
    mm = LinearRegression().fit(Xh, cn.AGE.values)
    bb = LinearRegression().fit(cn.AGE.values.reshape(-1, 1),
                                mm.predict(Xh) - cn.AGE.values)
    X = np.column_stack([a[c].values, a.ICV.values,
                         (a.PTGENDER == "Male").astype(int).values])
    G[:, r] = (mm.predict(X) - a.AGE.values) - bb.predict(a.AGE.values.reshape(-1, 1))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
ax = axes[0]
for nm, c in [("CN", "#4a7fb0"), ("MCI", "#e0a458"), ("Dementia", "#c0504d")]:
    mask = (a.DX == nm).values
    ax.bar(nm, G[mask].mean(), color=c, width=.55)
    ax.text(nm, G[mask].mean() + .15, f"{G[mask].mean():+.1f}", ha="center",
            fontweight="bold")
ax.set_ylabel("one number (years)")
ax.set_title("SCALAR: the whole brain squeezed\ninto ONE number", fontsize=11)

ax = axes[1]
x = np.arange(len(V)); w = .26
for k, (nm, c) in enumerate([("CN", "#4a7fb0"), ("MCI", "#e0a458"),
                             ("Dementia", "#c0504d")]):
    mask = (a.DX == nm).values
    ax.bar(x + (k - 1) * w, G[mask].mean(0), w, color=c, label=nm)
ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=9)
ax.axhline(0, color="k", lw=.8, ls=":")
ax.set_ylabel("gap per region (years)")
ax.set_title("PROFILE: one number PER REGION\n(hippocampus is worst)", fontsize=11)
ax.legend(fontsize=8.5)
fig.suptitle("FIGURE C — Scalar vs Profile: why compressing loses information",
             fontsize=13, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f"{OUT}/figC_profile.png", dpi=160)
plt.close()

# ============================== FIGURE D: landmark design
fig, ax = plt.subplots(figsize=(12.5, 4.4))
ax.set_xlim(-0.5, 10); ax.set_ylim(0, 4.2); ax.axis("off")
ax.add_patch(Rectangle((0, 0.4), 4.2, 3.4, fc="#eef3f8", ec="#2b5d8a", lw=1.5))
ax.add_patch(Rectangle((4.6, 0.4), 5.0, 3.4, fc="#faf0ef", ec="#c0504d", lw=1.5))
ax.text(2.1, 3.55, "OBSERVATION WINDOW", ha="center", fontsize=11,
        fontweight="bold", color="#2b5d8a")
ax.text(7.1, 3.55, "PREDICTION WINDOW", ha="center", fontsize=11,
        fontweight="bold", color="#c0504d")
for i, xx in enumerate([0.6, 1.9, 3.2]):
    ax.plot(xx, 2.6, "o", ms=14, color="#2b5d8a")
    ax.text(xx, 2.15, f"scan {i+1}", ha="center", fontsize=9)
ax.plot([0.6, 3.2], [2.6, 2.6], color="#2b5d8a", lw=2)
ax.text(1.9, 1.5, "measure the SLOPE of the gap\n= rate of brain ageing",
        ha="center", fontsize=10, style="italic", color="#2b5d8a")
ax.add_patch(FancyArrowPatch((4.2, 2.6), (4.6, 2.6), arrowstyle="-|>",
                             mutation_scale=20, lw=2, color="#333"))
ax.text(7.1, 2.6, "does the person convert\nto dementia AFTER this point?",
        ha="center", fontsize=10.5, color="#c0504d")
ax.text(7.1, 1.4, "predicted using ONLY the past\n→ no data leakage",
        ha="center", fontsize=9.5, style="italic", color="#8c3b38")
ax.text(4.4, 0.1, "landmark", ha="center", fontsize=9, fontweight="bold")
fig.suptitle("FIGURE D — The 'landmark' design used in experiment 8", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/figD_landmark.png", dpi=160)
plt.close()

# ============================== FIGURE E: the site trap
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
ax = axes[0]
vals = [0.942, 0.925]
ax.bar(["healthy ABIDE\nvs\nhealthy ADHD200", "autism\nvs\nADHD"], vals,
       color=["#8c3b38", "#c0504d"], width=.55)
for i, v in enumerate(vals):
    ax.text(i, v + .01, f"{v:.3f}", ha="center", fontweight="bold")
ax.axhline(.5, color="k", ls=":", lw=1)
ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUC")
ax.set_title("BEFORE: both ~0.93!\nBut the left bar should be 0.5\n"
             "→ the model reads the SCANNER", fontsize=10.5)
ax = axes[1]
vals2 = [0.702, 0.624]
ax.bar(["healthy vs healthy", "autism vs ADHD"], vals2,
       color=["#8c3b38", "#5a9367"], width=.55)
for i, v in enumerate(vals2):
    ax.text(i, v + .01, f"{v:.3f}", ha="center", fontweight="bold")
ax.axhline(.5, color="k", ls=":", lw=1)
ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUC")
ax.set_title("AFTER ComBat: the 'discovery'\ncollapsed 0.925 → 0.624\n"
             "and 0.70 site signal REMAINS", fontsize=10.5)
fig.suptitle("FIGURE E — The trap that nearly produced a false discovery",
             fontsize=13, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.91])
plt.savefig(f"{OUT}/figE_confound.png", dpi=160)
plt.close()

print("saved figA..figE")
