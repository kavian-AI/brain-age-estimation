#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure for Report 4, section 3: why PCA on raw pixels fails.

Two panels, because the argument has two halves:

  A  the same axial level in four different subjects. If the slices were
     registered these would be anatomically aligned; they are not, and the
     skull is still present. Centroids are marked so the misalignment is
     visible rather than asserted.

  B  what the principal components actually encode. Each component is
     correlated with head geometry and with age. Geometry wins on every early
     component, which is the failure stated numerically.

Labels are in Persian (reshaped + bidi) to match the report.
"""

import io
import os
import zipfile

import arabic_reshaper
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bidi.algorithm import get_display
from PIL import Image
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
ZIP = os.path.join(BASE, "Image data", "archive.zip")
OUT = os.path.join(BASE, "pilot")
NL, H, W = 13, 40, 80
BG = 0.25
BLUE, RED, GREY = "#2c6fbb", "#c0504d", "#8a8a8a"

plt.rcParams["font.family"] = "Tahoma"


def fa(t):
    """Shape and reorder Persian text so matplotlib renders it correctly."""
    return get_display(arabic_reshaper.reshape(t))


def sample_slices(n=4, level=130):
    """One raw slice at the SAME axial index from n different subjects."""
    idx = pd.read_csv(os.path.join(OUT, "images_index.csv"))
    sel = idx[idx.slice == level].drop_duplicates("subject").head(n)
    out = []
    with zipfile.ZipFile(ZIP) as z:
        for _, r in sel.iterrows():
            im = Image.open(io.BytesIO(z.read(r.member))).convert("L")
            out.append((r.subject, np.asarray(im, dtype=np.float32)))
    return out


def geometry_and_pcs():
    c = np.load(os.path.join(OUT, "oasis_img_cache.npz"), allow_pickle=True)
    acq = pd.DataFrame(dict(subject=c["subject"]))
    order = acq.groupby("subject").indices
    subs = sorted(order)
    Xs = np.stack([c["X"][order[s]].mean(0) for s in subs])

    idx = pd.read_csv(os.path.join(OUT, "images_index.csv"))
    meta = idx.groupby("subject").first().reset_index()[["subject", "Age"]]
    age = pd.DataFrame(dict(subject=subs)).merge(meta, on="subject",
                                                 how="left").Age.values

    V = Xs.reshape(-1, NL, H, W)[:, NL // 2]
    cx, area = [], []
    for im in V:
        m = im > BG
        if m.sum() < 20:
            cx.append(np.nan)
            area.append(np.nan)
            continue
        _, x = ndimage.center_of_mass(m)
        cx.append(x)
        area.append(m.sum())
    cx, area = np.array(cx), np.array(area, dtype=float)

    Z = PCA(n_components=6, random_state=0).fit_transform(
        StandardScaler().fit_transform(Xs))
    ok = np.isfinite(cx) & np.isfinite(age)
    rows = []
    for i in range(6):
        rows.append([abs(np.corrcoef(Z[ok, i], cx[ok])[0, 1]),
                     abs(np.corrcoef(Z[ok, i], area[ok])[0, 1]),
                     abs(np.corrcoef(Z[ok, i], age[ok])[0, 1])])
    return np.array(rows), cx, area


def main():
    slices = sample_slices()
    R, cx, area = geometry_and_pcs()

    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.5], hspace=0.30,
                          wspace=0.28, left=0.06, right=0.97,
                          top=0.90, bottom=0.10)

    # ---- A: the same anatomical level in four subjects
    for k, (sub, im) in enumerate(slices):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(im, cmap="gray")
        m = im > im.max() * 0.12
        if m.sum() > 50:
            cy0, cx0 = ndimage.center_of_mass(m)
            ax.axvline(cx0, color=RED, lw=1.0, ls="--", alpha=.9)
            ax.plot(cx0, cy0, "o", color=RED, ms=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(sub, fontsize=13.6, color="#444")
    fig.text(0.06, 0.945, fa("الف) یک سطح محوری یکسان در چهار فرد مختلف — "
                             "سرها هم‌تراز نیستند و جمجمه برداشته نشده است"),
             fontsize=15.5, ha="left")

    # ---- B: what the components encode
    ax = fig.add_subplot(gs[1, :2])
    x = np.arange(6)
    ax.bar(x - .27, R[:, 0], .27, color=RED, label=fa("موقعیت افقی سر"))
    ax.bar(x, R[:, 1], .27, color=GREY, label=fa("اندازه‌ی سر"))
    ax.bar(x + .27, R[:, 2], .27, color=BLUE, label=fa("سن فرد"))
    ax.set_xticks(x)
    PD = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
    ax.set_xticklabels([fa(f"مؤلفه {i+1}".translate(PD)) for i in range(6)],
                       fontsize=15.3)
    ax.set_ylabel(fa("قدر مطلق همبستگی"), fontsize=17.0)
    ax.set_ylim(0, max(0.9, R.max() * 1.2))
    ax.legend(fontsize=15.3, frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=.25)
    ax.set_title(fa("ب) مؤلفه‌ها هندسه‌ی سر را کد می‌کنند، نه سن را"),
                 fontsize=14.5, loc="left", pad=8)

    # ---- C: the consequence, in AUC
    ax = fig.add_subplot(gs[1, 2:])
    names = [fa("۸۰ مؤلفه‌ی اصلی\nروی ۴۱٬۶۰۰ پیکسل"),
             fa("یک ویژگی دست‌ساز:\nنسبت پیکسل تیره"),
             fa("سه عدد جدولی")]
    vals = [0.526, 0.607, 0.712]
    cols = [RED, "#e8a33d", BLUE]
    ax.bar(range(3), vals, color=cols, width=.6)
    for i, v in enumerate(vals):
        ax.text(i, v + .008, f"{v:.3f}", ha="center", fontsize=17.0)
    ax.axhline(.5, color="k", ls=":", lw=1)
    ax.text(2.45, .4875, fa("شانس"), fontsize=13.6, ha="right", color="#555")
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=14.4)
    ax.set_ylim(.48, .76)
    ax.set_ylabel("AUC", fontsize=17.0)
    ax.grid(axis="y", alpha=.25)
    ax.set_title(fa("ج) یک ویژگی ساده از هشتاد مؤلفه بهتر است"),
                 fontsize=14.5, loc="left", pad=8)

    p = os.path.join(OUT, "fig20_why_pca_fails.png")
    fig.savefig(p, dpi=150, facecolor="white")
    print("saved ->", p)
    print(f"head centroid SD across subjects: {np.nanstd(cx):.2f} px of {W}")
    print("PC correlations (|r| with cx, area, age):")
    for i, r in enumerate(R, 1):
        print(f"  PC{i}: cx {r[0]:.2f}  area {r[1]:.2f}  age {r[2]:.2f}")


if __name__ == "__main__":
    main()
