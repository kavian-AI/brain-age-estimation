#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Brain age from OASIS images with a convnet.

WHY THIS RUNS ON A GPU AND NOT LOCALLY
--------------------------------------
The local experiment applied the classic Franke-2010 recipe - downsample the
voxels, PCA, linear model - and it failed badly: disease AUC 0.53 from 80
components over 41,600 pixels, against 0.71 from three tabular numbers. The
diagnosis was not lack of signal. A single hand-made feature, the fraction of
dark (CSF) pixels inside the head, reached 0.607 on its own. The top principal
components correlated 0.42-0.58 with head centroid and head area, against a
maximum of 0.36 with age, because these slices are NOT registered to a template.
Franke's recipe assumes that registration step; without it, PCA spends its
components describing where the head sits in the frame.

A convnet is the right tool for unregistered input: convolution is translation
equivariant, so it does not have to burn capacity on head position. This is the
honest test of whether the image carries more than the volume numbers.

DESIGN
------
Model class differs between arms by necessity, so every arm is reported and the
tabular arm is computed here too, on identical folds, rather than quoted from
the earlier run.

  Q1  age regression on HEALTHY subjects only (n=266, ages 18-96)
      MAE reported on the full range and on 60+ separately. The full range
      flatters every model - separating a 20-year-old from an 80-year-old is
      not the clinical task - so the 60+ number is the one that means anything.

  Q2  disease (CDR>0 vs CDR=0), age-matched, three ways:
        gap        the brain-age gap alone, the classical pipeline
        feat+age   penultimate CNN features + chronological age, straight to
                   disease. This is the fair baseline: the gap is a function of
                   the image and age, so a model given both should not do worse.
        direct     a CNN trained on the disease label from scratch

  Q3  test-retest. Each subject has 3-4 acquisitions from the SAME session, so
      any variation in their gap is measurement noise, not biology.

LEAKAGE
-------
Splits are by SUBJECT everywhere. Each subject contributes ~4 acquisitions and
~200 slices; splitting on either would put the same brain in train and test,
which is why this dataset carries implausible published accuracies. The age
model is fitted on healthy TRAINING subjects only. Disease AUC is computed from
the raw gap rather than from a fitted classifier, so no second model is trained
on the test fold.

INTENSITY
---------
Each acquisition is divided by the 95th percentile of its foreground, not the
mean. Atrophy replaces bright tissue with dark CSF, so the foreground mean is
itself a strong predictor of disease here (AUC 0.398) and dividing by it cancels
the signal - measured locally, it drove a CSF feature from 0.607 to 0.523.
p95 tracks white matter and is close to neutral (AUC 0.512).
"""

import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

OUT = "/kaggle/working"


def find(pattern, what):
    """Locate an input by name. Kaggle's mount directory does not always match
    the dataset slug, so hard-coding /kaggle/input/<slug>/ is unreliable."""
    import glob
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        print(f"cannot find {what}. available inputs:")
        for d in sorted(glob.glob("/kaggle/input/*")):
            print("  ", d, os.listdir(d)[:6])
        sys.exit(1)
    return hits[0]


CSV = find("oasis_cross-sectional.csv", "the demographics table")
IMG = os.path.dirname(os.path.dirname(
    find("OAS1_0028_MR1_mpr-1_100.jpg", "the image folder")))
print("csv:", CSV, "\nimg:", IMG)

SLICES = list(range(100, 161, 4))        # 16 axial levels
H, W = 112, 224                          # source is 248x496, keep the 1:2 ratio
BG = 25                                  # 8-bit units, below this is background
FOLDS = 5
SEEDS = 3
EPOCHS = 40
BATCH = 16
LR = 3e-4

PAT = re.compile(r"(OAS1_\d{4})_MR(\d+)_mpr-(\d+)_(\d+)\.jpg$", re.I)
def pick_device():
    """Kaggle hands out whichever accelerator is free, and its preinstalled
    torch has no kernels for compute capability < 7.0 (the P100). A P100 does
    not fail at import, it fails on the first conv, ten minutes into the run -
    so probe it here and fall back rather than lose the run."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.nn.Conv2d(1, 1, 3).cuda()(torch.zeros(1, 1, 8, 8, device="cuda"))
        return "cuda"
    except Exception as e:
        print(f"GPU unusable ({torch.cuda.get_device_name(0)}): "
              f"{type(e).__name__}. Falling back to CPU.", flush=True)
        return "cpu"


dev = pick_device()


# ----------------------------------------------------------------- indexing

def build_index():
    rows = []
    for root, _, files in os.walk(IMG):
        for f in files:
            g = PAT.search(f)
            if g:
                rows.append(dict(path=os.path.join(root, f), subject=g.group(1),
                                 mr=int(g.group(2)), mpr=int(g.group(3)),
                                 slice=int(g.group(4))))
    df = pd.DataFrame(rows)
    cs = pd.read_csv(CSV)
    cs["subject"] = cs["ID"].str.extract(r"(OAS1_\d{4})")
    cs = cs.dropna(subset=["subject"]).drop_duplicates("subject")
    keep = ["subject", "Age", "M/F", "CDR", "nWBV", "eTIV"]
    df = df.merge(cs[keep], on="subject", how="left")
    return df[df.Age.notna()]


def load_all(idx):
    """Decode every acquisition once into a uint8 array held in RAM."""
    idx = idx[idx.slice.isin(SLICES)]
    keys, vols = [], []
    t0 = time.time()
    for n, ((s, mr, mpr), g) in enumerate(idx.groupby(["subject", "mr", "mpr"]), 1):
        g = g.sort_values("slice")
        if len(g) != len(SLICES):
            continue
        v = np.stack([np.asarray(Image.open(p).convert("L").resize(
            (W, H), Image.BILINEAR), dtype=np.float32) for p in g.path])
        fg = v[v > BG]
        if fg.size < 100:
            continue
        v = np.clip(v / np.percentile(fg, 95), 0, 2.0) * 127.5
        keys.append((s, mr, mpr))
        vols.append(v.astype(np.uint8))
        if n % 300 == 0:
            print(f"  decoded {n} acquisitions ({time.time()-t0:.0f}s)", flush=True)
    k = pd.DataFrame(keys, columns=["subject", "mr", "mpr"])
    return np.stack(vols), k


# -------------------------------------------------------------------- data

class Acq(Dataset):
    def __init__(self, X, y, train):
        self.X, self.y, self.train = X, y.astype(np.float32), train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        v = torch.from_numpy(self.X[i].astype(np.float32) / 127.5)
        if self.train:
            if np.random.rand() < 0.5:
                v = torch.flip(v, [2])                   # left-right
            ang = np.random.uniform(-8, 8)
            tx, ty = np.random.uniform(-0.06, 0.06, 2)
            sc = np.random.uniform(0.92, 1.08)
            v = affine(v, ang, tx, ty, sc)
            v = v * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)
        return v, self.y[i]


def affine(v, ang, tx, ty, sc):
    a = np.deg2rad(ang)
    m = torch.tensor([[np.cos(a) / sc, -np.sin(a) / sc, tx],
                      [np.sin(a) / sc, np.cos(a) / sc, ty]],
                     dtype=torch.float32).unsqueeze(0)
    g = F.affine_grid(m, (1,) + tuple(v.shape), align_corners=False)
    return F.grid_sample(v.unsqueeze(0), g, align_corners=False,
                         padding_mode="zeros")[0]


# ------------------------------------------------------------------- model

class Net(nn.Module):
    """Small residual convnet over the 16 slices as input channels.

    Trained from scratch: n=266 subjects is small, but each contributes ~4
    acquisitions and the augmentation is aggressive, and an ImageNet backbone
    expects 3 natural-image channels rather than 16 registered anatomical
    levels. Width is kept modest to match the sample size.
    """

    def __init__(self, cin, nout=1, width=32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(cin, w, 5, 2, 2), nn.BatchNorm2d(w),
                                  nn.ReLU(inplace=True))
        self.b1 = self._block(w, w * 2)
        self.b2 = self._block(w * 2, w * 4)
        self.b3 = self._block(w * 4, w * 8)
        self.head = nn.Linear(w * 8, nout)
        self.drop = nn.Dropout(0.3)

    @staticmethod
    def _block(a, b):
        return nn.Sequential(
            nn.Conv2d(a, b, 3, 2, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    def features(self, x):
        x = self.b3(self.b2(self.b1(self.stem(x))))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        return self.head(self.drop(self.features(x))).squeeze(-1)


def fit(X, y, epochs=EPOCHS, cls=False, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(len(SLICES)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    dl = DataLoader(Acq(X, y, True), batch_size=BATCH, shuffle=True,
                    drop_last=len(X) > BATCH, num_workers=2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, LR * 5, epochs=epochs, steps_per_epoch=max(1, len(dl)))
    lossf = nn.BCEWithLogitsLoss() if cls else nn.L1Loss()
    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
    return net


@torch.no_grad()
def predict(net, X, feats=False):
    net.eval()
    out = []
    for i in range(0, len(X), 64):
        xb = torch.from_numpy(X[i:i + 64].astype(np.float32) / 127.5).to(dev)
        out.append((net.features(xb) if feats else net(xb)).cpu().numpy())
    return np.concatenate(out)


# -------------------------------------------------------------------- main

def folds_of(subs, seed):
    u = np.unique(subs)
    r = np.random.RandomState(seed)
    f = dict(zip(u, r.permutation(len(u)) % FOLDS))
    return np.array([f[s] for s in subs])


def main():
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    idx = build_index()
    print(f"slices indexed: {len(idx):,}  subjects: {idx.subject.nunique()}")

    X, key = load_all(idx)
    print(f"acquisitions: {X.shape[0]}, volume {X.shape[1:]}, "
          f"{X.nbytes/1e9:.2f} GB uint8")

    meta = idx.groupby("subject").first().reset_index()[
        ["subject", "Age", "M/F", "CDR", "nWBV", "eTIV"]]
    key = key.merge(meta, on="subject", how="left")
    key["healthy"] = (key.CDR.fillna(0) == 0)
    subs = key.subject.values
    age = key.Age.values.astype(np.float32)

    sm = meta.copy()
    sm["healthy"] = sm.CDR.fillna(0) == 0
    lo = sm.Age[~sm.healthy].min()
    print(f"subjects: healthy {int(sm.healthy.sum())}, "
          f"impaired {int((~sm.healthy).sum())}, age-match cut {lo:.0f}")

    res = {}
    oof_gap = np.full((SEEDS, len(key)), np.nan)
    oof_feat = np.zeros((SEEDS, len(key), 256))
    for seed in range(SEEDS):
        f = folds_of(subs, seed)
        for k in range(FOLDS):
            tr = np.where((f != k) & key.healthy.values)[0]
            te = np.where(f == k)[0]
            net = fit(X[tr], age[tr], seed=seed * 10 + k)
            p = predict(net, X[te])
            oof_feat[seed, te] = predict(net, X[te], feats=True)
            # bias correction fitted on healthy training acquisitions
            ptr = predict(net, X[tr])
            b = LinearRegression().fit(age[tr].reshape(-1, 1), ptr - age[tr])
            oof_gap[seed, te] = (p - age[te]) - b.predict(age[te].reshape(-1, 1))
            print(f"  seed {seed} fold {k}: "
                  f"test MAE {np.abs(p - age[te]).mean():.2f} y", flush=True)

    # ---- collapse acquisitions to subjects
    key["fold0"] = folds_of(subs, 0)
    g = pd.DataFrame(dict(subject=subs))
    for s in range(SEEDS):
        g[f"gap{s}"] = oof_gap[s]
    per = g.groupby("subject").mean()
    sm = sm.set_index("subject").loc[per.index].reset_index()
    per = per.reset_index()

    print("\n" + "=" * 70)
    print("Q1  AGE PREDICTION IN HEALTHY SUBJECTS")
    print("=" * 70)
    hz = sm.healthy.values
    old = sm.Age.values >= 60
    for s in range(SEEDS):
        err = np.abs(per[f"gap{s}"].values)[hz]
        print(f"  seed {s}: MAE(18-96) {err.mean():.2f} y   "
              f"MAE(60+) {err[old[hz]].mean():.2f} y")

    print("\n" + "=" * 70)
    print("Q2  DISEASE, AGE-MATCHED")
    print("=" * 70)
    sel = sm.Age.values >= lo
    y = (~sm.healthy.values)[sel].astype(int)
    print(f"  n={sel.sum()} (healthy {int((y==0).sum())}, impaired {int(y.sum())})")

    a = [roc_auc_score(y, per[f"gap{s}"].values[sel]) for s in range(SEEDS)]
    res["gap"] = np.array(a)
    print(f"\n  CNN brain-age gap        AUC = {np.mean(a):.3f} "
          f"({', '.join(f'{v:.3f}' for v in a)})")

    # fair baseline: CNN features + chronological age
    fa = []
    for s in range(SEEDS):
        Fz = pd.DataFrame(oof_feat[s]).assign(subject=subs).groupby(
            "subject").mean().loc[per.subject].values
        D = np.column_stack([Fz, sm.Age.values])[sel]
        f5 = folds_of(sm.subject.values[sel], s)
        o = np.zeros(len(y))
        for k in range(FOLDS):
            tr, te = f5 != k, f5 == k
            sc = StandardScaler().fit(D[tr])
            c = LogisticRegression(max_iter=3000, C=0.05).fit(sc.transform(D[tr]), y[tr])
            o[te] = c.predict_proba(sc.transform(D[te]))[:, 1]
        fa.append(roc_auc_score(y, o))
    res["feat_age"] = np.array(fa)
    print(f"  CNN features + age       AUC = {np.mean(fa):.3f} "
          f"({', '.join(f'{v:.3f}' for v in fa)})")

    # direct CNN on the disease label
    da = []
    ysub = (~sm.healthy.values).astype(np.float32)
    keep = np.isin(subs, sm.subject.values[sel])
    for s in range(SEEDS):
        f2 = folds_of(subs[keep], s)
        yacq = pd.Series(ysub, index=sm.subject.values).loc[subs[keep]].values
        o = np.zeros(keep.sum())
        Xk = X[keep]
        for k in range(FOLDS):
            tr, te = np.where(f2 != k)[0], np.where(f2 == k)[0]
            net = fit(Xk[tr], yacq[tr], epochs=25, cls=True, seed=s * 10 + k)
            o[te] = predict(net, Xk[te])
        dfp = pd.DataFrame(dict(subject=subs[keep], p=o)).groupby("subject").mean()
        da.append(roc_auc_score(
            pd.Series(ysub, index=sm.subject.values).loc[dfp.index].values,
            dfp.p.values))
    res["direct"] = np.array(da)
    print(f"  CNN direct to disease    AUC = {np.mean(da):.3f} "
          f"({', '.join(f'{v:.3f}' for v in da)})")

    # tabular arm, same folds, computed here rather than quoted
    T = np.column_stack([sm.nWBV.values, sm.eTIV.values,
                         (sm["M/F"] == "M").astype(int).values, sm.Age.values])[sel]
    ta = []
    for s in range(SEEDS):
        f5 = folds_of(sm.subject.values[sel], s)
        o = np.zeros(len(y))
        for k in range(FOLDS):
            tr, te = f5 != k, f5 == k
            sc = StandardScaler().fit(T[tr])
            c = LogisticRegression(max_iter=3000).fit(sc.transform(T[tr]), y[tr])
            o[te] = c.predict_proba(sc.transform(T[te]))[:, 1]
        ta.append(roc_auc_score(y, o))
    res["tabular"] = np.array(ta)
    print(f"  nWBV+eTIV+sex+age        AUC = {np.mean(ta):.3f} "
          f"({', '.join(f'{v:.3f}' for v in ta)})   <- the number to beat")

    print("\n" + "=" * 70)
    print("Q3  TEST-RETEST WITHIN ONE SESSION")
    print("=" * 70)
    gg = pd.DataFrame(dict(subject=subs, gap=oof_gap.mean(0)))
    gg = gg[np.isin(gg.subject, sm.subject.values[hz])]
    st = gg.groupby("subject").gap.agg(["mean", "std", "count"])
    st = st[st["count"] >= 2]
    win, bet = st["std"].mean(), st["mean"].std()
    icc = (bet**2 - win**2 / st["count"].mean()) / bet**2
    print(f"  {len(st)} healthy subjects, {st['count'].mean():.1f} acquisitions each")
    print(f"  within-subject SD of the gap: {win:.2f} y")
    print(f"  between-subject SD:           {bet:.2f} y")
    print(f"  reliability:                  {icc:.3f}")

    np.savez(f"{OUT}/results.npz", **res, gap=per.values.astype(float),
             retest=np.array([win, bet, icc]))
    per.assign(Age=sm.Age.values, CDR=sm.CDR.values,
               healthy=sm.healthy.values).to_csv(f"{OUT}/subject_gaps.csv",
                                                 index=False)
    print("\nsaved results.npz and subject_gaps.csv")


if __name__ == "__main__":
    main()
