#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Brain age from ADNI 3D volumes with a convnet - the decisive image test.

WHY ADNI AND NOT OASIS
----------------------
The OASIS convnet run reached disease AUC 0.637 against 0.711 from three
tabular numbers. Two things limited it, and ADNI fixes both:

  sample     163 subjects for the disease task. Here: 549 for AD vs CN and 774
             for MCI vs CN, from 931 complete baselines.
  geometry   the OASIS slices are not in a common space, so the leading linear
             components tracked head geometry (|r| up to 0.58, against 0.36 for age).
             These ADNI tensors are registered - bounding boxes agree to within
             0.6-2.3 voxels on a 32 grid - and they are 3D rather than 13 slices.

WHAT IS BEING COMPARED
----------------------
The tabular arm is the 6 FreeSurfer volumes + ICV + sex that every earlier
experiment used, and it is recomputed here on identical folds rather than
quoted. The question is whether the image carries anything those numbers do not.

  Q1  age regression on CN subjects, then the gap
  Q2  AD vs CN and MCI vs CN:
        gap        the classical pipeline
        feat+age   penultimate CNN features + chronological age, straight to
                   disease - the fair baseline, since the gap is a function of
                   exactly those two things
        direct     CNN trained on the disease label from scratch
        finetune   CNN pretrained on age, then fine-tuned on the disease label
        tabular    volumes + ICV + sex + age
        both       tabular + CNN features
  Q3  does the image ADD to the volume numbers, or merely duplicate them?

The finetune arm tests something the OASIS run hinted at: there, the age-derived
gap (0.637) beat a CNN trained directly on the disease label (0.513), which is
what you expect when the disease sample is too small to train on but the age
task provides a large, cheaply labelled pretraining signal.

INTENSITY
---------
Volumes arrive scaled to [0,1] and are used as-is. On OASIS, rescaling by a
statistic that itself tracks atrophy (the foreground mean) cancelled the signal
- it drove a CSF feature from AUC 0.607 to 0.523. Here the foreground mean still
predicts diagnosis (AUC 0.399), i.e. the signal is intact, so leave it alone.

SPLITS
------
By SUBJECT, one baseline scan each.
"""

import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

OUT = "/kaggle/working"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
DOWN = 2                      # 128 -> 64
FOLDS, SEEDS, EPOCHS, BATCH, LR = 5, 3, 35, 8, 3e-4
FNAME = re.compile(r"^(\d{3}_S_\d{4,5})(?:_\d+)?_([A-Za-z]+\d*)\.npy$")


def pick_device():
    """Kaggle hands out whichever accelerator is free, and its torch build has
    no kernels below compute capability 7.0 (the P100). A P100 does not fail at
    import, it fails on the first conv - so probe here rather than lose the run."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.nn.Conv3d(1, 1, 3).cuda()(torch.zeros(1, 1, 8, 8, 8, device="cuda"))
        return "cuda"
    except Exception as e:
        print(f"GPU unusable ({torch.cuda.get_device_name(0)}): "
              f"{type(e).__name__} -> CPU", flush=True)
        return "cpu"


dev = pick_device()


def find(pattern, what):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        print(f"cannot find {what}; inputs are:")
        for d in sorted(glob.glob("/kaggle/input/*")):
            print("  ", d, os.listdir(d)[:5])
        sys.exit(1)
    return hits[0]


def load_data():
    csv = find("ADNIMERGE*.csv", "ADNIMERGE")
    a = pd.read_csv(csv, low_memory=False)
    lut = a[["PTID", "VISCODE", "AGE", "PTGENDER", "DX", "ICV"] + V].copy()
    lut["key"] = lut.PTID + "|" + lut.VISCODE.astype(str)
    lut = lut.drop_duplicates("key").set_index("key")

    root = os.path.dirname(find("*_S_*.npy", "the volume archive"))
    files = glob.glob(os.path.join(root, "**", "*.npy"), recursive=True)
    print(f"csv: {csv}\nvolumes on disk: {len(files)}")

    rows, X, t0 = [], [], time.time()
    for i, f in enumerate(files, 1):
        g = FNAME.match(os.path.basename(f))
        if not g:
            continue
        vis = "bl" if g.group(2) == "sc" else g.group(2)
        if re.fullmatch(r"v\d+", vis):
            continue                       # ADNI-GO/2 code, cannot map safely
        key = f"{g.group(1)}|{vis}"
        if vis != "bl" or key not in lut.index:
            continue
        r = lut.loc[key]
        if pd.isna(r.DX) or pd.isna(r.ICV) or pd.isna(r.AGE) or r[V].isna().any():
            continue
        v = np.load(f).astype(np.float32)
        if v.ndim != 3:
            continue
        s = v.shape[0] // DOWN
        v = v[:s * DOWN, :s * DOWN, :s * DOWN].reshape(
            s, DOWN, s, DOWN, s, DOWN).mean(axis=(1, 3, 5))
        X.append((np.clip(v, 0, 1) * 255).astype(np.uint8))
        rows.append(dict(PTID=g.group(1), AGE=r.AGE, DX=r.DX, ICV=r.ICV,
                         SEX=1 if r.PTGENDER == "Male" else 0,
                         **{k: r[k] for k in V}))
        if len(rows) % 200 == 0:
            print(f"  loaded {len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    m = pd.DataFrame(rows).drop_duplicates("PTID").reset_index(drop=True)
    X = np.stack(X)[pd.DataFrame(rows).drop_duplicates("PTID").index.values]
    return X, m


class Vol(Dataset):
    def __init__(self, X, y, train):
        self.X, self.y, self.train = X, y.astype(np.float32), train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        v = torch.from_numpy(self.X[i].astype(np.float32) / 255.).unsqueeze(0)
        if self.train:
            if np.random.rand() < .5:
                v = torch.flip(v, [1])                      # left-right
            sh = np.random.uniform(-.05, .05, 3)
            m = torch.eye(3, 4).unsqueeze(0)
            m[0, :, 3] = torch.tensor(sh, dtype=torch.float32)
            g = F.affine_grid(m, (1,) + tuple(v.shape), align_corners=False)
            v = F.grid_sample(v.unsqueeze(0), g, align_corners=False)[0]
            v = v * np.random.uniform(.92, 1.08)
        return v, self.y[i]


class Net3D(nn.Module):
    def __init__(self, nout=1, w=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, w, 5, 2, 2), nn.InstanceNorm3d(w),
                                  nn.ReLU(inplace=True))
        self.b1, self.b2, self.b3 = (self._blk(w, w * 2), self._blk(w * 2, w * 4),
                                     self._blk(w * 4, w * 8))
        self.drop = nn.Dropout(.4)
        self.head = nn.Linear(w * 8, nout)

    @staticmethod
    def _blk(a, b):
        return nn.Sequential(nn.Conv3d(a, b, 3, 2, 1), nn.InstanceNorm3d(b),
                             nn.ReLU(inplace=True),
                             nn.Conv3d(b, b, 3, 1, 1), nn.InstanceNorm3d(b),
                             nn.ReLU(inplace=True))

    def features(self, x):
        return F.adaptive_avg_pool3d(self.b3(self.b2(self.b1(self.stem(x)))),
                                     1).flatten(1)

    def forward(self, x):
        return self.head(self.drop(self.features(x))).squeeze(-1)


def fit(X, y, epochs=EPOCHS, cls=False, seed=0, init=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net3D().to(dev)
    if init is not None:
        sd = {k: v for k, v in init.items() if not k.startswith("head")}
        net.load_state_dict(sd, strict=False)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    dl = DataLoader(Vol(X, y, True), batch_size=BATCH, shuffle=True,
                    drop_last=len(X) > BATCH, num_workers=2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR * 5, epochs=epochs,
                                                steps_per_epoch=max(1, len(dl)))
    lossf = nn.BCEWithLogitsLoss() if cls else nn.L1Loss()
    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            lossf(net(xb.to(dev)), yb.to(dev)).backward()
            opt.step()
            sched.step()
    return net


@torch.no_grad()
def predict(net, X, feats=False):
    net.eval()
    o = []
    for i in range(0, len(X), 32):
        xb = torch.from_numpy(X[i:i + 32].astype(np.float32) / 255.).unsqueeze(1).to(dev)
        o.append((net.features(xb) if feats else net(xb)).cpu().numpy())
    return np.concatenate(o)


def folds_of(n, y, seed):
    r = np.random.RandomState(seed)
    f = np.zeros(n, int)
    for c in (0, 1):
        ix = np.where(y == c)[0]
        f[ix] = r.permutation(len(ix)) % FOLDS
    return f


def logit_cv(D, y, seed, C=.05):
    f = folds_of(len(y), y, seed)
    o = np.zeros(len(y))
    for k in range(FOLDS):
        tr, te = f != k, f == k
        s = StandardScaler().fit(D[tr])
        c = LogisticRegression(max_iter=4000, C=C).fit(s.transform(D[tr]), y[tr])
        o[te] = c.predict_proba(s.transform(D[te]))[:, 1]
    return roc_auc_score(y, o)


def run(name, X, m, pos):
    d = m[m.DX.isin(["CN"] + pos)].reset_index(drop=True)
    Xi = X[m.DX.isin(["CN"] + pos).values]
    y = d.DX.isin(pos).astype(int).values
    age = d.AGE.values.astype(float)
    tab = np.column_stack([d[V].values, d.ICV.values, d.SEX.values])
    print("\n" + "=" * 72)
    print(f"{name}   n={len(y)}  (CN {int((y==0).sum())}, {pos[0]} {int(y.sum())})")
    print("=" * 72)

    res = {k: [] for k in ("gap", "feat_age", "direct", "finetune", "tabular",
                           "both", "tab_gap")}
    for seed in range(SEEDS):
        f = folds_of(len(y), y, seed)
        gap = np.zeros(len(y))
        feat = np.zeros((len(y), 128))
        dire = np.zeros(len(y))
        fine = np.zeros(len(y))
        tgap = np.zeros(len(y))
        for k in range(FOLDS):
            tr, te = np.where(f != k)[0], np.where(f == k)[0]
            h = tr[y[tr] == 0]
            net = fit(Xi[h], age[h], seed=seed * 10 + k)
            p, ph = predict(net, Xi[te]), predict(net, Xi[h])
            b = LinearRegression().fit(age[h].reshape(-1, 1), ph - age[h])
            gap[te] = (p - age[te]) - b.predict(age[te].reshape(-1, 1))
            feat[te] = predict(net, Xi[te], feats=True)
            dire[te] = predict(fit(Xi[tr], y[tr].astype(float), epochs=25,
                                   cls=True, seed=seed * 10 + k), Xi[te])
            fine[te] = predict(fit(Xi[tr], y[tr].astype(float), epochs=18,
                                   cls=True, seed=seed * 10 + k,
                                   init=net.state_dict()), Xi[te])
            # tabular gap on the same fold
            sc = StandardScaler().fit(tab[h])
            mdl = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(tab[h]), age[h])
            bt = LinearRegression().fit(age[h].reshape(-1, 1),
                                        mdl.predict(sc.transform(tab[h])) - age[h])
            tgap[te] = (mdl.predict(sc.transform(tab[te])) - age[te]) \
                - bt.predict(age[te].reshape(-1, 1))
            print(f"  seed {seed} fold {k} done", flush=True)

        res["gap"].append(roc_auc_score(y, gap))
        res["tab_gap"].append(roc_auc_score(y, tgap))
        res["direct"].append(roc_auc_score(y, dire))
        res["finetune"].append(roc_auc_score(y, fine))
        res["feat_age"].append(logit_cv(np.column_stack([feat, age]), y, seed))
        res["tabular"].append(logit_cv(np.column_stack([tab, age]), y, seed, C=1.0))
        res["both"].append(logit_cv(np.column_stack([tab, age, feat]), y, seed))

    lbl = {"tab_gap": "tabular brain-age gap", "gap": "CNN brain-age gap",
           "feat_age": "CNN features + age", "direct": "CNN direct to disease",
           "finetune": "CNN age-pretrained, fine-tuned",
           "tabular": "volumes+ICV+sex+age  <- to beat",
           "both": "volumes+age + CNN features"}
    print()
    for k in ("tab_gap", "gap", "feat_age", "direct", "finetune", "tabular", "both"):
        a = np.array(res[k])
        print(f"  {lbl[k]:34s} AUC = {a.mean():.3f}  "
              f"({', '.join(f'{v:.3f}' for v in a)})")
    print(f"\n  does the image ADD to volumes? "
          f"{np.mean(res['both']) - np.mean(res['tabular']):+.3f}")
    print(f"  pretraining gain (finetune - direct): "
          f"{np.mean(res['finetune']) - np.mean(res['direct']):+.3f}")
    return {k: np.array(v) for k, v in res.items()}


def main():
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    X, m = load_data()
    print(f"\nbaselines: {len(m)}  {m.DX.value_counts().to_dict()}")
    print(f"volume {X.shape[1:]}, {X.nbytes/1e9:.2f} GB uint8")
    print(f"age {m.AGE.min():.0f}-{m.AGE.max():.0f}, mean {m.AGE.mean():.1f}")

    out = {}
    out["AD"] = run("AD vs CN", X, m, ["Dementia"])
    out["MCI"] = run("MCI vs CN  (earlier, harder)", X, m, ["MCI"])
    np.savez(f"{OUT}/adni_cnn_results.npz",
             **{f"{a}_{b}": v for a, r in out.items() for b, v in r.items()})
    m.to_csv(f"{OUT}/adni_cnn_subjects.csv", index=False)
    print("\nsaved adni_cnn_results.npz")


if __name__ == "__main__":
    main()
