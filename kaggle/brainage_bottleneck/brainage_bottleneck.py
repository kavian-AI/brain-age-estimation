#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THE ANATOMICAL BOTTLENECK: image -> volumes -> age -> disease.

THE IDEA
--------
Part II established that raw images lose badly to 6 FreeSurfer volumes: on ADNI,
AD vs CN reached 0.952 from the volumes and 0.648 from voxel PCA, and adding the
image to the volumes made things worse (-0.021). The natural reading is "the
image has less information", which cannot be true - the volumes are COMPUTED
from the image. The information is there; roughly 1,000 subjects is simply not
enough for a network to discover, unaided, the representation that FreeSurfer
spent decades of anatomical knowledge encoding.

So do not make it discover that representation. Supervise it. Fine-tune a
pretrained vision backbone to output the 6 volumes + ICV directly, then feed
those predicted numbers into the same two-stage pipeline every earlier
experiment used. The volumes become an anatomically meaningful bottleneck
instead of something the network has to invent.

WHAT THIS CAN SHOW
------------------
If predicted volumes recover most of the disease performance of true volumes,
then the earlier negative result was about REPRESENTATION, not about
information - and the honest conclusion of Part II changes from "images do not
help at this scale" to "images need the right supervision at this scale".
If they do not recover it, the negative result stands and is stronger for having
survived a serious attempt.

The comparison that matters is predicted-volumes vs TRUE-volumes on identical
folds, so true volumes are recomputed here rather than quoted.

LEAKAGE
-------
Outer 5-fold by SUBJECT. Within a fold the volume network sees only training
subjects, and every number it produces for a test subject is out-of-fold. It
trains on ALL VISITS of training subjects (more data, no leakage, since the
split is by subject) and predicts all visits of test subjects - which also
yields an image-derived RATE of change for the longitudinal arm.

Predicting anatomy is diagnosis-agnostic, so stage A trains on every diagnosis,
not just CN. The AGE model downstream is still fitted on CN only, as always.
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
TARGETS = V + ["ICV"]
FNAME = re.compile(r"^(\d{3}_S_\d{4,5})(?:_\d+)?_([A-Za-z]+\d*)\.npy$")

NSL = 12                # axial levels fed to the backbone
SIZE = 128
MAX_VISITS_TRAIN = 3    # cap per training subject, to bound runtime
FOLDS, EPOCHS, BATCH, LR = 5, 14, 8, 2e-4


def pick_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.nn.Conv2d(1, 1, 3).cuda()(torch.zeros(1, 1, 8, 8, device="cuda"))
        return "cuda"
    except Exception as e:
        print(f"GPU unusable: {type(e).__name__} -> CPU", flush=True)
        return "cpu"


dev = pick_device()


def find(pat, what):
    h = sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True))
    if not h:
        print(f"cannot find {what}; inputs:")
        for d in sorted(glob.glob("/kaggle/input/*")):
            print("  ", d, os.listdir(d)[:5])
        sys.exit(1)
    return h[0]


# ------------------------------------------------------------------ data

def load_data():
    a = pd.read_csv(find("ADNIMERGE*.csv", "ADNIMERGE"), low_memory=False)
    keep = ["PTID", "VISCODE", "AGE", "PTGENDER", "DX", "ICV", "Years_bl"] + V
    lut = a[keep].copy()
    lut["key"] = lut.PTID + "|" + lut.VISCODE.astype(str)
    lut = lut.drop_duplicates("key").set_index("key")

    root = os.path.dirname(find("*_S_*.npy", "the volume archive"))
    files = glob.glob(os.path.join(root, "**", "*.npy"), recursive=True)
    print(f"volumes on disk: {len(files)}", flush=True)

    rows, X, t0 = [], [], time.time()
    for f in files:
        g = FNAME.match(os.path.basename(f))
        if not g:
            continue
        vis = "bl" if g.group(2) == "sc" else g.group(2)
        if re.fullmatch(r"v\d+", vis):
            continue
        key = f"{g.group(1)}|{vis}"
        if key not in lut.index:
            continue
        r = lut.loc[key]
        if pd.isna(r.AGE) or pd.isna(r.Years_bl) or r[TARGETS].isna().any():
            continue
        v = np.load(f).astype(np.float32)
        if v.ndim != 3:
            continue
        # take NSL evenly spaced axial levels from the middle 70% of the brain
        z0, z1 = int(v.shape[2] * .18), int(v.shape[2] * .82)
        idx = np.linspace(z0, z1, NSL).astype(int)
        sl = np.stack([v[:, :, k] for k in idx])
        sl = torch.from_numpy(sl).unsqueeze(0)
        sl = F.interpolate(sl, size=(SIZE, SIZE), mode="bilinear",
                           align_corners=False)[0].numpy()
        X.append((np.clip(sl, 0, 1) * 255).astype(np.uint8))
        rows.append(dict(PTID=g.group(1), VISCODE=vis, AGE=r.AGE,
                         Years_bl=r.Years_bl, DX=r.DX,
                         SEX=1 if r.PTGENDER == "Male" else 0,
                         **{k: r[k] for k in TARGETS}))
        if len(rows) % 500 == 0:
            print(f"  loaded {len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    m = pd.DataFrame(rows)
    m["age_visit"] = m.AGE + m.Years_bl        # AGE is BASELINE age in ADNIMERGE
    return np.stack(X), m


class Slices(Dataset):
    def __init__(self, X, Y, train):
        self.X, self.Y, self.train = X, Y, train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        v = torch.from_numpy(self.X[i].astype(np.float32) / 255.)
        if self.train:
            if np.random.rand() < .5:
                v = torch.flip(v, [2])
            ang = np.random.uniform(-7, 7)
            tx, ty = np.random.uniform(-.05, .05, 2)
            a = np.deg2rad(ang)
            mtx = torch.tensor([[np.cos(a), -np.sin(a), tx],
                                [np.sin(a), np.cos(a), ty]],
                               dtype=torch.float32).unsqueeze(0)
            g = F.affine_grid(mtx, (1,) + tuple(v.shape), align_corners=False)
            v = F.grid_sample(v.unsqueeze(0), g, align_corners=False)[0]
            v = v * np.random.uniform(.93, 1.07)
        return v, torch.from_numpy(self.Y[i])


# ----------------------------------------------------------------- model

class VolNet(nn.Module):
    """Pretrained 2D backbone applied per slice, pooled across slices.

    Each axial level is passed through the same ImageNet backbone as a 3-channel
    image (the slice repeated), features are averaged over levels, and a linear
    head regresses the 7 anatomical targets. ImageNet weights are a strange prior
    for MRI, but they supply generic edge and texture filters that 931 subjects
    cannot buy, which is exactly the shortage this experiment is about.
    """

    def __init__(self, nout, pretrained=True):
        super().__init__()
        import torchvision
        try:
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = torchvision.models.resnet18(weights=w)
            self.pretrained = pretrained
        except Exception as e:
            print(f"  pretrained weights unavailable ({type(e).__name__}), "
                  f"training from scratch", flush=True)
            net = torchvision.models.resnet18(weights=None)
            self.pretrained = False
        self.stem = nn.Sequential(*list(net.children())[:-1])
        self.head = nn.Sequential(nn.Dropout(.3), nn.Linear(512, nout))

    def features(self, x):
        b, s, h, w = x.shape
        z = x.reshape(b * s, 1, h, w).repeat(1, 3, 1, 1)
        z = self.stem(z).flatten(1)
        return z.reshape(b, s, -1).mean(1)

    def forward(self, x):
        return self.head(self.features(x))


def train_volnet(X, Y, seed=0, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = VolNet(Y.shape[1]).to(dev)
    opt = torch.optim.AdamW([
        {"params": net.stem.parameters(), "lr": LR},
        {"params": net.head.parameters(), "lr": LR * 10}], weight_decay=1e-4)
    dl = DataLoader(Slices(X, Y, True), batch_size=BATCH, shuffle=True,
                    drop_last=len(X) > BATCH, num_workers=2)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, [LR * 4, LR * 40], epochs=epochs, steps_per_epoch=max(1, len(dl)))
    lf = nn.SmoothL1Loss()
    net.train()
    for ep in range(epochs):
        tot = 0
        for xb, yb in dl:
            opt.zero_grad()
            loss = lf(net(xb.to(dev)), yb.to(dev))
            loss.backward()
            opt.step()
            sch.step()
            tot += loss.item()
        if ep % 4 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:2d}  loss {tot/max(1,len(dl)):.4f}", flush=True)
    return net


@torch.no_grad()
def predict(net, X, bs=16):
    net.eval()
    o = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs].astype(np.float32) / 255.).to(dev)
        o.append(net(xb).cpu().numpy())
    return np.concatenate(o)


# ------------------------------------------------------- downstream suite

def gap_pipeline(tab_fit, age_fit, sets):
    sc = StandardScaler().fit(tab_fit)
    mdl = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(tab_fit), age_fit)
    b = LinearRegression().fit(age_fit.reshape(-1, 1),
                               mdl.predict(sc.transform(tab_fit)) - age_fit)
    out = []
    for tab, age in sets:
        out.append((mdl.predict(sc.transform(tab)) - age)
                   - b.predict(age.reshape(-1, 1)))
    return out


def evaluate(name, tab, age, y, folds, n_fold=FOLDS):
    """Identical two-stage pipeline to every earlier experiment."""
    gap = np.zeros(len(y))
    dire = np.zeros(len(y))
    mae = []
    for k in range(n_fold):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        h = tr[y[tr] == 0]
        if len(h) < 20:
            continue
        g_te, = gap_pipeline(tab[h], age[h], [(tab[te], age[te])])
        gap[te] = g_te
        sc = StandardScaler().fit(np.column_stack([tab[tr], age[tr]]))
        c = LogisticRegression(max_iter=4000, C=1.0).fit(
            sc.transform(np.column_stack([tab[tr], age[tr]])), y[tr])
        dire[te] = c.predict_proba(
            sc.transform(np.column_stack([tab[te], age[te]])))[:, 1]
        hh = te[y[te] == 0]
        if len(hh):
            gh, = gap_pipeline(tab[h], age[h], [(tab[hh], age[hh])])
            sc2 = StandardScaler().fit(tab[h])
            m2 = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc2.transform(tab[h]), age[h])
            mae.append(np.abs(m2.predict(sc2.transform(tab[hh])) - age[hh]).mean())
    return dict(gap=roc_auc_score(y, gap), direct=roc_auc_score(y, dire),
                age_mae=float(np.mean(mae)))


def main():
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    X, m = load_data()
    print(f"\nscans: {len(m)}, subjects: {m.PTID.nunique()}")
    print(f"slices tensor {X.shape}, {X.nbytes/1e9:.2f} GB uint8")
    bl = m.VISCODE == "bl"
    print(f"baselines: {int(bl.sum())}  {m[bl].DX.value_counts(dropna=False).to_dict()}")

    Y = m[TARGETS].values.astype(np.float32)
    ysc = StandardScaler().fit(Y)              # equal weight across targets
    Yz = ysc.transform(Y).astype(np.float32)

    subs = m.PTID.values
    uq = np.unique(subs)
    rng = np.random.RandomState(0)
    fmap = dict(zip(uq, rng.permutation(len(uq)) % FOLDS))
    fold_scan = np.array([fmap[s] for s in subs])

    # ---------------- stage A: image -> volumes, out of fold
    pred = np.zeros_like(Yz)
    for k in range(FOLDS):
        tr = np.where(fold_scan != k)[0]
        te = np.where(fold_scan == k)[0]
        # cap visits per training subject to bound runtime
        d = m.iloc[tr].copy()
        d["_i"] = tr
        keep = d.sort_values("Years_bl").groupby("PTID").head(MAX_VISITS_TRAIN)._i.values
        print(f"\nfold {k}: train scans {len(keep)} "
              f"(from {len(tr)}), test scans {len(te)}", flush=True)
        net = train_volnet(X[keep], Yz[keep], seed=k)
        pred[te] = predict(net, X[te])
        del net
        torch.cuda.empty_cache()

    P = ysc.inverse_transform(pred)
    m_pred = m.copy()
    for j, t in enumerate(TARGETS):
        m_pred[t] = P[:, j]

    print("\n" + "=" * 74)
    print("STAGE A - how well does the image reproduce the FreeSurfer numbers?")
    print("=" * 74)
    print(f"  {'target':16s} {'r':>6} {'MAE':>10}  {'MAE % of mean':>14}")
    q = {}
    for j, t in enumerate(TARGETS):
        r = np.corrcoef(P[:, j], Y[:, j])[0, 1]
        mae = np.abs(P[:, j] - Y[:, j]).mean()
        q[t] = r
        print(f"  {t:16s} {r:6.3f} {mae:10.0f}  {100*mae/Y[:, j].mean():13.1f}%")

    # ---------------- downstream, identical folds, predicted vs true
    b = m.VISCODE.values == "bl"
    mb = m[b].reset_index(drop=True)
    Pb, Tb = P[b], Y[b]
    dxok = mb.DX.notna().values
    for task, pos in [("AD vs CN", "Dementia"), ("MCI vs CN", "MCI")]:
        sel = dxok & mb.DX.isin(["CN", pos]).values
        d = mb[sel]
        y = (d.DX == pos).astype(int).values
        age = d.age_visit.values.astype(float)
        f = np.array([fmap[s] for s in d.PTID.values])
        print("\n" + "=" * 74)
        print(f"{task}   n={len(y)} (CN {int((y==0).sum())}, {pos} {int(y.sum())})")
        print("=" * 74)
        res = {}
        for nm, M in [("TRUE FreeSurfer volumes", Tb[sel]),
                      ("PREDICTED from image", Pb[sel])]:
            tab = np.column_stack([M, d.SEX.values])
            r = evaluate(nm, tab, age, y, f)
            res[nm] = r
            print(f"  {nm:26s} gap {r['gap']:.3f}   direct {r['direct']:.3f}   "
                  f"age MAE {r['age_mae']:.1f} y")
        rec = (res["PREDICTED from image"]["direct"]
               / res["TRUE FreeSurfer volumes"]["direct"])
        print(f"\n  predicted recovers {100*rec:.1f}% of the true-volume AUC")
        print(f"  for reference, voxel PCA reached "
              f"{0.648 if pos=='Dementia' else 0.613:.3f} and a 3D CNN gap "
              f"{'(see adni_cnn run)' if pos=='Dementia' else ''}")

    m_pred.to_csv(f"{OUT}/predicted_volumes.csv", index=False)
    np.savez(f"{OUT}/bottleneck.npz", pred=P, true=Y,
             subs=m.PTID.values, vis=m.VISCODE.values)
    print("\nsaved predicted_volumes.csv (all visits -> enables the RATE arm)")


if __name__ == "__main__":
    main()
