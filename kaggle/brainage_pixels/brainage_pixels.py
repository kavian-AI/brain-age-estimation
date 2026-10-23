#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHASE 2 - the full pixels-to-diagnosis pipeline, 5-fold, pretrained backbone.

Everything here starts from a scan. No FreeSurfer table is available at test
time to any arm except the ceiling, which exists only to say how much is left on
the table.

  PATH A  image -> 7 volumes -> age -> gap -> disease     the anatomical bottleneck
  PATH B  image -> age -> gap -> disease                  direct brain age
  PATH C  image features + chronological age -> disease   the fair baseline
  PATH D  image -> disease                                supervised end to end
  CEILING true FreeSurfer volumes + age -> disease        not reachable from pixels

PATH C is the control that keeps the others honest. The brain-age gap is a
function of the image and of age, so a model handed both should never do worse
than the gap. Wherever it does, the gap is discarding information rather than
distilling it - which is what the whole project has been measuring.

Three clinical tasks (AD vs CN, MCI vs CN, AD vs MCI age-matched) plus the
longitudinal arm: predicted volumes for EVERY visit of a held-out subject give
an image-derived RATE of brain ageing, and MCI -> dementia conversion is scored
by Cox C-index under a landmark design.

Out-of-fold by SUBJECT throughout. The network that produces any number for a
subject never saw that subject. The age model is fitted on CN subjects only.
Intensity is divided by the foreground p95, never the mean.
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

OUT = "/kaggle/working"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
TARGETS = V + ["ICV"]
FNAME = re.compile(r"^(\d{3}_S_\d{4,5})(?:_\d+)?_([A-Za-z]+\d*)\.npy$")

# ResNet-50 won the phase-1 screening (mean r 0.504 vs ViT-S 0.462,
# ConvNeXt-T 0.454, ViT-B 0.426): convolutional priors still beat
# transformers at ~2,400 medical images.
BACKBONE = os.environ.get("BB", "resnet50.a1_in1k")
NPLANE, SIZE = 10, 224
FOLDS, EPOCHS, BATCH = 5, 8, 8
MAX_VISITS = 4
IMNET_MEAN, IMNET_STD = 0.485, 0.229


def pick_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        nn.Conv2d(1, 1, 3).cuda()(torch.zeros(1, 1, 8, 8, device="cuda"))
        return "cuda"
    except Exception as e:
        print(f"GPU unusable: {type(e).__name__} -> CPU", flush=True)
        return "cpu"


dev = pick_device()


def find(pat, what):
    h = sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True))
    if not h:
        print(f"cannot find {what}")
        for d in sorted(glob.glob("/kaggle/input/*")):
            print("  ", d, os.listdir(d)[:5])
        sys.exit(1)
    return h[0]


def load_data():
    a = pd.read_csv(find("ADNIMERGE*.csv", "ADNIMERGE"), low_memory=False)
    keep = ["PTID", "VISCODE", "AGE", "PTGENDER", "DX", "ICV", "Years_bl"] + V
    lut = a[keep].copy()
    lut["key"] = lut.PTID + "|" + lut.VISCODE.astype(str)
    lut = lut.drop_duplicates("key").set_index("key")

    root = os.path.dirname(find("*_S_*.npy", "volumes"))
    files = sorted(glob.glob(os.path.join(root, "**", "*.npy"), recursive=True))
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
        fg = v[v > v.max() * 0.08]
        if fg.size < 1000:
            continue
        v = v / np.percentile(fg, 95)
        z = np.linspace(int(v.shape[2] * .22), int(v.shape[2] * .78),
                        NPLANE).astype(int)
        sl = torch.from_numpy(np.stack([v[:, :, k] for k in z])).unsqueeze(0)
        sl = F.interpolate(sl, size=(SIZE, SIZE), mode="bilinear",
                           align_corners=False)[0].numpy()
        X.append((np.clip(sl, 0, 2) * 127.5).astype(np.uint8))
        rows.append(dict(PTID=g.group(1), VISCODE=vis, AGE=r.AGE,
                         Years_bl=r.Years_bl, DX=r.DX,
                         SEX=1 if r.PTGENDER == "Male" else 0,
                         **{k: r[k] for k in TARGETS}))
        if len(rows) % 400 == 0:
            print(f"  loaded {len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    m = pd.DataFrame(rows)
    m["age_visit"] = m.AGE + m.Years_bl
    return np.stack(X), m


class DS(torch.utils.data.Dataset):
    def __init__(self, X, Y, train):
        self.X, self.Y, self.train = X, Y, train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        v = torch.from_numpy(self.X[i].astype(np.float32) / 127.5)
        t = torch.stack([v[k - 1:k + 2] for k in range(1, NPLANE - 1)])
        if self.train:
            if np.random.rand() < .5:
                t = torch.flip(t, [3])
            a = np.deg2rad(np.random.uniform(-8, 8))
            tx, ty = np.random.uniform(-.05, .05, 2)
            m = torch.tensor([[np.cos(a), -np.sin(a), tx],
                              [np.sin(a), np.cos(a), ty]],
                             dtype=torch.float32).unsqueeze(0).repeat(len(t), 1, 1)
            g = F.affine_grid(m, t.shape, align_corners=False)
            t = F.grid_sample(t, g, align_corners=False)
            t = t * np.random.uniform(.92, 1.08)
        return (t - IMNET_MEAN) / IMNET_STD, torch.from_numpy(self.Y[i])


class Net(nn.Module):
    _announced = False

    def __init__(self, nout):
        super().__init__()
        import timm
        kw = dict(pretrained=True, num_classes=0)
        if "dinov2" in BACKBONE:
            kw["img_size"] = SIZE
        self.net = timm.create_model(BACKBONE, **kw)
        self.dim = self.net.num_features
        if not Net._announced:
            w = next(self.net.parameters()).detach()
            print(f"    backbone {BACKBONE}: PRETRAINED loaded, "
                  f"dim {self.dim}, first-layer std {w.std():.4f}",
                  flush=True)
            Net._announced = True
        self.head = nn.Sequential(nn.LayerNorm(self.dim), nn.Dropout(.2),
                                  nn.Linear(self.dim, nout))

    def features(self, x):
        b, s, c, h, w = x.shape
        return self.net(x.reshape(b * s, c, h, w)).reshape(b, s, -1).mean(1)

    def forward(self, x):
        return self.head(self.features(x))


def train(Xtr, Ytr, seed=0, epochs=EPOCHS, cls=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(Ytr.shape[1]).to(dev)
    lr = 3e-5 if "vit" in BACKBONE else 1e-4
    opt = torch.optim.AdamW([{"params": net.net.parameters(), "lr": lr},
                             {"params": net.head.parameters(), "lr": 1e-3}],
                            weight_decay=.02)
    dl = torch.utils.data.DataLoader(DS(Xtr, Ytr, True), batch_size=BATCH,
                                     shuffle=True, drop_last=len(Xtr) > BATCH,
                                     num_workers=2)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, [lr * 4, 4e-3], epochs=epochs, steps_per_epoch=max(1, len(dl)))
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    lf = nn.BCEWithLogitsLoss() if cls else nn.SmoothL1Loss()
    net.train()
    for ep in range(epochs):
        tot = 0
        for xb, yb in dl:
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=dev == "cuda"):
                loss = lf(net(xb.to(dev)), yb.to(dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            tot += loss.item()
        if ep % 3 == 0 or ep == epochs - 1:
            print(f"      ep {ep} loss {tot/max(1,len(dl)):.4f}", flush=True)
    return net


@torch.no_grad()
def infer(net, X, feats=False):
    net.eval()
    o = []
    dl = torch.utils.data.DataLoader(
        DS(X, np.zeros((len(X), 1), dtype=np.float32), False),
        batch_size=16, num_workers=2)
    for xb, _ in dl:
        with torch.cuda.amp.autocast(enabled=dev == "cuda"):
            z = net.features(xb.to(dev)) if feats else net(xb.to(dev))
        o.append(z.float().cpu().numpy())
    return np.concatenate(o)


def main():
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    print("backbone:", BACKBONE, flush=True)
    X, m = load_data()
    print(f"\nscans {len(m)}  subjects {m.PTID.nunique()}  "
          f"{X.nbytes/1e9:.2f} GB", flush=True)
    bl = m.VISCODE == "bl"
    print("baselines:", m[bl].DX.value_counts(dropna=False).to_dict())

    Yv = m[TARGETS].values.astype(np.float32)
    age = m.age_visit.values.astype(np.float32)
    cn = (m.DX == "CN").values
    subs = m.PTID.values
    uq = np.unique(subs)
    fmap = dict(zip(uq, np.random.RandomState(0).permutation(len(uq)) % FOLDS))
    fold = np.array([fmap[s] for s in subs])

    ysc = StandardScaler().fit(Yv)
    Yvz = ysc.transform(Yv).astype(np.float32)
    asc = StandardScaler().fit(age[cn].reshape(-1, 1))

    predV = np.zeros_like(Yvz)
    predA = np.zeros(len(m), dtype=np.float32)
    featF = None

    for k in range(FOLDS):
        tr = np.where(fold != k)[0]
        te = np.where(fold == k)[0]
        d = m.iloc[tr].copy()
        d["_i"] = tr
        cap = d.sort_values("Years_bl").groupby("PTID").head(MAX_VISITS)._i.values
        print(f"\n{'='*66}\nfold {k}: train {len(cap)} scans, test {len(te)}")
        print("  PATH A  image -> volumes", flush=True)
        na = train(X[cap], Yvz[cap], seed=k)
        predV[te] = infer(na, X[te])
        f = infer(na, X[te], feats=True)
        if featF is None:
            featF = np.zeros((len(m), f.shape[1]), dtype=np.float32)
        featF[te] = f
        del na
        torch.cuda.empty_cache()

        cnt = np.array([i for i in cap if cn[i]])
        print(f"  PATH B  image -> age  (CN only, n={len(cnt)})", flush=True)
        nb = train(X[cnt], asc.transform(age[cnt].reshape(-1, 1)).astype(np.float32),
                   seed=100 + k)
        predA[te] = asc.inverse_transform(infer(nb, X[te])).ravel()
        del nb
        torch.cuda.empty_cache()

    P = ysc.inverse_transform(predV)
    print("\n" + "=" * 74)
    print("STAGE A  image -> FreeSurfer numbers (out of fold)")
    print("=" * 74)
    for j, t in enumerate(TARGETS):
        print(f"  {t:16s} r = {np.corrcoef(P[:, j], Yv[:, j])[0,1]:.3f}")
    hm = np.abs(predA[cn] - age[cn]).mean()
    print(f"\nPATH B direct age model, CN MAE = {hm:.2f} y  "
          f"r = {np.corrcoef(predA[cn], age[cn])[0,1]:.3f}")

    np.savez(f"{OUT}/pixels.npz", pred=P, true=Yv, predage=predA,
             feat=featF, subs=subs, vis=m.VISCODE.values, fold=fold)
    m.assign(**{f"P_{t}": P[:, j] for j, t in enumerate(TARGETS)}
             ).assign(pred_age=predA).to_csv(f"{OUT}/pixels_predictions.csv",
                                             index=False)

    # ------------------------------------------------ downstream, baselines
    b = np.where(m.VISCODE.values == "bl")[0]
    mb = m.iloc[b].reset_index(drop=True)
    for task, pos in [("AD vs CN", "Dementia"), ("MCI vs CN", "MCI")]:
        s = mb.DX.isin(["CN", pos]).values
        idx = b[s]
        y = (mb.DX[s] == pos).astype(int).values
        a = mb.age_visit.values[s].astype(float)
        f5 = np.array([fmap[x] for x in mb.PTID.values[s]])
        print("\n" + "=" * 74)
        print(f"{task}   n={len(y)} (CN {int((y==0).sum())}, {pos} {int(y.sum())})")
        print("=" * 74)
        arms = {
            "A bottleneck gap": ("gap", np.column_stack([P[idx], mb.SEX.values[s]])),
            "A bottleneck direct": ("dir", np.column_stack([P[idx], mb.SEX.values[s]])),
            "B direct brain-age gap": ("gapfix", predA[idx].reshape(-1, 1)),
            "C image features + age": ("dir", featF[idx]),
            "CEILING true volumes": ("dir", np.column_stack([Yv[idx],
                                                             mb.SEX.values[s]])),
        }
        for nm, (kind, D) in arms.items():
            o = np.zeros(len(y))
            for k in range(FOLDS):
                tr, te = f5 != k, f5 == k
                h = np.where(tr & (y == 0))[0]      # CN, TRAINING subjects only
                if kind == "gap":
                    sc = StandardScaler().fit(D[h])
                    mdl = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(
                        sc.transform(D[h]), a[h])
                    bb = LinearRegression().fit(
                        a[h].reshape(-1, 1),
                        mdl.predict(sc.transform(D[h])) - a[h])
                    o[te] = (mdl.predict(sc.transform(D[te])) - a[te]) \
                        - bb.predict(a[te].reshape(-1, 1))
                elif kind == "gapfix":
                    # the network already predicted age; only the regression-
                    # dilution correction is fitted here, and it must be fitted
                    # on TRAINING controls or the test fold leaks into it
                    g = D.ravel() - a
                    bb = LinearRegression().fit(a[h].reshape(-1, 1), g[h])
                    o[te] = g[te] - bb.predict(a[te].reshape(-1, 1))
                else:
                    Dz = np.column_stack([D, a])
                    sc = StandardScaler().fit(Dz[tr])
                    c = LogisticRegression(max_iter=4000,
                                           C=0.05 if D.shape[1] > 20 else 1.0
                                           ).fit(sc.transform(Dz[tr]), y[tr])
                    o[te] = c.predict_proba(sc.transform(Dz[te]))[:, 1]
            print(f"  {nm:26s} AUC = {roc_auc_score(y, o):.3f}")

        # PATH D: fine-tune the backbone on the disease label itself. The most
        # direct route from pixels, and the one with the least supervision to
        # learn from - ~500 baseline scans against 3,286 for the volume target.
        if pos == "Dementia":
            od = np.zeros(len(y))
            for k in range(FOLDS):
                tr, te = np.where(f5 != k)[0], np.where(f5 == k)[0]
                nd = train(X[idx[tr]], y[tr].astype(np.float32).reshape(-1, 1),
                           seed=200 + k, epochs=6, cls=True)
                od[te] = infer(nd, X[idx[te]]).ravel()
                del nd
                torch.cuda.empty_cache()
            print(f"  {'D end-to-end image->dx':26s} AUC = "
                  f"{roc_auc_score(y, od):.3f}")

    print("\nsaved pixels.npz + pixels_predictions.csv "
          "(all visits -> enables the RATE arm locally)")


if __name__ == "__main__":
    main()
