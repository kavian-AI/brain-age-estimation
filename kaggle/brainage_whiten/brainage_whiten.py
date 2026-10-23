#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WHY THE ANATOMICAL BOTTLENECK COLLAPSED, AND WHETHER THE TARGET CAN FIX IT.

E54 ran the whole E48 pipeline on the previous bottleneck network's predicted
volumes and got a flat result: brain-age gap 0.834, one component 0.834, six
deviations 0.834, identical to three decimals, with controls-only loadings of
+0.42 +0.42 +0.42 +0.42 -0.33 +0.42. The diagnosis was unambiguous:

    mean |r| between regional deviations   image 0.908   FreeSurfer 0.495
    effective number of dimensions         image 1.16    FreeSurfer 2.60

The network did not learn six regional measurements. It learned ONE global
atrophy score and wrote it six times. A rotation has nothing to rotate, so E54
tested the network, not E48.

THE HYPOTHESIS. The six volumes share an enormous common factor - head size and
global atrophy. The old kernel standardised each target, which equalises their
SCALE but does nothing about the shared FACTOR. Squared error is then minimised
by learning the dominant direction and stopping; the regional residuals are small
and the network never spends capacity on them. If that is the cause, the cure is
in the TARGET, not the architecture - which would be the same lesson E48 taught
about chronological age, one level down.

THREE HEADS ON ONE BACKBONE, so the ablation is exact and costs one training run:

    head V   raw volumes + ICV, standardised        the old target
    head Z   normative deviations z, standardised   global factor removed
    head W   whitened z                             every direction equal variance

If Z beats V, the shared factor was the problem. If W beats Z, the remaining
variance imbalance was too. If neither moves, the collapse is a sample-size limit
and no choice of target rescues it at n~900 - which is worth knowing and would
close the question honestly.

Head Z and head W need no ICV at inference, which the volume route does. One less
thing to estimate from the image.

LEAKAGE. Outer 5-fold by SUBJECT. The normative model, the whitening matrix, the
downstream age model and the rotation are all fitted on training subjects only
(controls only where the method requires it), and every number reported for a
test subject is out of fold.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

OUT = "/kaggle/working"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
TARGETS = V + ["ICV"]
FNAME = re.compile(r"^(\d{3}_S_\d{4,5})(?:_\d+)?_([A-Za-z]+\d*)\.npy$")

NSL, SIZE = 12, 128
MAX_VISITS_TRAIN = 3
FOLDS, EPOCHS, BATCH, LR = 5, 12, 8, 2e-4
WHITEN_RIDGE = 0.05      # damps near-null directions so whitening cannot
                         # amplify pure noise into the loss


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
print(f"device: {dev}", flush=True)


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
        z0, z1 = int(v.shape[2] * .18), int(v.shape[2] * .82)
        idx = np.linspace(z0, z1, NSL).astype(int)
        sl = torch.from_numpy(np.stack([v[:, :, k] for k in idx])).unsqueeze(0)
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
    m["age_visit"] = m.AGE + m.Years_bl
    return np.stack(X), m.reset_index(drop=True)


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
class TriNet(nn.Module):
    """One ResNet-18 trunk, three linear heads. The heads differ only in what
    they are asked to predict, which is the whole point of the experiment."""

    def __init__(self, n_v, n_z, n_w):
        super().__init__()
        import torchvision
        try:
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            net = torchvision.models.resnet18(weights=w)
            self.pretrained = True
        except Exception as e:
            print(f"  pretrained weights unavailable ({type(e).__name__})",
                  flush=True)
            net = torchvision.models.resnet18(weights=None)
            self.pretrained = False
        self.stem = nn.Sequential(*list(net.children())[:-1])
        self.drop = nn.Dropout(.3)
        self.hv = nn.Linear(512, n_v)
        self.hz = nn.Linear(512, n_z)
        self.hw = nn.Linear(512, n_w)

    def forward(self, x):
        b, s, h, w = x.shape
        z = x.reshape(b * s, 1, h, w).repeat(1, 3, 1, 1)
        z = self.stem(z).flatten(1).reshape(b, s, -1).mean(1)
        z = self.drop(z)
        return self.hv(z), self.hz(z), self.hw(z)


def train_net(X, Yv, Yz, Yw, seed=0, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = TriNet(Yv.shape[1], Yz.shape[1], Yw.shape[1]).to(dev)
    Y = np.concatenate([Yv, Yz, Yw], axis=1).astype(np.float32)
    kv, kz = Yv.shape[1], Yv.shape[1] + Yz.shape[1]
    opt = torch.optim.AdamW([
        {"params": net.stem.parameters(), "lr": LR},
        {"params": list(net.hv.parameters()) + list(net.hz.parameters())
         + list(net.hw.parameters()), "lr": LR * 10}], weight_decay=1e-4)
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
            pv, pz, pw = net(xb.to(dev))
            yb = yb.to(dev)
            # equal weight per head, so no head can dominate by output count
            loss = (lf(pv, yb[:, :kv]) + lf(pz, yb[:, kv:kz])
                    + lf(pw, yb[:, kz:]))
            loss.backward()
            opt.step()
            sch.step()
            tot += loss.item()
        if ep % 3 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:2d}  loss {tot/max(1,len(dl)):.4f}", flush=True)
    return net


@torch.no_grad()
def predict(net, X, bs=16):
    net.eval()
    a, b, c = [], [], []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs].astype(np.float32) / 255.).to(dev)
        pv, pz, pw = net(xb)
        a.append(pv.cpu().numpy())
        b.append(pz.cpu().numpy())
        c.append(pw.cpu().numpy())
    return np.vstack(a), np.vstack(b), np.vstack(c)


# ------------------------------------------------------------- normative
def fit_normative(df, cn_mask):
    """Per-region model of volume on age, age^2, sex, ICV, fitted on controls.
    Returns a function mapping any frame to its six deviations."""
    ag = df.age_visit.values.astype(float)
    X = np.column_stack([ag, ag ** 2, df.SEX.values, df.ICV.values])
    sc = StandardScaler().fit(X[cn_mask])
    models, sds = [], []
    for v in V:
        y = df[v].values.astype(float)
        m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[cn_mask]),
                                                       y[cn_mask])
        models.append(m)
        sds.append(max((y[cn_mask] - m.predict(sc.transform(X[cn_mask]))
                        ).std(ddof=1), 1e-9))

    def apply(d2):
        a2 = d2.age_visit.values.astype(float)
        X2 = np.column_stack([a2, a2 ** 2, d2.SEX.values, d2.ICV.values])
        return np.column_stack([
            (d2[v].values.astype(float) - models[j].predict(sc.transform(X2)))
            / sds[j] for j, v in enumerate(V)])
    return apply


def whitener(Z):
    """PCA whitening with a ridge, plus its inverse. Fitted on training rows."""
    mu = Z.mean(0)
    C = np.cov((Z - mu).T)
    ev, EV = np.linalg.eigh(C)
    ev = np.clip(ev, 0, None) + WHITEN_RIDGE * np.trace(C) / len(C)
    Wf = EV @ np.diag(ev ** -0.5)          # forward: (Z-mu) @ Wf
    Wi = np.diag(ev ** 0.5) @ EV.T         # inverse: Zw @ Wi + mu
    return (lambda Zx: (Zx - mu) @ Wf), (lambda Zw: Zw @ Wi + mu)


# --------------------------------------------------------------- scoring
def eff_dim(Z):
    C = np.corrcoef(Z.T)
    e = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
    off = np.abs(C[np.triu_indices(C.shape[0], 1)])
    return (e.sum() ** 2) / (e ** 2).sum(), off.mean(), 100 * e.max() / e.sum()


def auc_of(X, y):
    sc = StandardScaler().fit(X)
    c = LogisticRegression(max_iter=4000).fit(sc.transform(X), y)
    return roc_auc_score(y, c.predict_proba(sc.transform(X))[:, 1])


def report_arm(name, Zoof, Ztrue, gap, dx, age, cn_all):
    sel = np.isin(dx, ["CN", "Dementia"])
    y = (dx[sel] == "Dementia").astype(int)
    ag = age[sel].reshape(-1, 1)
    Zs, cns = Zoof[sel], (dx[sel] == "CN")
    sc = StandardScaler().fit(Zs[cns])
    p = PCA(n_components=1, random_state=0).fit(sc.transform(Zs[cns]))
    pc = p.transform(sc.transform(Zs))
    ed, mo, v1 = eff_dim(Zoof)
    rs = [np.corrcoef(Zoof[:, j], Ztrue[:, j])[0, 1] for j in range(len(V))]
    print(f"\n  --- {name} ---")
    print(f"    agreement with FreeSurfer z: " +
          "  ".join(f"{v[:9]} {r:.3f}" for v, r in zip(V, rs)))
    print(f"    mean |r| between regions {mo:.3f}   first direction {v1:.1f}%"
          f"   EFFECTIVE DIMENSIONS {ed:.2f}/6")
    a_gap = auc_of(np.column_stack([gap[sel], ag]), y)
    a_pc = auc_of(np.column_stack([pc, ag]), y)
    a_six = auc_of(np.column_stack([Zs, ag]), y)
    print(f"    AD vs CN   gap {a_gap:.3f} | one component {a_pc:.3f} | "
          f"six {a_six:.3f} | component minus gap {a_pc-a_gap:+.3f}")
    print("    loadings: " + "  ".join(f"{v[:9]} {w:+.2f}"
                                       for v, w in zip(V, p.components_[0])))
    return dict(name=name, eff_dim=ed, mean_r=mo, auc_gap=a_gap, auc_pc=a_pc,
                auc_six=a_six, region_r=rs)


# ------------------------------------------------------------------ main
def main():
    X, m = load_data()
    print(f"\nusable scans {len(m)}, subjects {m.PTID.nunique()}", flush=True)
    print(m.DX.value_counts(dropna=False).to_string(), flush=True)

    subs = m.PTID.values
    uq = np.unique(subs)
    fm = dict(zip(uq, np.random.RandomState(0).permutation(len(uq)) % FOLDS))
    fold = np.array([fm[s] for s in subs])

    n = len(m)
    P_vol = np.zeros((n, len(TARGETS)))
    P_z = np.zeros((n, len(V)))
    P_w = np.zeros((n, len(V)))
    Z_true = np.zeros((n, len(V)))
    gap_img = np.zeros(n)
    gap_true = np.zeros(n)

    for k in range(FOLDS):
        t0 = time.time()
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        cn_tr = (m.DX.values[tr] == "CN")
        print(f"\n=== fold {k}: train {len(tr)} scans "
              f"({cn_tr.sum()} CN), test {len(te)} ===", flush=True)

        # --- targets, all fitted on TRAIN only
        norm = fit_normative(m.iloc[tr], cn_tr)
        Ztr = norm(m.iloc[tr])
        Z_true[te] = norm(m.iloc[te])
        fwd, inv = whitener(Ztr)

        ysc_v = StandardScaler().fit(m.iloc[tr][TARGETS].values.astype(np.float32))
        ysc_z = StandardScaler().fit(Ztr)
        Yv = ysc_v.transform(m.iloc[tr][TARGETS].values).astype(np.float32)
        Yz = ysc_z.transform(Ztr).astype(np.float32)
        Yw = fwd(Ztr).astype(np.float32)

        # cap visits per training subject to bound runtime
        d_tr = m.iloc[tr].copy()
        d_tr["_i"] = np.arange(len(tr))
        keep = d_tr.groupby("PTID").head(MAX_VISITS_TRAIN)._i.values
        net = train_net(X[tr][keep], Yv[keep], Yz[keep], Yw[keep], seed=k)

        pv, pz, pw = predict(net, X[te])
        P_vol[te] = ysc_v.inverse_transform(pv)
        P_z[te] = ysc_z.inverse_transform(pz)
        P_w[te] = inv(pw)
        del net
        if dev == "cuda":
            torch.cuda.empty_cache()

        print(f"  fold done in {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------- downstream, 2nd pass
    # The age model for the image arm has to be TRAINED on image volumes, or it
    # is a FreeSurfer-scale model applied to CNN-scale inputs and the image arm
    # loses to a domain mismatch rather than to anything real. That needs the
    # training rows' image volumes, which only exist once every fold has run -
    # hence a second pass over the same folds, now with P_vol complete and
    # out of fold everywhere.
    def gap_from(Xall, tr, te, cn_tr):
        ag_tr = m.iloc[tr].age_visit.values.astype(float)
        ag_te = m.iloc[te].age_visit.values.astype(float)
        sc = StandardScaler().fit(Xall[tr][cn_tr])
        ma = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(
            sc.transform(Xall[tr][cn_tr]), ag_tr[cn_tr])
        bb = LinearRegression().fit(
            ag_tr[cn_tr].reshape(-1, 1),
            ma.predict(sc.transform(Xall[tr][cn_tr])) - ag_tr[cn_tr])
        return (ma.predict(sc.transform(Xall[te])) - ag_te) \
            - bb.predict(ag_te.reshape(-1, 1))

    X_img = np.column_stack([P_vol, m.SEX.values])
    X_true = np.column_stack([m[v].values for v in V]
                             + [m.ICV.values, m.SEX.values])
    for k in range(FOLDS):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        cn_tr = (m.DX.values[tr] == "CN")
        gap_img[te] = gap_from(X_img, tr, te, cn_tr)
        gap_true[te] = gap_from(X_true, tr, te, cn_tr)

    # ---------------------------------------------------------- reporting
    # one row per subject, the earliest visit, so rows stay independent
    d = m.copy()
    d["_i"] = np.arange(len(d))
    first = d.sort_values(["PTID", "Years_bl"]).drop_duplicates("PTID")._i.values
    dx = m.DX.values[first]
    age = m.age_visit.values[first]
    cn_all = (dx == "CN")

    print("\n" + "=" * 74)
    print("THE ABLATION LADDER — does the TARGET fix the collapse?")
    print("=" * 74)
    print(f"  one row per subject: {len(first)}  "
          f"(CN {int((dx=='CN').sum())}, MCI {int((dx=='MCI').sum())}, "
          f"AD {int((dx=='Dementia').sum())})")

    # z implied by the predicted VOLUMES, i.e. exactly what E54 scored
    norm_all = fit_normative(m, m.DX.values == "CN")
    dv = m.copy()
    for j, v in enumerate(V):
        dv[v] = P_vol[:, j]
    dv["ICV"] = P_vol[:, 6]
    Z_from_vol = norm_all(dv)

    res = []
    res.append(report_arm("head V: raw volumes (the old target)",
                          Z_from_vol[first], Z_true[first], gap_img[first],
                          dx, age, cn_all))
    res.append(report_arm("head Z: normative deviations",
                          P_z[first], Z_true[first], gap_img[first],
                          dx, age, cn_all))
    res.append(report_arm("head W: whitened deviations",
                          P_w[first], Z_true[first], gap_img[first],
                          dx, age, cn_all))
    res.append(report_arm("FreeSurfer (the reference)",
                          Z_true[first], Z_true[first], gap_true[first],
                          dx, age, cn_all))

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  {'arm':40s} {'eff dim':>8s} {'gap':>7s} {'1 comp':>8s} "
          f"{'six':>7s} {'comp-gap':>9s}")
    for r in res:
        print(f"  {r['name']:40s} {r['eff_dim']:8.2f} {r['auc_gap']:7.3f} "
              f"{r['auc_pc']:8.3f} {r['auc_six']:7.3f} "
              f"{r['auc_pc']-r['auc_gap']:+9.3f}")

    np.savez(os.path.join(OUT, "whiten.npz"), P_vol=P_vol, P_z=P_z, P_w=P_w,
             Z_true=Z_true, gap_img=gap_img, gap_true=gap_true,
             first=first, fold=fold)
    outd = m[["PTID", "VISCODE", "DX", "age_visit", "SEX"]].copy()
    for j, v in enumerate(V):
        outd["zV_" + v] = Z_from_vol[:, j]
        outd["zZ_" + v] = P_z[:, j]
        outd["zW_" + v] = P_w[:, j]
        outd["zT_" + v] = Z_true[:, j]
    outd["gap_img"] = gap_img
    outd["gap_true"] = gap_true
    outd.to_csv(os.path.join(OUT, "deviations.csv"), index=False)
    print("\nwrote whiten.npz and deviations.csv", flush=True)


if __name__ == "__main__":
    main()
