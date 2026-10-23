#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIAMESE DELTA MODEL - can a network be made sensitive to CHANGE?

E32 measured that image-predicted volumes report only part of the change
FreeSurfer measures: 94% for ventricles, 53% hippocampus, 13% entorhinal, 42% on
average. Any rate built on them is shrunk toward zero before a survival model
sees it, which is enough to explain why every image-derived rate here failed
(0.509, 0.521, 0.551) while cross-sectional measures reached 0.836.

E33 tested the cheap fix - regress the volume difference on the FEATURE
difference, using frozen features - and it collapsed completely (attenuation
-0.002). Frozen features are dominated by subject identity, so differencing two
visits of one person cancels almost everything informative. That refutes the
cheap version, not the idea.

This is the proper version. A shared pretrained backbone sees BOTH scans of a
pair and a head predicts the change from [f1, f2, f2-f1], so the network can
learn features sensitive to change instead of having change subtracted out of
features chosen for absolute accuracy.

CRITICAL DETAIL: the SAME spatial augmentation is applied to both scans of a
pair. Augmenting them independently injects apparent change that is pure noise
and would corrupt the very quantity being measured.

Related published work: Regional Deep Atrophy trains on longitudinal pairs to
localise AD progression. It optimises classification of progression; this run
measures ATTENUATION - how much of the real change is reported - which is a
different quantity.

Splits are by SUBJECT: a subject's pairs never straddle train and test.
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
from sklearn.preprocessing import StandardScaler

OUT = "/kaggle/working"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
TARGETS = V + ["ICV"]
FNAME = re.compile(r"^(\d{3}_S_\d{4,5})(?:_\d+)?_([A-Za-z]+\d*)\.npy$")

BACKBONE = "resnet18.a1_in1k"
NPLANE, SIZE = 14, 224
FOLDS, EPOCHS_S, BATCH = 5, 15, 4
MIN_GAP = 0.75
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
                         **{k: r[k] for k in TARGETS}))
        if len(rows) % 500 == 0:
            print(f"  loaded {len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    return np.stack(X), pd.DataFrame(rows)


class Siam(nn.Module):
    """Shared backbone over both scans; head sees f1, f2 and their difference."""

    _announced = False

    def __init__(self, nout):
        super().__init__()
        import timm
        self.net = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
        d = self.net.num_features
        if not Siam._announced:
            w = next(self.net.parameters()).detach()
            print(f"    {BACKBONE}: PRETRAINED loaded, dim {d}, "
                  f"first-layer std {w.std():.4f}", flush=True)
            Siam._announced = True
        self.head = nn.Sequential(nn.LayerNorm(3 * d), nn.Dropout(.2),
                                  nn.Linear(3 * d, nout))

    def feat(self, x):
        b, s, c, h, w = x.shape
        return self.net(x.reshape(b * s, c, h, w)).reshape(b, s, -1).mean(1)

    def forward(self, x1, x2):
        f1, f2 = self.feat(x1), self.feat(x2)
        return self.head(torch.cat([f1, f2, f2 - f1], 1))


class PairDS(torch.utils.data.Dataset):
    def __init__(self, X, pi, pj, Y, train):
        self.X, self.pi, self.pj, self.Y, self.train = X, pi, pj, Y, train

    def __len__(self):
        return len(self.Y)

    def _prep(self, idx):
        v = torch.from_numpy(self.X[idx].astype(np.float32) / 127.5)
        return torch.stack([v[k - 1:k + 2] for k in range(1, NPLANE - 1)])

    def __getitem__(self, k):
        a, b = self._prep(self.pi[k]), self._prep(self.pj[k])
        if self.train:
            # ONE transform applied to BOTH scans. Augmenting them independently
            # would manufacture apparent change that is not in the data.
            if np.random.rand() < .5:
                a, b = torch.flip(a, [3]), torch.flip(b, [3])
            ang = np.deg2rad(np.random.uniform(-8, 8))
            tx, ty = np.random.uniform(-.05, .05, 2)
            m = torch.tensor([[np.cos(ang), -np.sin(ang), tx],
                              [np.sin(ang), np.cos(ang), ty]],
                             dtype=torch.float32).unsqueeze(0).repeat(len(a), 1, 1)
            g = F.affine_grid(m, a.shape, align_corners=False)
            a = F.grid_sample(a, g, align_corners=False)
            b = F.grid_sample(b, g, align_corners=False)
            sc = np.random.uniform(.92, 1.08)
            a, b = a * sc, b * sc
        return ((a - IMNET_MEAN) / IMNET_STD, (b - IMNET_MEAN) / IMNET_STD,
                torch.from_numpy(self.Y[k]))


def train_siam(X, pi, pj, Y, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Siam(Y.shape[1]).to(dev)
    opt = torch.optim.AdamW([{"params": net.net.parameters(), "lr": 1e-4},
                             {"params": net.head.parameters(), "lr": 1e-3}],
                            weight_decay=.02)
    dl = torch.utils.data.DataLoader(PairDS(X, pi, pj, Y, True),
                                     batch_size=BATCH, shuffle=True,
                                     drop_last=len(Y) > BATCH, num_workers=2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [4e-4, 4e-3], epochs=EPOCHS_S,
                                              steps_per_epoch=max(1, len(dl)))
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    lf = nn.SmoothL1Loss()
    net.train()
    for ep in range(EPOCHS_S):
        tot = 0
        for a, b, y in dl:
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=dev == "cuda"):
                loss = lf(net(a.to(dev), b.to(dev)), y.to(dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            tot += loss.item()
        if ep % 4 == 0 or ep == EPOCHS_S - 1:
            print(f"      ep {ep:2d} loss {tot/max(1,len(dl)):.4f}", flush=True)
    return net


@torch.no_grad()
def infer_siam(net, X, pi, pj):
    net.eval()
    o = []
    dl = torch.utils.data.DataLoader(
        PairDS(X, pi, pj, np.zeros((len(pi), 1), dtype=np.float32), False),
        batch_size=8, num_workers=2)
    for a, b, _ in dl:
        with torch.cuda.amp.autocast(enabled=dev == "cuda"):
            o.append(net(a.to(dev), b.to(dev)).float().cpu().numpy())
    return np.concatenate(o)


def main():
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    X, m = load_data()
    print(f"scans {len(m)}  subjects {m.PTID.nunique()}", flush=True)

    rows = []
    for s, g in m.groupby("PTID"):
        g = g.sort_values("Years_bl")
        ii = g.index.values
        for a in range(len(ii)):
            for b in range(a + 1, len(ii)):
                dt = g.Years_bl.iloc[b] - g.Years_bl.iloc[a]
                if dt >= MIN_GAP:
                    rows.append(dict(PTID=s, i=ii[a], j=ii[b], dt=dt))
    P = pd.DataFrame(rows)
    print(f"visit pairs: {len(P)} from {P.PTID.nunique()} subjects "
          f"(median interval {P.dt.median():.1f} y)", flush=True)

    Yv = m[TARGETS].values.astype(np.float32)
    dY = (Yv[P.j.values] - Yv[P.i.values]).astype(np.float32)
    dsc = StandardScaler().fit(dY)
    dYz = dsc.transform(dY).astype(np.float32)

    uq = np.unique(P.PTID.values)
    fmap = dict(zip(uq, np.random.RandomState(0).permutation(len(uq)) % FOLDS))
    fp = np.array([fmap[s] for s in P.PTID.values])

    pred = np.zeros_like(dYz)
    for k in range(FOLDS):
        tr, te = np.where(fp != k)[0], np.where(fp == k)[0]
        print(f"\nfold {k}: train pairs {len(tr)}, test {len(te)}", flush=True)
        net = train_siam(X, P.i.values[tr], P.j.values[tr], dYz[tr], seed=k)
        pred[te] = infer_siam(net, X, P.i.values[te], P.j.values[te])
        del net
        torch.cuda.empty_cache()
    D = dsc.inverse_transform(pred)

    print("\n" + "=" * 76)
    print("SIAMESE DELTA MODEL - how much of the real change is reported?")
    print("=" * 76)
    print(f"  {'target':16s} {'sensitivity':>12} {'attenuation':>12}   E31 absolute")
    e31 = dict(Hippocampus=.529, Entorhinal=.128, Fusiform=.260, MidTemp=.292,
               Ventricles=.940, WholeBrain=.425, ICV=.393)
    ss, aa = [], []
    for j, t in enumerate(TARGETS):
        sens = np.corrcoef(D[:, j], dY[:, j])[0, 1]
        att = np.polyfit(dY[:, j], D[:, j], 1)[0]
        ss.append(sens)
        aa.append(att)
        print(f"  {t:16s} {sens:12.3f} {att:12.3f}   {e31[t]:12.3f}")
    print(f"  {'MEAN':16s} {np.mean(ss):12.3f} {np.mean(aa):12.3f}   {0.424:12.3f}")
    print(f"\n  change in attenuation vs E31 absolute: {np.mean(aa)-0.424:+.3f}")
    if np.mean(aa) > 0.474:
        print("  -> the siamese target DOES make the model change-sensitive.")
    elif np.mean(aa) < 0.374:
        print("  -> WORSE than differencing an absolute model.")
    else:
        print("  -> no material gain: attenuation is a property of the")
        print("     representation, not of how the target is posed.")

    np.savez(f"{OUT}/siamese.npz", dpred=D, dtrue=dY, subs=P.PTID.values,
             i=P.i.values, j=P.j.values, dt=P.dt.values)
    m.to_csv(f"{OUT}/siamese_meta.csv", index=False)
    print(f"\nsaved -> {OUT}/siamese.npz")


if __name__ == "__main__":
    main()
