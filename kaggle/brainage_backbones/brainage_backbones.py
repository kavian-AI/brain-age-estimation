#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHASE 1 - which pretrained backbone can read a brain MRI?

THE POINT OF WORKING FROM PIXELS
--------------------------------
Not to show that images beat volumes. The volumes are COMPUTED from the images,
so that contest is meaningless. The point is that in deployment there is no
FreeSurfer table waiting - there is a scan. Everything has to be reachable from
pixels, either

    image -> volumes -> age -> disease        (the anatomical bottleneck)
    image -> age -> disease                   (direct)

and this run measures how far a properly pretrained, fine-tuned backbone gets on
each. An earlier 3D convnet trained from scratch sat at chance (AUC 0.49-0.53 on
ADNI) on ~313 volumes per fold, which is a statement about sample size, not about
images. Pretrained features are the way past that.

WHAT IS COMPARED
----------------
Seven arms, one fold, identical data and schedule. Screening only - the winner
goes to a full 5-fold run in phase 2.

  vit_small / vit_base          supervised ImageNet-21k -> 1k
  dinov2_small                  self-supervised, no labels at all
  convnext_tiny, resnet50       modern and classic convolutional
  resnet50 SCRATCH              the control for "does pretraining matter?"
  vit_small FROZEN              linear probe - the control for "does
                                fine-tuning matter, or are the features
                                already enough?"

TASK A  image -> 6 FreeSurfer volumes + ICV     (scored by correlation)
TASK B  image -> age, healthy subjects only     (scored by MAE)

VERIFICATION
------------
The previous bottleneck run produced an empty log, so it was never confirmed that
the ImageNet weights actually downloaded - if they had silently failed, that
result was a from-scratch model wearing a pretrained label. Every arm here prints
whether its weights loaded, and a download failure is recorded as FAILED rather
than quietly falling back to random initialisation.

Splits are by SUBJECT. Intensity is divided by the 95th percentile of the
foreground, never the mean: atrophy lowers the mean, so dividing by it cancels
the disease signal (measured: a CSF feature fell from AUC 0.607 to 0.523).
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
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

OUT = "/kaggle/working"
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
TARGETS = V + ["ICV"]
FNAME = re.compile(r"^(\d{3}_S_\d{4,5})(?:_\d+)?_([A-Za-z]+\d*)\.npy$")

NPLANE = 8         # contiguous axial planes stored per scan (-> 6 triplets)
MAX_VISITS = 3     # cap per subject; screening ranks backbones, not ceilings
SIZE = 224
EPOCHS = 6
BATCH = 8
IMNET_MEAN, IMNET_STD = 0.485, 0.229

ARMS = [
    ("vit_small",      "vit_small_patch16_224.augreg_in21k_ft_in1k", True,  True),
    ("vit_base",       "vit_base_patch16_224.augreg_in21k_ft_in1k",  True,  True),
    ("dinov2_small",   "vit_small_patch14_dinov2.lvd142m",           True,  True),
    ("convnext_tiny",  "convnext_tiny.in12k_ft_in1k",                True,  True),
    ("resnet50",       "resnet50.a1_in1k",                           True,  True),
    ("resnet50_SCRATCH", "resnet50.a1_in1k",                         False, True),
    ("vit_small_FROZEN", "vit_small_patch16_224.augreg_in21k_ft_in1k", True, False),
]


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
        print(f"cannot find {what}; inputs:")
        for d in sorted(glob.glob("/kaggle/input/*")):
            print("  ", d, os.listdir(d)[:5])
        sys.exit(1)
    return h[0]


# ------------------------------------------------------------------- data

def load_data():
    a = pd.read_csv(find("ADNIMERGE*.csv", "ADNIMERGE"), low_memory=False)
    keep = ["PTID", "VISCODE", "AGE", "PTGENDER", "DX", "ICV", "Years_bl"] + V
    lut = a[keep].copy()
    lut["key"] = lut.PTID + "|" + lut.VISCODE.astype(str)
    lut = lut.drop_duplicates("key").set_index("key")

    root = os.path.dirname(find("*_S_*.npy", "the volume archive"))
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
        if pd.isna(r.AGE) or r[TARGETS].isna().any():
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
        rows.append(dict(PTID=g.group(1), VISCODE=vis, AGE=r.AGE, DX=r.DX,
                         Years_bl=r.Years_bl,
                         SEX=1 if r.PTGENDER == "Male" else 0,
                         **{k: r[k] for k in TARGETS}))
        if len(rows) % 300 == 0:
            print(f"  loaded {len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    m = pd.DataFrame(rows)
    # cap visits per subject: screening only needs enough data to rank
    # backbones, and the tail of long follow-up series adds time, not signal
    m["_i"] = np.arange(len(m))
    keep = m.sort_values("Years_bl").groupby("PTID").head(MAX_VISITS)._i.values
    keep.sort()
    print(f"  kept {len(keep)} of {len(m)} scans "
          f"(<= {MAX_VISITS} visits/subject)", flush=True)
    return np.stack(X)[keep], m.iloc[keep].drop(columns="_i").reset_index(drop=True)


class DS(torch.utils.data.Dataset):
    """Each sample is one scan: 8 axial levels, each an RGB triplet of its
    neighbouring planes so the backbone sees a little local 3D context in the
    channels it was pretrained to expect."""

    def __init__(self, X, Y, train):
        self.X, self.Y, self.train = X, Y, train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        v = torch.from_numpy(self.X[i].astype(np.float32) / 127.5)
        trip = torch.stack([v[k - 1:k + 2] for k in range(1, NPLANE - 1)])
        if self.train:
            if np.random.rand() < .5:
                trip = torch.flip(trip, [3])
            ang = np.random.uniform(-8, 8)
            tx, ty = np.random.uniform(-.05, .05, 2)
            a = np.deg2rad(ang)
            m = torch.tensor([[np.cos(a), -np.sin(a), tx],
                              [np.sin(a), np.cos(a), ty]],
                             dtype=torch.float32).unsqueeze(0).repeat(len(trip), 1, 1)
            g = F.affine_grid(m, trip.shape, align_corners=False)
            trip = F.grid_sample(trip, g, align_corners=False)
            trip = trip * np.random.uniform(.92, 1.08)
        trip = (trip - IMNET_MEAN) / IMNET_STD
        return trip, torch.from_numpy(self.Y[i])


# ------------------------------------------------------------------ model

class Backbone(nn.Module):
    def __init__(self, timm_name, pretrained, nout):
        super().__init__()
        import timm
        self.loaded = False
        try:
            kw = dict(pretrained=pretrained, num_classes=0)
            if "dinov2" in timm_name:
                kw["img_size"] = SIZE
            self.net = timm.create_model(timm_name, **kw)
            self.loaded = bool(pretrained)
        except Exception as e:
            print(f"    !! weight download FAILED ({type(e).__name__}: "
                  f"{str(e)[:80]}); this arm is INVALID", flush=True)
            self.net = timm.create_model(timm_name, pretrained=False,
                                         num_classes=0)
            if "dinov2" in timm_name:
                self.net = timm.create_model(timm_name, pretrained=False,
                                             num_classes=0, img_size=SIZE)
            self.failed = True
        self.dim = self.net.num_features
        self.head = nn.Sequential(nn.LayerNorm(self.dim), nn.Dropout(.2),
                                  nn.Linear(self.dim, nout))

    def features(self, x):
        b, s, c, h, w = x.shape
        z = self.net(x.reshape(b * s, c, h, w))
        return z.reshape(b, s, -1).mean(1)

    def forward(self, x):
        return self.head(self.features(x))


def run_arm(name, timm_name, pretrained, finetune, Xtr, Ytr, Xte, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Backbone(timm_name, pretrained, Ytr.shape[1]).to(dev)
    tag = ("pretrained" if getattr(net, "loaded", False) else "RANDOM INIT") + \
          ("" if finetune else " + FROZEN backbone")
    print(f"    weights: {tag}   feature dim {net.dim}", flush=True)
    if not finetune:
        for p in net.net.parameters():
            p.requires_grad = False

    groups = [{"params": net.head.parameters(), "lr": 1e-3}]
    if finetune:
        lr = 3e-5 if "vit" in timm_name else 1e-4
        groups.append({"params": net.net.parameters(), "lr": lr})
    opt = torch.optim.AdamW(groups, weight_decay=.02)
    dl = torch.utils.data.DataLoader(DS(Xtr, Ytr, True), batch_size=BATCH,
                                     shuffle=True, drop_last=True, num_workers=2)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, [g["lr"] * 4 for g in groups], epochs=EPOCHS,
        steps_per_epoch=max(1, len(dl)))
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    lf = nn.SmoothL1Loss()
    net.train()
    for ep in range(EPOCHS):
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
        print(f"      epoch {ep} loss {tot/max(1,len(dl)):.4f}", flush=True)

    net.eval()
    o = []
    with torch.no_grad():
        dte = torch.utils.data.DataLoader(DS(Xte, np.zeros((len(Xte), 1),
                                                           dtype=np.float32), False),
                                          batch_size=16, num_workers=2)
        for xb, _ in dte:
            with torch.cuda.amp.autocast(enabled=dev == "cuda"):
                o.append(net(xb.to(dev)).float().cpu().numpy())
    return np.concatenate(o), getattr(net, "loaded", False)


def main():
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    import timm
    print("timm", timm.__version__, flush=True)

    X, m = load_data()
    print(f"\nbaseline scans: {len(m)}, subjects: {m.PTID.nunique()}")
    print(m.DX.value_counts(dropna=False).to_dict())
    print(f"tensor {X.shape} {X.nbytes/1e9:.2f} GB uint8", flush=True)

    sub = m.PTID.values
    uq = np.unique(sub)
    rng = np.random.RandomState(0)
    te_sub = set(uq[rng.permutation(len(uq))[:len(uq) // 5]])
    te = np.array([s in te_sub for s in sub])
    tr = ~te
    print(f"train {tr.sum()}  test {te.sum()} (split by subject)")

    Yv = m[TARGETS].values.astype(np.float32)
    ysc = StandardScaler().fit(Yv[tr])
    Yvz = ysc.transform(Yv).astype(np.float32)

    cn = (m.DX == "CN").values
    # ADNIMERGE AGE is age at BASELINE; follow-up scans are older than that
    age = (m.AGE + m.Years_bl).values.astype(np.float32)
    asc = StandardScaler().fit(age[tr & cn].reshape(-1, 1))
    agez = asc.transform(age.reshape(-1, 1)).astype(np.float32)

    res = {}
    for name, tn, pre, ft in ARMS:
        print("\n" + "=" * 72)
        print(f"ARM  {name}   ({tn})")
        print("=" * 72, flush=True)
        t0 = time.time()

        print("  TASK A  image -> 7 FreeSurfer numbers", flush=True)
        pa, ok = run_arm(name, tn, pre, ft, X[tr], Yvz[tr], X[te])
        Pa = ysc.inverse_transform(pa)
        rs = [np.corrcoef(Pa[:, j], Yv[te][:, j])[0, 1] for j in range(len(TARGETS))]

        print("  TASK B  image -> age (healthy only)", flush=True)
        pb, _ = run_arm(name, tn, pre, ft, X[tr & cn], agez[tr & cn], X[te & cn])
        Pb = asc.inverse_transform(pb).ravel()
        mae = np.abs(Pb - age[te & cn]).mean()
        rb = np.corrcoef(Pb, age[te & cn])[0, 1]

        res[name] = dict(r=rs, mean_r=float(np.mean(rs)), mae=float(mae),
                         r_age=float(rb), pretrained=bool(ok),
                         mins=(time.time() - t0) / 60)
        print(f"\n  >> mean r over targets {np.mean(rs):.3f}   "
              f"age MAE {mae:.2f} y (r={rb:.3f})   [{res[name]['mins']:.1f} min]",
              flush=True)
        for t, r in zip(TARGETS, rs):
            print(f"       {t:16s} r={r:.3f}")
        torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("PHASE 1 SUMMARY   (1 fold, screening only)")
    print("=" * 72)
    print(f"  {'arm':20s} {'weights':12s} {'mean r':>7} {'age MAE':>9} {'r age':>7}")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["mean_r"]):
        print(f"  {k:20s} {'pretrained' if v['pretrained'] else 'scratch':12s} "
              f"{v['mean_r']:7.3f} {v['mae']:8.2f}y {v['r_age']:7.3f}")

    # the two controls the design exists to test
    if "resnet50" in res and "resnet50_SCRATCH" in res:
        d = res["resnet50"]["mean_r"] - res["resnet50_SCRATCH"]["mean_r"]
        print(f"\n  pretraining effect (resnet50 - scratch): mean r {d:+.3f}, "
              f"age MAE {res['resnet50_SCRATCH']['mae']-res['resnet50']['mae']:+.2f} y")
    if "vit_small" in res and "vit_small_FROZEN" in res:
        d = res["vit_small"]["mean_r"] - res["vit_small_FROZEN"]["mean_r"]
        print(f"  fine-tuning effect (vit_small - frozen): mean r {d:+.3f}, "
              f"age MAE {res['vit_small_FROZEN']['mae']-res['vit_small']['mae']:+.2f} y")

    pd.DataFrame(res).T.to_csv(f"{OUT}/phase1_backbones.csv")
    print(f"\nsaved -> {OUT}/phase1_backbones.csv")


if __name__ == "__main__":
    main()
