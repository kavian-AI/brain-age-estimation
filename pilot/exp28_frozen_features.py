#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pretrained backbones on ADNI, without a GPU: frozen features.

WHY THIS SHAPE OF EXPERIMENT
----------------------------
Fine-tuning needs backprop and therefore a GPU, and the Kaggle account currently
has neither GPU nor internet (both gated behind phone verification - the last
run silently fell back to CPU with RANDOM weights and burned 20 h before being
cancelled). A forward pass needs neither. So this runs the backbones frozen, as
feature extractors, which is a weaker method than fine-tuning but a real test of
the question underneath: are ImageNet/self-supervised features useful for brain
MRI at all?

It also isolates the effect of pretraining cleanly. The comparison

    frozen pretrained features   vs   voxel PCA on the same slices

changes only the representation - both are fixed, neither is trained on the
target - so any difference is attributable to the pretraining itself rather than
to capacity or optimisation.

PIPELINE (identical to every earlier experiment downstream)
    image -> frozen features -> 7 FreeSurfer volumes -> age -> gap -> disease
    image -> frozen features -> age -> gap -> disease
    image -> frozen features + age -> disease            (fair baseline)
    true FreeSurfer volumes + age -> disease             (ceiling, not reachable)

Volumes are streamed from the 5.9 GB archive: it expands to 33.6 GB (93 %
background zeros) and extracting it is not affordable.

Each axial level is presented as an RGB triplet of neighbouring planes, so the
backbone receives local 3D context in the three channels it was pretrained to
expect rather than one grey plane repeated.

Intensity is divided by the foreground p95, never the mean - atrophy lowers the
mean, so dividing by it cancels the disease signal (measured: a CSF feature fell
from AUC 0.607 to 0.523).

Splits are by SUBJECT throughout.
"""

import io
import os
import re
import sys
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
ZIP = os.path.join(BASE, "adni_img", "adni-mri-tensors.zip")
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "pilot")
BASELINE_ONLY = os.environ.get("ALLVISITS", "0") != "1"
# separate cache per mode, so an all-visits run cannot clobber the baseline one
SUF = "" if BASELINE_ONLY else "_allvisits"
CACHE = os.path.join(OUT, f"adni_frozen_feats{SUF}.npz")
META = os.path.join(OUT, f"adni_frozen_meta{SUF}.csv")

V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
TARGETS = V + ["ICV"]
FNAME = re.compile(r"(\d{3}_S_\d{4,5})(?:_\d+)?_([A-Za-z]+\d*)\.npy$")

NPLANE, SIZE = 8, 224          # 8 planes -> 6 RGB triplets
IMNET_MEAN, IMNET_STD = 0.485, 0.229
BACKBONES = [
    ("vit_small", "vit_small_patch16_224.augreg_in21k_ft_in1k"),
    ("resnet50", "resnet50.a1_in1k"),
    ("convnext_tiny", "convnext_tiny.in12k_ft_in1k"),
]
def build_lut():
    a = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"), low_memory=False)
    keep = ["PTID", "VISCODE", "AGE", "PTGENDER", "DX", "ICV", "Years_bl"] + V
    lut = a[keep].copy()
    lut["key"] = lut.PTID + "|" + lut.VISCODE.astype(str)
    return lut.drop_duplicates("key").set_index("key")


def planes_from(v):
    """NPLANE axial levels from the middle of the brain, resized, p95-scaled."""
    fg = v[v > v.max() * 0.08]
    if fg.size < 1000:
        return None
    v = v / np.percentile(fg, 95)
    z = np.linspace(int(v.shape[2] * .22), int(v.shape[2] * .78), NPLANE).astype(int)
    t = torch.from_numpy(np.stack([v[:, :, k] for k in z])).unsqueeze(0)
    t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0]
    return torch.clamp(t, 0, 2)


def main():
    import timm
    lut = build_lut()
    models = {}
    for nm, tn in BACKBONES:
        print(f"loading {nm} ...", flush=True)
        m = timm.create_model(tn, pretrained=True, num_classes=0).eval()
        assert m.num_features > 0
        models[nm] = m
        print(f"  {nm}: pretrained OK, dim {m.num_features}", flush=True)

    feats = {nm: [] for nm in models}
    rows = []
    t0 = time.time()
    with zipfile.ZipFile(ZIP) as z:
        members = [m for m in z.namelist() if m.endswith(".npy")]
        print(f"\nmembers: {len(members)}  (baseline only: {BASELINE_ONLY})",
              flush=True)
        for i, mem in enumerate(members, 1):
            g = FNAME.search(os.path.basename(mem))
            if not g:
                continue
            vis = "bl" if g.group(2) == "sc" else g.group(2)
            if re.fullmatch(r"v\d+", vis):
                continue
            if BASELINE_ONLY and vis != "bl":
                continue
            key = f"{g.group(1)}|{vis}"
            if key not in lut.index:
                continue
            r = lut.loc[key]
            if pd.isna(r.AGE) or pd.isna(r.Years_bl) or r[TARGETS].isna().any():
                continue
            try:
                with z.open(mem) as fh:
                    v = np.load(io.BytesIO(fh.read())).astype(np.float32)
            except Exception:
                continue
            if v.ndim != 3:
                continue
            p = planes_from(v)
            if p is None:
                continue
            trip = torch.stack([p[k - 1:k + 2] for k in range(1, NPLANE - 1)])
            trip = (trip - IMNET_MEAN) / IMNET_STD
            with torch.no_grad():
                for nm, m in models.items():
                    feats[nm].append(m(trip).mean(0).numpy().astype(np.float32))
            rows.append(dict(PTID=g.group(1), VISCODE=vis, AGE=r.AGE,
                             Years_bl=r.Years_bl, DX=r.DX,
                             SEX=1 if r.PTGENDER == "Male" else 0,
                             **{k: r[k] for k in TARGETS}))
            if len(rows) % 25 == 0:
                el = time.time() - t0
                print(f"  {len(rows)} scans  {el/60:.1f} min "
                      f"({el/len(rows):.2f} s/scan)", flush=True)

    meta = pd.DataFrame(rows)
    meta["age_visit"] = meta.AGE + meta.Years_bl
    np.savez_compressed(CACHE, **{f"F_{k}": np.stack(v) for k, v in feats.items()})
    meta.to_csv(META, index=False)
    print(f"\ncached {len(meta)} scans, {meta.PTID.nunique()} subjects")
    for nm in models:
        print(f"  {nm}: {np.stack(feats[nm]).shape}")
    print(f"saved -> {CACHE}")


if __name__ == "__main__":
    main()
