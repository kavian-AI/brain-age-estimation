#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decode the OASIS slices out of the archive once and cache them.

Every subject has 3-4 repeat acquisitions (mpr-1..mpr-4) from the SAME session.
We keep them separate rather than averaging, because two scans of one brain
taken minutes apart give us something no other dataset here does: a direct
measurement of how much of the brain-age gap is measurement noise.

Intensity: JPEG brightness is arbitrary per acquisition, so each acquisition has
to be rescaled or the model reads scanner gain instead of anatomy. WHICH
statistic we divide by turns out to matter enormously. The obvious choice, the
mean of the non-background pixels, is wrong: atrophy replaces bright tissue with
dark CSF, so that mean is itself one of the strongest predictors of disease here
(AUC 0.398 on the age-matched subset). Dividing by it cancels the very signal we
are looking for - it drove a hand-made CSF feature from AUC 0.607 down to 0.523.

So we divide by the 95th percentile of the foreground instead, which tracks white
matter and is close to neutral with respect to diagnosis (AUC 0.512). Candidates
measured: p50 0.410, p75 0.397, p90 0.484, p95 0.512, p98 0.515, mean 0.398.

Normalising per acquisition (not per slice) preserves the relative brightness
between slices, which is anatomy.
"""

import io
import os
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
ZIP = os.path.join(BASE, "Image data", "archive.zip")
OUT = os.path.join(BASE, "pilot")
CACHE = os.path.join(OUT, "oasis_img_cache.npz")

SLICES = list(range(100, 161, 5))       # 13 axial levels, skull base -> vertex
H, W = 40, 80                           # keep the 1:2 aspect of the source
BG = 12                                 # below this is background, not tissue


def main():
    idx = pd.read_csv(os.path.join(OUT, "images_index.csv"))
    idx = idx[idx.slice.isin(SLICES)]
    print(f"slices selected: {len(idx):,} "
          f"({len(SLICES)} levels x {idx.groupby(['subject','mpr']).ngroups} acquisitions)")

    keys, X = [], []
    with zipfile.ZipFile(ZIP) as z:
        # a handful of subjects were scanned in two sessions (MR1 and MR2), so
        # the session must be part of the key or their acquisitions collide
        for n, ((sub, mr, mpr), g) in enumerate(
                idx.groupby(["subject", "mr", "mpr"]), 1):
            g = g.sort_values("slice")
            if len(g) != len(SLICES):
                continue                # incomplete acquisition, skip
            vol = []
            for m in g.member:
                im = Image.open(io.BytesIO(z.read(m))).convert("L")
                vol.append(np.asarray(im.resize((W, H), Image.BILINEAR),
                                      dtype=np.float32))
            v = np.stack(vol)                        # (levels, H, W)
            fg = v[v > BG]
            if fg.size < 100:
                continue
            v /= np.percentile(fg, 95)               # per-acquisition gain
            keys.append((sub, mr, mpr))
            X.append(v.reshape(-1))
            if n % 200 == 0:
                print(f"  {n} acquisitions decoded", flush=True)

    X = np.asarray(X, dtype=np.float32)
    k = pd.DataFrame(keys, columns=["subject", "mr", "mpr"])
    np.savez_compressed(CACHE, X=X, subject=k.subject.values,
                        mr=k.mr.values, mpr=k.mpr.values)
    print(f"\ncached {X.shape[0]} acquisitions x {X.shape[1]:,} pixels "
          f"({X.nbytes / 1e6:.0f} MB)")
    print(f"subjects: {k.subject.nunique()}")
    print(f"saved -> {CACHE}")


if __name__ == "__main__":
    main()
