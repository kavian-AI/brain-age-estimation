#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stream the ADNI volumes straight out of the zip, downsample, and cache.

The archive holds 9,072 volumes of 128^3 float32 = 33.6 GB uncompressed, but it
is only 5.9 GB on disk because the volumes are mostly background zeros.
Extracting it would need 33 GB we do not have, so each member is read from the
archive into memory, downsampled, and discarded. Output is ~0.5 GB.

Downsampling by block-mean rather than decimation, so we average instead of
sampling - closer to the smoothing step in the classic brain-age pipeline, and
it does not alias.

Naming. Two conventions appear in the archive:
    002_S_0295_m06.npy        ADNI 1/GO/2, four-digit id
    002_S_10814_4_sc.npy      ADNI 4, five-digit id with a phase suffix
Only the first kind is expected to join, because ADNIMERGE_05Mar2026 uses
four-digit PTIDs, but both are parsed so the join failure is visible rather than
silent.

Visit codes. ADNIMERGE uses bl/m03/m06/..., while the archive also carries the
ADNI-GO/2 v-codes (v02, v11, v21, ...). Mapping those correctly needs the visit
key table, which we do not have, so v-coded scans are counted and skipped rather
than guessed at - a wrong visit code would silently pair a scan with the wrong
age and diagnosis.
"""

import io
import os
import re
import sys
import time
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
ZIP = os.path.join(BASE, "adni_img", "adni-mri-tensors.zip")
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "pilot")
CACHE = os.path.join(OUT, "adni_vox32.npz")

FACTOR = 4                                        # 128 -> 32
V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
FNAME = re.compile(r"^(\d{3}_S_\d{4,5})(?:_\d+)?_([A-Za-z]+\d*)\.npy$")


def block_mean(a, f):
    s = a.shape[0] // f
    return a[:s * f, :s * f, :s * f].reshape(s, f, s, f, s, f).mean(axis=(1, 3, 5))


def main():
    if not os.path.exists(ZIP):
        print("zip not found:", ZIP)
        sys.exit(1)

    adni = pd.read_csv(os.path.join(DATA, "ADNIMERGE_05Mar2026.csv"),
                       low_memory=False)
    lut = adni[["PTID", "VISCODE", "AGE", "PTGENDER", "DX", "ICV"] + V].copy()
    lut["key"] = lut.PTID + "|" + lut.VISCODE.astype(str)
    lut = lut.drop_duplicates("key").set_index("key")

    with zipfile.ZipFile(ZIP) as z:
        members = [m for m in z.namelist() if m.endswith(".npy")]
        print(f"members in archive: {len(members)}")

        # decide what to read BEFORE decompressing anything
        plan, skip = [], Counter()
        for m in members:
            g = FNAME.match(os.path.basename(m))
            if not g:
                skip["unparsed filename"] += 1
                continue
            ptid, vis = g.group(1), g.group(2)
            if vis == "sc":
                vis = "bl"                       # screening scan is the baseline
            if re.fullmatch(r"v\d+", vis):
                skip["ADNI-GO/2 v-code, unmappable"] += 1
                continue
            key = f"{ptid}|{vis}"
            if key not in lut.index:
                skip["no matching ADNIMERGE visit"] += 1
                continue
            plan.append((m, key))

        print(f"to read: {len(plan)}")
        for k, v in skip.most_common():
            print(f"  skipped, {k}: {v}")

        rows, vox, t0 = [], [], time.time()
        for i, (m, key) in enumerate(plan, 1):
            try:
                with z.open(m) as fh:
                    a = np.load(io.BytesIO(fh.read()))
            except Exception as e:
                skip[f"unreadable ({type(e).__name__})"] += 1
                continue
            if a.ndim != 3 or a.shape[0] < FACTOR:
                skip["unexpected shape"] += 1
                continue
            vox.append(block_mean(a.astype(np.float32), FACTOR).reshape(-1))
            r = lut.loc[key]
            rows.append(dict(PTID=key.split("|")[0], VISCODE=key.split("|")[1],
                             AGE=r.AGE, SEX=1 if r.PTGENDER == "Male" else 0,
                             DX=r.DX, ICV=r.ICV, **{v: r[v] for v in V}))
            if i % 500 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(plan)}  kept={len(rows)}  "
                      f"{el:.0f}s  eta {el/i*(len(plan)-i):.0f}s", flush=True)

    X = np.asarray(vox, dtype=np.float32)
    meta = pd.DataFrame(rows)
    np.savez_compressed(CACHE, X=X)
    meta.to_csv(os.path.join(OUT, "adni_vox32_meta.csv"), index=False)

    print(f"\ncached {X.shape[0]} volumes x {X.shape[1]:,} voxels "
          f"({X.nbytes/1e6:.0f} MB)")
    print(f"subjects: {meta.PTID.nunique()}")
    print("\nvisits per subject:")
    print(meta.groupby("PTID").size().value_counts().sort_index().to_string())
    print("\nDX at baseline:")
    bl = meta[meta.VISCODE == "bl"]
    print(bl.DX.value_counts(dropna=False).to_string())
    print(f"\nbaseline scans with full volumetrics + DX: "
          f"{bl.dropna(subset=['DX', 'ICV'] + V).shape[0]}")
    print(f"saved -> {CACHE}")


if __name__ == "__main__":
    main()
