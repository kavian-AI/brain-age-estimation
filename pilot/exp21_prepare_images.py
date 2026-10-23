#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare the OASIS-1 image dataset for a genuine image-based brain-age experiment.

The Kaggle release gives 2D axial slices in folders named by dementia severity,
but no age. The filenames however carry the subject id
(e.g. OAS1_0028_MR1_mpr-1_100.jpg), so we can join them to the OASIS-1
demographics table and recover AGE, CDR, sex, nWBV and eTIV.

Read straight from the archive: extracting 86,437 small JPEGs costs 1.3 GB of
disk we would rather leave for the ADNI download, and NTFS is slow with that
many files.

CRITICAL: each subject contributes ~200 slices across 3-4 repeat acquisitions.
Splitting by slice would put the same brain in train and test, which is the
known reason this dataset gets implausibly high published accuracies. Every
split downstream is by SUBJECT.
"""

import os
import re
import sys
import zipfile

import pandas as pd

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
ZIP = os.path.join(BASE, "Image data", "archive.zip")
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "pilot")

PAT = re.compile(r"(OAS1_\d{4})_MR(\d+)_mpr-(\d+)_(\d+)\.jpg$", re.I)


def scan():
    with zipfile.ZipFile(ZIP) as z:
        names = [m for m in z.namelist() if m.lower().endswith(".jpg")]
    rows = []
    for m in names:
        g = PAT.search(m.split("/")[-1])
        if not g:
            continue
        rows.append(dict(member=m,
                         subject=g.group(1),
                         mr=int(g.group(2)),
                         mpr=int(g.group(3)),
                         slice=int(g.group(4)),
                         folder=m.split("/")[1]))
    return pd.DataFrame(rows)


def main():
    if not os.path.exists(ZIP):
        print("archive not found:", ZIP)
        sys.exit(1)

    print("scanning archive ...", flush=True)
    df = scan()
    print(f"  images: {len(df):,}")
    print(f"  subjects: {df.subject.nunique()}")
    print(f"  acquisitions per subject: "
          f"{df.groupby('subject').mpr.nunique().value_counts().to_dict()}")
    print(f"  slices per acquisition: {df.groupby(['subject','mpr']).size().median():.0f}"
          f"  (index {df.slice.min()}-{df.slice.max()})")
    print("\n  folder labels:")
    print(df.folder.value_counts().to_string())

    # ---- join to demographics
    cs = pd.read_csv(os.path.join(DATA, "oasis_cross-sectional.csv"))
    cs["subject"] = cs["ID"].str.extract(r"(OAS1_\d{4})")
    cs = cs.dropna(subset=["subject"]).drop_duplicates("subject")
    keep = ["subject", "Age", "M/F", "CDR", "nWBV", "eTIV", "Educ", "SES", "MMSE"]
    m = df.merge(cs[keep], on="subject", how="left")

    ok = m.Age.notna()
    print(f"\n  slices matched to demographics: {ok.sum():,} / {len(m):,}")
    print(f"  subjects matched: {m[ok].subject.nunique()} "
          f"of {df.subject.nunique()} in archive "
          f"({cs.subject.nunique()} in the demographics table)")

    sub = m[ok].groupby("subject").first().reset_index()
    print(f"\n  subject-level summary (n={len(sub)}):")
    print(f"    age {sub.Age.min():.0f}-{sub.Age.max():.0f}, "
          f"mean {sub.Age.mean():.1f}")
    print("    CDR distribution:")
    print(sub.CDR.value_counts(dropna=False).sort_index().to_string())
    print(f"    healthy (CDR=0): {(sub.CDR == 0).sum()}, "
          f"impaired (CDR>0): {(sub.CDR > 0).sum()}, "
          f"CDR missing: {int(sub.CDR.isna().sum())}")

    # the folder label is derived from CDR; confirm it, because a mismatch would
    # mean the archive and the table are not describing the same scans
    print("\n  folder label vs CDR (sanity check):")
    print(pd.crosstab(sub.folder, sub.CDR.fillna(-1)).to_string())

    m[ok].to_csv(os.path.join(OUT, "images_index.csv"), index=False)
    print(f"\nsaved index -> {OUT}\\images_index.csv ({ok.sum():,} rows)")


if __name__ == "__main__":
    main()
