#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Harvest ADHD-200 FreeSurfer aseg.stats from the open fcp-indi S3 bucket,
and merge with the per-site phenotypic CSVs."""

import concurrent.futures as cf
import glob
import os
import re

import pandas as pd
import requests

BASE = "https://s3.amazonaws.com/fcp-indi"
PREFIX = "data/Projects/ADHD200/surfaces/freesurfer/5.3/"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "adhd200_merged.csv")
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0"


def list_subjects():
    subs, token = [], None
    while True:
        url = f"{BASE}?list-type=2&prefix={PREFIX}&delimiter=/&max-keys=1000"
        if token:
            url += f"&continuation-token={requests.utils.quote(token, safe='')}"
        r = S.get(url, timeout=60)
        subs += [p.rsplit("/", 2)[-2] for p in
                 re.findall(r"<Prefix>([^<]*)</Prefix>", r.text)
                 if p.rstrip("/") != PREFIX.rstrip("/")]
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", r.text)
        if not m:
            break
        token = m.group(1)
    return sorted(set(subs))


def parse_aseg(text):
    out = {}
    for m in re.finditer(r"# Measure ([\w\-\.]+), [^,]*, [^,]*, ([\d\.\-eE]+),", text):
        out[m.group(1)] = float(m.group(2))
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 5:
            try:
                out[p[4]] = float(p[3])
            except ValueError:
                pass
    return out


def fetch(sub):
    try:
        r = S.get(f"{BASE}/{PREFIX}{sub}/stats/aseg.stats", timeout=45)
        if r.status_code != 200:
            return None
        d = parse_aseg(r.text)
        if not d:
            return None
        d["subject"] = sub
        d["ScanDirID"] = int(sub)
        return d
    except Exception:
        return None


def load_pheno():
    frames = []
    for f in glob.glob(os.path.join(HERE, "adhd_pheno", "*_phenotypic.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df.columns = [c.strip() for c in df.columns]
        idc = next((c for c in df.columns if c.replace(" ", "").lower()
                    in ("scandirid", "scandir_id")), None)
        if idc is None or "DX" not in df.columns:
            continue
        keep = df[[idc, "DX", "Age", "Gender", "Site"]].copy()
        keep.columns = ["ScanDirID", "DX", "Age", "Gender", "Site"]
        keep["source"] = os.path.basename(f)
        frames.append(keep)
    p = pd.concat(frames, ignore_index=True)
    p["ScanDirID"] = pd.to_numeric(p["ScanDirID"], errors="coerce")
    p = p.dropna(subset=["ScanDirID"]).drop_duplicates("ScanDirID")
    p["ScanDirID"] = p["ScanDirID"].astype(int)
    return p


def main():
    subs = list_subjects()
    print(f"subjects on S3: {len(subs)}", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for i, res in enumerate(ex.map(fetch, subs), 1):
            if res:
                rows.append(res)
            if i % 150 == 0:
                print(f"  {i}/{len(subs)} ok={len(rows)}", flush=True)
    vol = pd.DataFrame(rows)
    print("volumes:", vol.shape)

    ph = load_pheno()
    print("phenotypes:", ph.shape)
    m = vol.merge(ph, on="ScanDirID", how="inner")
    m.to_csv(OUT, index=False)
    print(f"merged {m.shape} -> {OUT}")
    print(m.DX.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
