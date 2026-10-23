#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Harvest ABIDE-I FreeSurfer aseg.stats (subcortical volumes) from the open S3 bucket."""

import concurrent.futures as cf
import io
import re
import sys

import pandas as pd
import requests

BASE = "https://s3.amazonaws.com/fcp-indi"
PREFIX = "data/Projects/ABIDE_Initiative/Outputs/freesurfer/5.1/"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data\abide_aseg.csv"
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
    """Pull 'Measure' header values and the per-structure volume table."""
    out = {}
    for m in re.finditer(r"# Measure ([\w\-\.]+), [^,]*, [^,]*, ([\d\.\-eE]+),", text):
        out[m.group(1)] = float(m.group(2))
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 5:
            try:
                out[p[4]] = float(p[3])          # StructName -> Volume_mm3
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
        d["SUB_ID"] = int(sub.rsplit("_", 1)[-1])
        return d
    except Exception:
        return None


def main():
    subs = list_subjects()
    print(f"subjects found: {len(subs)}", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for i, res in enumerate(ex.map(fetch, subs), 1):
            if res:
                rows.append(res)
            if i % 100 == 0:
                print(f"  {i}/{len(subs)}  ok={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nsaved {df.shape} -> {OUT}")
    print("columns sample:", [c for c in df.columns][:18])


if __name__ == "__main__":
    main()
