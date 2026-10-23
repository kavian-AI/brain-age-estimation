# Pooled healthy reference in FastSurfer space: IXI + DLBS (+ OASIS-2 nondemented).
#
# All three were segmented with the same program (FastSurfer v2.4.2, aseg + DKT),
# so their regional volumes are directly comparable; ADNI (FreeSurfer 4.3/5.1
# tables) stays separate and is used for replication only.
#
# load_ixi(), load_dlbs(), load_oasis2() return one tidy frame each with columns
#   dataset, subject, scan, age, sex (1 = male), icv, site, wave, <region volumes>
# where icv is the SynthStrip intracranial volume (inner skull incl. CSF; Kaggle
# icv-synthstrip-* kernels), the same measure for all three datasets. The older
# proxies are kept as icv_mask (FastSurfer brain mask; shrinks with age) and
# etiv (OASIS-2 published eTIV).
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAG = os.path.join(ROOT, "kaggle")

# bilateral composites (left + right labels) of FastSurfer aseg / DKT labels
BILAT = {
    "hippocampus": (17, 53), "amygdala": (18, 54), "lateral_ventricles": (4, 43),
    "inf_lat_ventricles": (5, 44), "thalamus": (10, 49), "caudate": (11, 50),
    "putamen": (12, 51), "pallidum": (13, 52), "accumbens": (26, 58),
    "ventral_dc": (28, 60), "cerebellum_cortex": (8, 47), "cerebellum_wm": (7, 46),
    "cerebral_wm": (2, 41), "choroid_plexus": (31, 63),
}
SINGLE = {"third_ventricle": 14, "fourth_ventricle": 15, "brainstem": 16, "csf": 24, "wm_hypo": 77}
DKT = {
    2: "caudalanteriorcingulate", 3: "caudalmiddlefrontal", 5: "cuneus", 6: "entorhinal",
    7: "fusiform", 8: "inferiorparietal", 9: "inferiortemporal", 10: "isthmuscingulate",
    11: "lateraloccipital", 12: "lateralorbitofrontal", 13: "lingual", 14: "medialorbitofrontal",
    15: "middletemporal", 16: "parahippocampal", 17: "paracentral", 18: "parsopercularis",
    19: "parsorbitalis", 20: "parstriangularis", 21: "pericalcarine", 22: "postcentral",
    23: "posteriorcingulate", 24: "precentral", 25: "precuneus", 26: "rostralanteriorcingulate",
    27: "rostralmiddlefrontal", 28: "superiorfrontal", 29: "superiorparietal",
    30: "superiortemporal", 31: "supramarginal", 34: "transversetemporal", 35: "insula",
}


def regions_from_labels(row, prefix):
    """Label-volume dict -> region-volume dict (bilateral composites + hemispheres)."""
    g = lambda l: float(row.get(f"{prefix}{l}", 0.0))
    out = {}
    for n, (l, r) in BILAT.items():
        out[n] = g(l) + g(r)
        out["L_" + n], out["R_" + n] = g(l), g(r)
    for n, l in SINGLE.items():
        out[n] = g(l)
    ctx_l = ctx_r = 0.0
    for k, n in DKT.items():
        out["L_" + n], out["R_" + n] = g(1000 + k), g(2000 + k)
        out[n] = out["L_" + n] + out["R_" + n]
        ctx_l += out["L_" + n]
        ctx_r += out["R_" + n]
    out["cortex"] = ctx_l + ctx_r
    out["subcortical_gm"] = sum(out[n] for n in ("thalamus", "caudate", "putamen", "pallidum",
                                                   "hippocampus", "amygdala", "accumbens", "ventral_dc"))
    out["total_brain"] = out["cortex"] + out["cerebral_wm"] + out["subcortical_gm"] + \
        out["cerebellum_cortex"] + out["cerebellum_wm"] + out["brainstem"]
    out["ventricles_all"] = out["lateral_ventricles"] + out["inf_lat_ventricles"] + \
        out["third_ventricle"] + out["fourth_ventricle"]
    return out


def _synthstrip():
    rows = _read_jsonl(os.path.join(KAG, "icv_results", "icv_*_part*.jsonl"))
    return {r["scan"]: r["icv_synthstrip"] for r in rows}


def _read_jsonl(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += [json.loads(l) for l in open(f)]
    return rows


def load_ixi():
    SS = _synthstrip()
    rows = _read_jsonl(os.path.join(KAG, "ixi_lba_results", "ixi_lba_part*.jsonl"))
    demo = pd.read_csv(os.path.join(ROOT, "data", "ixi", "IXI_demographics.csv"))
    demo = demo.rename(columns={"SEX_ID (1=m, 2=f)": "SEX"})
    demo = demo[demo.AGE.notna()]
    # duplicated IDs: keep only if age and sex agree
    agree = demo.groupby("IXI_ID").agg(na=("AGE", lambda s: s.round(2).nunique()),
                                       ns=("SEX", "nunique"))
    ok = agree[(agree.na == 1) & (agree.ns == 1)].index
    demo = demo[demo.IXI_ID.isin(ok)].drop_duplicates("IXI_ID")
    demo = demo.set_index("IXI_ID")
    out = []
    for r in rows:
        iid = int(r["file"][3:6])
        if iid not in demo.index:
            continue
        site = r["file"].split("-")[1]
        d = {"dataset": "IXI", "subject": f"IXI{iid:03d}", "scan": r["file"],
             "age": float(demo.loc[iid, "AGE"]), "sex": 1 if demo.loc[iid, "SEX"] == 1 else 0,
             "icv": SS.get(r["file"], np.nan), "icv_mask": float(r["mask_vox"]),
             "site": f"IXI-{site}", "wave": 1}
        d.update(regions_from_labels(r, "V"))
        d.update({k: v for k, v in r.items() if k.startswith(("cap_", "raw_", "hm_"))})
        d["gm_over_wm"], d["wm_mean_nu"] = r.get("gm_over_wm"), r.get("wm_mean_nu")
        out.append(d)
    df = pd.DataFrame(out)
    # a few IXI subjects were scanned twice; keep the first scan as the reference
    return df.drop_duplicates("subject", keep="first").reset_index(drop=True)


def load_dlbs():
    SS = _synthstrip()
    rows = _read_jsonl(os.path.join(KAG, "dlbs_fs_results", "dlbs_fs_part*.jsonl"))
    p = pd.read_csv(os.path.join(ROOT, "data", "dlbs", "participants.tsv"), sep="\t", na_values="n/a")
    p["sid"] = p.participant_id.str.replace("sub-", "", regex=False)
    p = p.set_index("sid")
    out = []
    for r in rows:
        s, w, run = r["scan"].split("-")
        if s not in p.index:
            continue
        age = p.loc[s, f"AgeMRI_W{w}"]
        if pd.isna(age):
            continue
        d = {"dataset": "DLBS", "subject": f"DLBS{s}", "scan": r["scan"], "age": float(age),
             "sex": 1 if p.loc[s, "Sex"] == "m" else 0, "icv": SS.get(r["scan"], np.nan),
             "icv_mask": float(r.get("mask_vol", np.nan)),
             "site": "DLBS", "wave": int(w), "run": int(run),
             "mmse": p.loc[s, f"MMSE_W{w}"] if f"MMSE_W{w}" in p.columns else np.nan}
        d.update(regions_from_labels(r, "V"))
        out.append(d)
    return pd.DataFrame(out)


def load_oasis2(nondemented_only=True):
    SS = _synthstrip()
    rows = _read_jsonl(os.path.join(KAG, "oasis_fs_results", "oasis_fs_part*.jsonl"))
    demo = pd.read_csv(os.path.join(ROOT, "data", "oasis_longitudinal.csv"))
    demo = demo.set_index("MRI ID")
    out = []
    for r in rows:
        if r["mpr"] != "mpr-1" or r["session"] not in demo.index:
            continue
        m = demo.loc[r["session"]]
        if nondemented_only and m["Group"] != "Nondemented":
            continue
        d = {"dataset": "OASIS2", "subject": m["Subject ID"], "scan": r["session"],
             "age": float(m["Age"]), "sex": 1 if m["M/F"] == "M" else 0,
             "icv": SS.get(r["session"], np.nan), "etiv": float(m["eTIV"]),
             "site": "OASIS2", "wave": int(m["Visit"]), "cdr": m["CDR"], "mmse": m["MMSE"]}
        d.update(regions_from_labels(r, "L"))
        out.append(d)
    return pd.DataFrame(out)


if __name__ == "__main__":
    for f in (load_ixi, load_dlbs, load_oasis2):
        try:
            d = f()
            print(f.__name__, len(d), "scans", d.subject.nunique(), "people, age",
                  round(d.age.min()), "-", round(d.age.max()))
        except Exception as e:
            print(f.__name__, "not ready:", type(e).__name__, e)
