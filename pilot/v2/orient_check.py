"""Pre-flight for hemisphere models: (1) does FOV QC also blank the left/right columns?
(2) per-dataset left/right orientation check from healthy population asymmetries."""
import os, sys, numpy as np, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "pilot"))
import p3_pooled as P
import p5_full_pool as F
F.OUT = os.path.join(HERE, "_tmp"); os.makedirs(F.OUT, exist_ok=True)
allp, *_ = F.full_pool()

PAIRS = {"hippocampus": "hippocampus", "amygdala": "amygdala", "lateral ventricles": "lateral_ventricles",
         "inf. lat. ventricles": "inf_lat_ventricles", "thalamus": "thalamus", "putamen": "putamen",
         "cerebellum cortex": "cerebellum_cortex", "cerebral WM": "cerebral_wm", "entorhinal": "entorhinal",
         "precuneus": "precuneus", "superior frontal": "superiorfrontal", "lateral occipital": "lateraloccipital",
         "precentral": "precentral"}
print("=== (1) Bilateral value missing but left/right present? (would leak QC-failed scans into hemisphere models)")
for reg, h in PAIRS.items():
    miss = allp[reg].isna() | (allp[reg] <= 0)
    leak = miss & allp["L_" + h].notna() & (allp["L_" + h] > 0)
    s = allp["L_" + h] + allp["R_" + h]
    ok = ~miss & s.notna()
    rel = (np.abs(s[ok] - allp.loc[ok, reg]) / allp.loc[ok, reg]).max()
    print(f"  {reg:<20} bilateral missing {int(miss.sum()):>4} | of those with L/R present {int(leak.sum()):>4} | max |L+R - bilateral| / bilateral {rel:.1e}")

print("\n=== (2) Orientation: % of healthy people with RIGHT > LEFT, per dataset (should agree across datasets)")
chk = ["hippocampus", "thalamus", "lateral_ventricles", "putamen", "cerebral_wm"]
rows = []
for ds, g in allp.groupby("dataset"):
    row = {"dataset": ds, "n": len(g)}
    for h in chk:
        a = g[(g["L_" + h] > 0) & (g["R_" + h] > 0)]
        row[h] = 100 * (a["R_" + h] > a["L_" + h]).mean()
    rows.append(row)
T = pd.DataFrame(rows).set_index("dataset")
med = T[chk].median()
T["max |dev| from median"] = (T[chk] - med).abs().max(axis=1)
T["flag"] = np.where(T["max |dev| from median"] > 25, "<-- CHECK", "")
print(T.round(0).to_string())
print("\nmedian across datasets:", med.round(0).to_dict())
