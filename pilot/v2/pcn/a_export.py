"""Step A (main env): export the 358 p6 people + the OASIS adaptation pools with FreeSurfer
aseg names, so PCN braincharts can score them with the SAME folds and the SAME controls.

Uses p6_disease_test.build() unchanged (same QC, same OASIS-1 hemisphere fix, same ICV).
Run from the repo root:  python pilot/v2/pcn/a_export.py   (writes people.csv, adapt_pool.csv here)
"""
import os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "pilot"))

import p3_pooled as P
import p6_disease_test as S6
S6.F.OUT = os.path.join(OUT, "_tmp"); os.makedirs(S6.F.OUT, exist_ok=True)   # QC side files, never p5_out

# PCN aseg name -> our hemisphere column
PCN2OURS = {}
for side, pre in (("Left", "L_"), ("Right", "R_")):
    PCN2OURS.update({
        f"{side}-Hippocampus": pre + "hippocampus", f"{side}-Amygdala": pre + "amygdala",
        f"{side}-Lateral-Ventricle": pre + "lateral_ventricles", f"{side}-Inf-Lat-Vent": pre + "inf_lat_ventricles",
        f"{side}-Thalamus-Proper": pre + "thalamus", f"{side}-Putamen": pre + "putamen",
        f"{side}-Cerebellum-Cortex": pre + "cerebellum_cortex", f"{side}-Caudate": pre + "caudate",
        f"{side}-Pallidum": pre + "pallidum", f"{side}-Accumbens-area": pre + "accumbens",
        f"{side}-VentralDC": pre + "ventral_dc", f"{side}-Cerebellum-White-Matter": pre + "cerebellum_wm"})
PCN2OURS.update({"3rd-Ventricle": "third_ventricle", "4th-Ventricle": "fourth_ventricle", "Brain-Stem": "brainstem"})

allp, o2, people, licv_mean = S6.build()

# folds: take them from the published scores so both atlases use identical splits
sc = pd.read_csv(os.path.join(REPO, "pilot", "p6_out", "scores.csv"))
key = ["subject", "group"]
assert not sc.duplicated(key).any() and not people.duplicated(key).any()
m = people.merge(sc[key + ["fold"]], on=key, how="inner", suffixes=("_rebuilt", ""))
agree = float((m.fold_rebuilt == m.fold).mean())
print(f"people rebuilt: {len(people)}, matched to published scores: {len(m)} / {len(sc)}; "
      f"fold agreement rebuilt vs published: {agree:.3f}")
assert len(m) == len(sc), "could not match every published person"

missing = [v for v in PCN2OURS.values() if v not in m.columns]
assert not missing, f"hemisphere columns missing: {missing}"

def tidy(d):
    t = pd.DataFrame({"subject": d.subject.values, "age": d.age.astype(float).values,
                      "sex": d.sex.astype(int).values, "icv": d.icv.astype(float).values})
    for pcn, ours in PCN2OURS.items():
        t[pcn] = d[ours].astype(float).values
    # OASIS-2 patients come through a rename (third_ventricle -> "third ventricle") in build()
    if "third ventricle" in d.columns:
        alt = d["third ventricle"].astype(float).values
        both = ~np.isnan(t["3rd-Ventricle"].values) & ~np.isnan(alt)
        if both.any():
            assert np.allclose(t["3rd-Ventricle"].values[both], alt[both]), "3rd ventricle columns disagree"
        t["3rd-Ventricle"] = np.where(np.isnan(t["3rd-Ventricle"].values), alt, t["3rd-Ventricle"].values)
    return t

ppl = tidy(m)
for c in ("cohort", "group", "is_patient", "fold"):
    ppl[c] = m[c].values
ppl.to_csv(os.path.join(OUT, "people.csv"), index=False)

# adaptation pools = every healthy pool member of that OASIS site (what our atlas saw), incl. <55
pool = []
for ds, coh in (("oasis1", "OASIS-1"), ("OASIS2", "OASIS-2")):
    d = allp[allp.dataset == ds]
    t = tidy(d); t["cohort"] = coh
    pool.append(t)
pool = pd.concat(pool, ignore_index=True)
pool.to_csv(os.path.join(OUT, "adapt_pool.csv"), index=False)

print(ppl.groupby(["cohort", "group"]).size().to_string())
print("adaptation pools:", pool.groupby("cohort").size().to_dict(),
      "| OASIS-1 pool ages", f"{pool[pool.cohort=='OASIS-1'].age.min():.0f}-{pool[pool.cohort=='OASIS-1'].age.max():.0f}")
nan = ppl[list(PCN2OURS)].isna().sum(); print("NaN per phenotype (people):", nan[nan > 0].to_dict() or "none")
print("scored controls all inside their adaptation pool:",
      bool(ppl[~ppl.is_patient].subject.isin(pool.subject).all()))
