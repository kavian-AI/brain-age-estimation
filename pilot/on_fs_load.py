# Loads the FastSurfer + SynthStrip results of the big Kaggle queue (kaggle/on_fs_results/,
# OpenNeuro + OpenBHB + SALD + OASIS-1) into one tidy frame, with the known fixes applied:
#   - OASIS-1 RAW is Analyze orient=2 (sagittal) which nibabel reads as axial:
#       rows without o1_orient  -> rotated head, invalid, dropped
#       o1_orient == "ASR-v1"   -> correct volumes but hemispheres swapped (axis2 runs R->L, shown by
#                                  the cohort's L/R asymmetries) -> left/right labels swapped back here
#       o1_orient == "ASL-v2"   -> correct
#   - a scan processed twice keeps the newest valid row
#   - qc_fail marks failed segmentations (hippocampus < 4500 mm3 or labelled volume < 800 cm3)
import glob
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "kaggle", "on_fs_results")
ASEG_PAIRS = [(2, 41), (3, 42), (4, 43), (5, 44), (7, 46), (8, 47), (10, 49), (11, 50), (12, 51), (13, 52),
              (17, 53), (18, 54), (26, 58), (28, 60), (30, 62), (31, 63)]


def _swap_hemispheres(r):
    out = dict(r)
    pairs = ASEG_PAIRS + [(1000 + k, 2000 + k) for k in range(1, 36)]
    for a, b in pairs:
        va, vb = r.get(f"V{a}"), r.get(f"V{b}")
        out.pop(f"V{a}", None)
        out.pop(f"V{b}", None)
        if vb is not None:
            out[f"V{a}"] = vb
        if va is not None:
            out[f"V{b}"] = va
    return out


def load_rows():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "*", "*.jsonl")), key=os.path.getmtime):
        kernel = os.path.basename(os.path.dirname(f))
        for line in open(f):
            r = json.loads(line)
            r["kernel"] = kernel
            if r["scan"].startswith("oasis1__"):
                tag = r.get("o1_orient")
                if tag == "ASR-v1":
                    r = _swap_hemispheres(r)
                elif tag != "ASL-v2":
                    continue
            rows.append(r)
    R = pd.DataFrame(rows).drop_duplicates("scan", keep="last").reset_index(drop=True)
    vols = R.filter(regex=r"^V\d+$")
    R["qc_fail"] = ((R.get("V17", 0).fillna(0) + R.get("V53", 0).fillna(0)) < 4500) | (vols.sum(1) < 8e5)
    R["source"] = R.scan.str.split("__").str[0]
    return R


if __name__ == "__main__":
    R = load_rows()
    print(len(R), "scans;", int(R.qc_fail.sum()), "QC failures")
    print(R.groupby("source").size().to_string())
