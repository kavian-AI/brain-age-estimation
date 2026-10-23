#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THE RICH ATLAS: 98 FreeSurfer regions per subject, from TADPOLE.

ADNIMERGE ships six regions. TADPOLE_D1_D2 (the same ADNI subjects) ships the
UCSF FreeSurfer tables behind them: 346 measures per pipeline, in TWO pipelines
run on the same scans - cross-sectional (UCSFFSX) and longitudinal (UCSFFSL).

The data dictionary is distributed only through LONI, so the region names were
RECONSTRUCTED here and then checked. The UCSF ST codes are alphabetical over a
combined cortical+subcortical list, and every right-hemisphere region sits exactly
59 codes after its left twin. Six anchors were verified against the ADNIMERGE
summary columns that live in the same file (r >= 0.9965, median ratio 1.000):
    Hippocampus = ST29SV + ST88SV        Entorhinal = ST24CV + ST83CV
    Fusiform    = ST26CV + ST85CV        MidTemp    = ST40CV + ST99CV
    Ventricles  = ST30SV + ST37SV + ST89SV + ST96SV          ICV = ST10CV
Every other name follows from the alphabetical order and was sanity-checked
against typical volume, left/right correlation and the sign of the age effect.

Excluded on purpose: ICV (a covariate, not a region); insula (ST64/123, present
only for FreeSurfer 5.1 subjects, 47% coverage); whole-hemisphere cortex/WM
(ST19/20/78/79, same 47% problem); CSF, choroid plexus, vessels, optic chiasm,
hypointensities and the fifth ventricle (not tissue, or not reliably segmented).

Sign convention for averaging: tissue counts as worse when SMALLER, fluid spaces
(ventricles) count as worse when LARGER. That is an anatomical prior decided
before looking at any diagnosis, so the uniform average stays unsupervised.
"""

import os
import re

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root (local or Kaggle)
TADPOLE = os.path.join(BASE, "data", "tadpole", "TADPOLE_D1_D2.csv")
PIPE = {"X": "UCSFFSX_11_02_15_UCSFFSX51_08_01_16",       # cross-sectional
        "L": "UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16"}     # longitudinal

CORTICAL = {13: "bankssts", 14: "caudalanteriorcingulate", 15: "caudalmiddlefrontal",
            23: "cuneus", 24: "entorhinal", 25: "frontalpole", 26: "fusiform",
            31: "inferiorparietal", 32: "inferiortemporal", 34: "isthmuscingulate",
            35: "lateraloccipital", 36: "lateralorbitofrontal", 38: "lingual",
            39: "medialorbitofrontal", 40: "middletemporal", 43: "paracentral",
            44: "parahippocampal", 45: "parsopercularis", 46: "parsorbitalis",
            47: "parstriangularis", 48: "pericalcarine", 49: "postcentral",
            50: "posteriorcingulate", 51: "precentral", 52: "precuneus",
            54: "rostralanteriorcingulate", 55: "rostralmiddlefrontal",
            56: "superiorfrontal", 57: "superiorparietal", 58: "superiortemporal",
            59: "supramarginal", 60: "temporalpole", 62: "transversetemporal"}
SUBCORT = {11: "accumbens", 12: "amygdala", 16: "caudate", 17: "cerebellumcortex",
           18: "cerebellumwm", 29: "hippocampus", 42: "pallidum", 53: "putamen",
           61: "thalamus", 65: "ventraldc"}
VENTR = {30: "inflatvent", 37: "latvent"}
MIDLINE = {1: ("brainstem", -1), 2: ("cc_anterior", -1), 3: ("cc_central", -1),
           4: ("cc_midanterior", -1), 5: ("cc_midposterior", -1),
           6: ("cc_posterior", -1), 9: ("fourthventricle", +1),
           127: ("thirdventricle", +1)}

# Region CLASS, used only to read the age model's weights afterwards.
MTL = {"entorhinal", "parahippocampal", "hippocampus", "amygdala", "fusiform",
       "temporalpole", "inferiortemporal", "middletemporal"}


def region_table(kind_cort="CV"):
    """(code, measure kind, name, sign, class) for all 98 regions."""
    rows = []
    for n, nm in CORTICAL.items():
        for side, off in (("L", 0), ("R", 59)):
            cls = "mtl" if nm in MTL else "cortex"
            rows.append((n + off, kind_cort, f"{side}_{nm}", -1, cls))
    for n, nm in SUBCORT.items():
        for side, off in (("L", 0), ("R", 59)):
            cls = "mtl" if nm in MTL else (
                "cerebellum" if nm.startswith("cerebellum") else "subcortical")
            rows.append((n + off, "SV", f"{side}_{nm}", -1, cls))
    for n, nm in VENTR.items():
        for side, off in (("L", 0), ("R", 59)):
            rows.append((n + off, "SV", f"{side}_{nm}", +1, "ventricle"))
    for n, (nm, s) in MIDLINE.items():
        cls = "ventricle" if s > 0 else ("callosum" if nm.startswith("cc") else "brainstem")
        rows.append((n, "SV", nm, s, cls))
    return pd.DataFrame(rows, columns=["code", "kind", "name", "sign", "cls"])


def load_regions(pipe="X", visit="bl", kind_cort="CV"):
    """One row per PTID at the requested visit: 98 regional measures + ICV +
    the FreeSurfer version, from the chosen pipeline. Numeric, NaN for missing."""
    T = region_table(kind_cort)
    suf = PIPE[pipe]
    need = [f"ST{c}{k}_{suf}" for c, k in zip(T.code, T.kind)] + [f"ST10CV_{PIPE['X']}"]
    d = pd.read_csv(TADPOLE, low_memory=False,
                    usecols=["PTID", "VISCODE", "FSVERSION"] + list(dict.fromkeys(need)))
    d = d[d.VISCODE == visit].drop_duplicates("PTID").reset_index(drop=True)
    out = pd.DataFrame({"PTID": d.PTID.values,
                        "FSVERSION_T": d.FSVERSION.astype(str).values})
    for (c, k, nm) in zip(T.code, T.kind, T.name):
        out[nm] = pd.to_numeric(d[f"ST{c}{k}_{suf}"], errors="coerce").values
    out["ICV_T"] = pd.to_numeric(d[f"ST10CV_{PIPE['X']}"], errors="coerce").values
    return out, T
