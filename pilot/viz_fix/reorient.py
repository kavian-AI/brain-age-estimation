# The QC plots were drawn from FastSurfer's conformed (LIA) volume with a plain rot90, which shows the
# sagittal view with the face up and the coronal view upside down. This rotates those panels in the
# PNGs to standard radiological display (sagittal: top = superior, face left; coronal: top = superior).
import glob, os, sys
import numpy as np
from PIL import Image

def panels(a):
    nonwhite = (a < 245).any(axis=2)
    H, W = nonwhite.shape
    cols = nonwhite[60:H - 40].sum(0) > 150
    runs, s = [], None
    for x, v in enumerate(list(cols) + [False]):
        if v and s is None: s = x
        if not v and s is not None: runs.append((s, x)); s = None
    x0, x1 = runs[0]                      # all panels are equal squares in one row: take rows from the first
    ys = np.where(nonwhite[:, x0:x1].mean(1) > 0.9)[0]
    return [(a_, b_, ys.min(), ys.max() + 1) for a_, b_ in runs]

for src in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(src, "**", "*.png"), recursive=True)):
        im = Image.open(f).convert("RGB"); a = np.asarray(im).copy()
        P = panels(a)
        sag = P[:-2]          # last two panels are coronal, axial; the rest are sagittal views
        cor = P[-2]
        for x0, x1, y0, y1 in sag:
            crop = Image.fromarray(a[y0:y1, x0:x1])
            r = crop.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT).resize((x1 - x0, y1 - y0))
            a[y0:y1, x0:x1] = np.asarray(r)
        x0, x1, y0, y1 = cor
        a[y0:y1, x0:x1] = a[y0:y1, x0:x1][::-1]
        tag = "b" if "viz_check2" in f.replace(os.sep, "/") else "a"
        dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), tag + "_" + os.path.basename(f))
        Image.fromarray(a).save(dst)
        print(dst, len(P), "panels")
