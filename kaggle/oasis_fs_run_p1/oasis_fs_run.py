# OASIS-2 through FastSurfer: the external cohort.
#
# Every claim in this project so far rests on ADNI. OASIS-2 is a different study,
# different scanner, different site and different recruitment: 150 people aged
# 60-96, followed longitudinally with the Clinical Dementia Rating, 373 imaging
# sessions. Its raw T1 scans are segmented here with FastSurfer (deep-learning
# FreeSurfer, DKT atlas + aseg) - a DIFFERENT segmentation program from the
# FreeSurfer tables ADNI ships, which makes this a software shift as well.
#
# Two jobs per kernel:
#   mpr-1 of EVERY session          -> diagnosis, prognosis, longitudinal change
#   mpr-2 of every FIRST session    -> a same-session repeat scan, i.e. a free
#                                      test-retest measurement of every summary
#
# The work is split by subject-number parity across two kernels (PART 0 and 1).
# Results are appended to CSV after EVERY scan so a timeout loses nothing.
import glob, json, os, re, shutil, subprocess, sys, time

PART = int(open("/kaggle/working/.part").read()) if os.path.exists("/kaggle/working/.part") else 1
T0 = time.time()
BUDGET_S = 11.0 * 3600          # stop launching new scans after 11 hours
OUT = f"/kaggle/working/oasis_fs_part{PART}.jsonl"


def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    return r


print("PART", PART, flush=True)
r = sh("git clone --depth 1 --branch v2.4.2 https://github.com/Deep-MI/FastSurfer.git /tmp/FastSurfer "
       "|| git clone --depth 1 https://github.com/Deep-MI/FastSurfer.git /tmp/FastSurfer")
print("clone rc", r.returncode, flush=True)
r = sh("pip install -q yacs simpleitk nibabel scikit-image h5py lapy pandas")
print("pip rc", r.returncode, flush=True)
os.environ["PYTHONPATH"] = "/tmp/FastSurfer"

import nibabel as nib
import numpy as np
import pandas as pd

hdrs = sorted(glob.glob("/kaggle/input/**/OAS2_*_MR*/RAW/mpr-*.nifti.hdr", recursive=True))
by = {}
for h in hdrs:
    m = re.search(r"(OAS2_(\d{4})_MR(\d+))/RAW/(mpr-\d)\.nifti\.hdr$", h)
    if m:
        by.setdefault((m.group(1), m.group(4)), h)
jobs = []
for (sess, mpr), h in sorted(by.items()):
    subj = int(sess.split("_")[1])
    if subj % 2 != PART:
        continue
    visit = int(sess.split("MR")[1])
    if mpr == "mpr-1" or (mpr == "mpr-2" and visit == 1):
        jobs.append((sess, mpr, h))
print(f"sessions on disk {len(set(s for s,_ in by))}, jobs for this part {len(jobs)}", flush=True)

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        d = json.loads(line)
        done.add((d["session"], d["mpr"]))
rows_written = len(done)
for i, (sess, mpr, h) in enumerate(jobs):
    if (sess, mpr) in done:
        continue
    if time.time() - T0 > BUDGET_S:
        print("time budget reached, stopping", flush=True)
        break
    t1 = time.time()
    try:
        im = nib.load(h)
        arr = np.asanyarray(im.dataobj).astype(np.float32).squeeze()
        os.makedirs("/tmp/in", exist_ok=True)
        nib.save(nib.Nifti1Image(arr, im.affine), "/tmp/in/t1.nii.gz")
        shutil.rmtree("/tmp/out", ignore_errors=True)
        r = sh("cd /tmp/FastSurfer && python FastSurferCNN/run_prediction.py "
               "--t1 /tmp/in/t1.nii.gz --sid s --sd /tmp/out --device cuda "
               "--viewagg_device cuda --threads 2")
        seg = glob.glob("/tmp/out/s/mri/aparc.DKTatlas+aseg.deep.mgz")
        if not seg:
            print(f"  FAIL {sess} {mpr}: {r.stderr[-400:]}", flush=True)
            continue
        s = nib.load(seg[0])
        a = np.asanyarray(s.dataobj).astype(np.int32)
        vox = float(np.prod(s.header.get_zooms()[:3]))
        labs, cnt = np.unique(a[a > 0], return_counts=True)
        row = {"session": sess, "mpr": mpr, "seconds": round(time.time() - t1, 1)}
        row.update({f"L{int(l)}": float(c) * vox for l, c in zip(labs, cnt)})
        with open(OUT, "a") as fh:          # one JSON object per scan: robust to
            fh.write(json.dumps(row) + chr(10))  # label sets that differ between scans
        rows_written += 1
        if rows_written % 10 == 1:
            print(f"  [{rows_written}/{len(jobs)}] {sess} {mpr} {row['seconds']}s  "
                  f"hippo L/R {row.get('L17',0):.0f}/{row.get('L53',0):.0f}  "
                  f"elapsed {(time.time()-T0)/3600:.2f} h", flush=True)
    except Exception as e:
        print(f"  ERROR {sess} {mpr}: {type(e).__name__} {e}", flush=True)
print(f"finished: {rows_written} rows, {(time.time()-T0)/3600:.2f} h", flush=True)
