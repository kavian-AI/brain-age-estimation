# Intracranial volume (ICV) with SynthStrip (Hoopes et al., NeuroImage 2022).
#
# The FastSurfer brain mask we used as a head-size proxy shrinks with age
# (IXI -0.18 %/yr, DLBS -0.08 %/yr), so "normalising" by it removes part of the
# aging signal. SynthStrip's default mask keeps the CSF around the brain, i.e. it
# follows the inner skull, which does not shrink with brain atrophy. Its volume is
# our ICV. OASIS-2 is run too, so it can be checked against OASIS's published eTIV.
#
# CPU kernel (GPU quota is spent). One JSON row per scan, appended as it goes.
import glob, json, os, re, subprocess, time, urllib.request

DATASET = "oasis"      # ixi | dlbs | oasis
PART, NPARTS = 0, 1
T0 = time.time()
BUDGET_S = 11.3 * 3600
OUT = f"/kaggle/working/icv_{DATASET}_part{PART}.jsonl"
DLBS_IDS = "".split()


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)


r = sh("pip install -q surfa nibabel")
print("pip surfa (PyPI) rc", r.returncode, r.stderr[-300:], flush=True)
if sh("python -c 'import surfa'").returncode != 0:
    r = sh("pip install -q git+https://github.com/freesurfer/surfa.git")
    print("pip surfa (GitHub) rc", r.returncode, r.stderr[-300:], flush=True)
os.makedirs("/tmp/fs/models", exist_ok=True)
urllib.request.urlretrieve("https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthstrip/mri_synthstrip",
                           "/tmp/fs/mri_synthstrip")
urllib.request.urlretrieve("https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt",
                           "/tmp/fs/models/synthstrip.1.pt")
# torch >= 2.6 defaults to weights_only=True; the checkpoint holds plain python objects
src = open("/tmp/fs/mri_synthstrip").read().replace(
    "torch.load(modelfile, map_location=device)", "torch.load(modelfile, map_location=device, weights_only=False)")
open("/tmp/fs/mri_synthstrip", "w").write(src)
os.environ["FREESURFER_HOME"] = "/tmp/fs"
print("model bytes", os.path.getsize("/tmp/fs/models/synthstrip.1.pt"), flush=True)

import nibabel as nib
import numpy as np

# ---------------------------------------------------------------- job list
if DATASET == "ixi":
    files = sorted(glob.glob("/kaggle/input/**/IXI*-T1.nii*", recursive=True))
    assert len(files) > 500, len(files)
    jobs = [(os.path.basename(f).split(".")[0], f) for f in files]
elif DATASET == "oasis":
    hdrs = sorted(glob.glob("/kaggle/input/**/OAS2_*_MR*/RAW/mpr-1.nifti.hdr", recursive=True))
    assert len(hdrs) > 300, len(hdrs)
    jobs = [(re.search(r"(OAS2_\d{4}_MR\d+)", h).group(1), h) for h in hdrs]
else:
    S3 = "https://s3.amazonaws.com/openneuro.org/ds004856"
    jobs = []
    for i in DLBS_IDS:
        s, w, rr = i.split("-")
        jobs.append((i, f"{S3}/sub-{s}/ses-wave{w}/anat/sub-{s}_ses-wave{w}_acq-MPRAGE_run-{rr}_T1w.nii.gz"))
jobs = jobs[PART::NPARTS]
print(DATASET, "part", PART, "jobs", len(jobs), flush=True)

done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["scan"] for l in open(OUT)}
n = len(done)
fails = 0
for k, (sid, src) in enumerate(jobs):
    if sid in done:
        continue
    if time.time() - T0 > BUDGET_S:
        print("time budget reached", flush=True)
        break
    t1 = time.time()
    try:
        if src.startswith("http"):
            urllib.request.urlretrieve(src, "/tmp/in.nii.gz")
            path = "/tmp/in.nii.gz"
        else:
            im = nib.load(src)
            nib.save(nib.Nifti1Image(np.asanyarray(im.dataobj).astype(np.float32).squeeze(), im.affine),
                     "/tmp/in.nii.gz")
            path = "/tmp/in.nii.gz"
        r = sh(f"python /tmp/fs/mri_synthstrip -i {path} -m /tmp/mask.nii.gz -t 4")
        if not os.path.exists("/tmp/mask.nii.gz"):
            print(f"  FAIL {sid}: {r.stdout[-300:]} {r.stderr[-500:]}", flush=True)
            fails += 1
            if n == 0 and fails >= 3:
                raise SystemExit("first scans all failed - stopping")
            continue
        m = nib.load("/tmp/mask.nii.gz")
        vox = float(np.prod(m.header.get_zooms()[:3]))
        icv = float((np.asanyarray(m.dataobj) > 0).sum()) * vox
        os.remove("/tmp/mask.nii.gz")
        with open(OUT, "a") as fh:
            fh.write(json.dumps({"scan": sid, "icv_synthstrip": icv, "seconds": round(time.time() - t1, 1)}) + chr(10))
        n += 1
        if n % 25 == 1:
            print(f"  [{n}/{len(jobs)}] {sid} ICV {icv/1000:.0f} cm3 {time.time()-t1:.0f}s elapsed {(time.time()-T0)/3600:.2f} h", flush=True)
    except Exception as e:
        print(f"  ERROR {sid}: {type(e).__name__} {e}", flush=True)
print(f"finished {n} rows {(time.time()-T0)/3600:.2f} h", flush=True)
