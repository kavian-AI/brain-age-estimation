# OpenNeuro healthy adults through FastSurfer + SynthStrip (GPU), streamed from S3.
#
# 12 CC0 OpenNeuro datasets (Precision Aging Network, NIMH Healthy Volunteers, MiND,
# Neurocognitive aging, GRAND, adult-lifespan sets, ...): 2,618 healthy people, 1,356
# aged 60+, plus 667 extra scans (second sessions / same-session runs) for repeat noise.
# Per scan: FastSurfer v2.4.2 aseg+DKT volumes (same program as IXI / DLBS / OASIS-2),
# SynthStrip intracranial volume (head size), and the scanner from the BIDS sidecar
# (manufacturer, model, field strength) so each scanner can be its own "site".
# Chunks of 12 scans: the next chunk downloads while the GPU works; rows are appended
# after every chunk, so a timeout loses at most one chunk.
import concurrent.futures as cf
import glob, gzip, json, os, shutil, subprocess, threading, time, urllib.request

PART = 3
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["ds007831__sub-OZS__162", "ds007831/sub-OZS/anat/sub-OZS_T1w.nii.gz"], ["ds007522__sub-HML0130__163", "ds007522/sub-HML0130/ses-01/anat/sub-HML0130_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0182__164", "ds007522/sub-HML0182/ses-01/anat/sub-HML0182_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0227__165", "ds007522/sub-HML0227/ses-01/anat/sub-HML0227_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0274__166", "ds007522/sub-HML0274/ses-01/anat/sub-HML0274_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0341__167", "ds007522/sub-HML0341/ses-01/anat/sub-HML0341_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0394__168", "ds007522/sub-HML0394/ses-01/anat/sub-HML0394_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0426__169", "ds007522/sub-HML0426/ses-01/anat/sub-HML0426_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0579__170", "ds007522/sub-HML0579/ses-01/anat/sub-HML0579_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0720__171", "ds007522/sub-HML0720/ses-01/anat/sub-HML0720_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0799__172", "ds007522/sub-HML0799/ses-01/anat/sub-HML0799_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0844__173", "ds007522/sub-HML0844/ses-01/anat/sub-HML0844_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0846__174", "ds007522/sub-HML0846/ses-01/anat/sub-HML0846_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0855__175", "ds007522/sub-HML0855/ses-01/anat/sub-HML0855_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0957__176", "ds007522/sub-HML0957/ses-01/anat/sub-HML0957_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0968__177", "ds007522/sub-HML0968/ses-01/anat/sub-HML0968_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0983__178", "ds007522/sub-HML0983/ses-01/anat/sub-HML0983_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1052__179", "ds007522/sub-HML1052/ses-01/anat/sub-HML1052_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1173__180", "ds007522/sub-HML1173/ses-01/anat/sub-HML1173_ses-01_rec-defaced_T1w.nii.gz"], ["ds003592__sub-104__197", "ds003592/sub-104/ses-1/anat/sub-104_ses-1_T1w.nii.gz"], ["ds003592__sub-203__198", "ds003592/sub-203/ses-1/anat/sub-203_ses-1_T1w.nii.gz"], ["ds003592__sub-299__199", "ds003592/sub-299/ses-1/anat/sub-299_ses-1_T1w.nii.gz"], ["ds005270__sub-041__200", "ds005270/sub-041/anat/sub-041_T1w.nii.gz"], ["ds005405__sub-062__201", "ds005405/sub-062/ses-bhb/anat/sub-062_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-103__202", "ds005405/sub-103/ses-bhb/anat/sub-103_ses-bhb_run-1_T1w.nii"], ["ds004711__sub-176__203", "ds004711/sub-176/anat/sub-176_T1w.nii.gz"], ["ds004711__sub-161__204", "ds004711/sub-161/anat/sub-161_T1w.nii.gz"], ["ds004725__sub-21007__205", "ds004725/sub-21007/anat/sub-21007_T1w.nii"], ["ds004711__sub-175__206", "ds004711/sub-175/anat/sub-175_T1w.nii.gz"], ["ds004725__sub-21009__207", "ds004725/sub-21009/anat/sub-21009_T1w.nii"], ["ds004725__sub-22044__208", "ds004725/sub-22044/anat/sub-22044_T1w.nii"], ["ds004725__sub-21004__209", "ds004725/sub-21004/anat/sub-21004_T1w.nii"], ["ds004711__sub-177__210", "ds004711/sub-177/anat/sub-177_T1w.nii.gz"], ["ds007522__sub-HML0037__211", "ds007522/sub-HML0037/ses-01/anat/sub-HML0037_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0082__212", "ds007522/sub-HML0082/ses-01/anat/sub-HML0082_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0092__213", "ds007522/sub-HML0092/ses-01/anat/sub-HML0092_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0113__214", "ds007522/sub-HML0113/ses-01/anat/sub-HML0113_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0148__215", "ds007522/sub-HML0148/ses-01/anat/sub-HML0148_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0198__216", "ds007522/sub-HML0198/ses-01/anat/sub-HML0198_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0199__217", "ds007522/sub-HML0199/ses-01/anat/sub-HML0199_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0203__218", "ds007522/sub-HML0203/ses-01/anat/sub-HML0203_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0263__219", "ds007522/sub-HML0263/ses-01/anat/sub-HML0263_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0271__220", "ds007522/sub-HML0271/ses-01/anat/sub-HML0271_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0324__221", "ds007522/sub-HML0324/ses-01/anat/sub-HML0324_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0337__222", "ds007522/sub-HML0337/ses-01/anat/sub-HML0337_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0385__223", "ds007522/sub-HML0385/ses-01/anat/sub-HML0385_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0387__224", "ds007522/sub-HML0387/ses-01/anat/sub-HML0387_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0425__225", "ds007522/sub-HML0425/ses-01/anat/sub-HML0425_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0470__226", "ds007522/sub-HML0470/ses-01/anat/sub-HML0470_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0475__227", "ds007522/sub-HML0475/ses-01/anat/sub-HML0475_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0489__228", "ds007522/sub-HML0489/ses-01/anat/sub-HML0489_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0496__229", "ds007522/sub-HML0496/ses-01/anat/sub-HML0496_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0616__230", "ds007522/sub-HML0616/ses-01/anat/sub-HML0616_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0716__231", "ds007522/sub-HML0716/ses-01/anat/sub-HML0716_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0724__232", "ds007522/sub-HML0724/ses-01/anat/sub-HML0724_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0735__233", "ds007522/sub-HML0735/ses-01/anat/sub-HML0735_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0887__234", "ds007522/sub-HML0887/ses-01/anat/sub-HML0887_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0909__235", "ds007522/sub-HML0909/ses-01/anat/sub-HML0909_ses-01_rec-defaced_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
CHUNK = 12


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)


print("PART", PART, "scans", len(JOBS), flush=True)
r = sh("git clone --depth 1 --branch v2.4.2 https://github.com/Deep-MI/FastSurfer.git /tmp/FastSurfer")
print("fastsurfer clone rc", r.returncode, flush=True)
r = sh("pip install -q yacs simpleitk nibabel scikit-image h5py lapy pandas surfa")
print("pip rc", r.returncode, r.stderr[-200:], flush=True)
if sh("python -c 'import surfa'").returncode != 0:
    print("surfa from GitHub rc", sh("pip install -q git+https://github.com/freesurfer/surfa.git").returncode, flush=True)
os.makedirs("/tmp/fs/models", exist_ok=True)
urllib.request.urlretrieve("https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthstrip/mri_synthstrip",
                           "/tmp/fs/mri_synthstrip")
urllib.request.urlretrieve("https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt",
                           "/tmp/fs/models/synthstrip.1.pt")
src = open("/tmp/fs/mri_synthstrip").read().replace(
    "torch.load(modelfile, map_location=device)", "torch.load(modelfile, map_location=device, weights_only=False)")
open("/tmp/fs/mri_synthstrip", "w").write(src)
os.environ["FREESURFER_HOME"] = "/tmp/fs"

import nibabel as nib
import numpy as np

done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["scan"] for l in open(OUT)}
todo = [j for j in JOBS if j[0] not in done]
chunks = [todo[k:k + CHUNK] for k in range(0, len(todo), CHUNK)]


def fetch_one(job, d):
    sid, key = job
    for attempt in range(4):
        try:
            p = f"{d}/{sid}.nii.gz"
            if key.endswith(".nii.gz"):
                urllib.request.urlretrieve(S3 + key, p)
            else:
                urllib.request.urlretrieve(S3 + key, f"{d}/{sid}.nii")
                with open(f"{d}/{sid}.nii", "rb") as fi, gzip.open(p, "wb", compresslevel=1) as fo:
                    shutil.copyfileobj(fi, fo)
                os.remove(f"{d}/{sid}.nii")
            meta = {}
            ds = key.split("/")[0]
            for side in (key.rsplit(".nii", 1)[0] + ".json", f"{ds}/T1w.json"):   # BIDS inheritance
                try:
                    meta = json.loads(urllib.request.urlopen(S3 + side, timeout=60).read())
                    break
                except Exception:
                    pass
            keep = {k: meta.get(k) for k in ("Manufacturer", "ManufacturersModelName", "MagneticFieldStrength",
                                             "SoftwareVersions", "InstitutionName")}
            json.dump(keep, open(f"{d}/{sid}.meta.json", "w"))
            return True
        except Exception as e:
            print(f"  download retry {sid}: {type(e).__name__}", flush=True)
            time.sleep(10)
    return False


def fetch_chunk(ch, d):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    with cf.ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: fetch_one(j, d), ch))


n_ok = len(done)
dirs = ["/tmp/dlA", "/tmp/dlB"]
pre = None
if chunks:
    fetch_chunk(chunks[0], dirs[0])
for c, ch in enumerate(chunks):
    if time.time() - T0 > BUDGET_S:
        print("time budget reached, stopping", flush=True)
        break
    d = dirs[c % 2]
    if pre is not None:
        pre.join()
    if c + 1 < len(chunks):
        pre = threading.Thread(target=fetch_chunk, args=(chunks[c + 1], dirs[(c + 1) % 2]))
        pre.start()
    t1 = time.time()
    shutil.rmtree("/tmp/out", ignore_errors=True)
    r = sh("cd /tmp/FastSurfer && PYTHONPATH=/tmp/FastSurfer python FastSurferCNN/run_prediction.py "
           f"--in_dir {d} --tag '*.nii.gz' --remove_suffix .nii.gz --sd /tmp/out "
           "--device cpu --viewagg_device cpu --threads 4 --vox_size 1 --batch_size 8")
    rows = []
    for sid, key in ch:
        segp = f"/tmp/out/{sid}/mri/aparc.DKTatlas+aseg.deep.mgz"
        if not os.path.exists(segp):
            print(f"  FAIL seg {sid}", flush=True)
            continue
        try:
            s = nib.load(segp)
            a = np.asanyarray(s.dataobj).astype(np.int32)
            vox = float(np.prod(s.header.get_zooms()[:3]))
            labs, cnt = np.unique(a[a > 0], return_counts=True)
            im = nib.load(f"{d}/{sid}.nii.gz")
            row = {"scan": sid, "key": key, "zooms_in": [float(z) for z in im.header.get_zooms()[:3]],
                   "shape_in": [int(x) for x in im.shape[:3]]}
            row.update(json.load(open(f"{d}/{sid}.meta.json")) if os.path.exists(f"{d}/{sid}.meta.json") else {})
            mp = f"/tmp/out/{sid}/mri/mask.mgz"
            if os.path.exists(mp):
                row["mask_vol"] = float((np.asanyarray(nib.load(mp).dataobj) > 0).sum()) * vox
            rs = sh(f"python /tmp/fs/mri_synthstrip -i {d}/{sid}.nii.gz -m /tmp/ssmask.nii.gz")
            if os.path.exists("/tmp/ssmask.nii.gz"):
                mm = nib.load("/tmp/ssmask.nii.gz")
                row["icv_synthstrip"] = float((np.asanyarray(mm.dataobj) > 0).sum()) * float(np.prod(mm.header.get_zooms()[:3]))
                os.remove("/tmp/ssmask.nii.gz")
            else:
                print(f"  synthstrip fail {sid}: {rs.stderr[-200:]}", flush=True)
            row.update({f"V{int(l)}": float(n) * vox for l, n in zip(labs, cnt)})
            rows.append(row)
        except Exception as e:
            print(f"  ERROR {sid}: {type(e).__name__} {e}", flush=True)
    with open(OUT, "a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + chr(10))
    n_ok += len(rows)
    dt = time.time() - t1
    print(f"chunk {c+1}/{len(chunks)}: {len(rows)}/{len(ch)} ok, {dt:.0f}s ({dt/max(len(ch),1):.1f}s/scan), "
          f"total {n_ok}/{len(JOBS)}, elapsed {(time.time()-T0)/3600:.2f} h", flush=True)
    if len(rows) == 0:
        print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
        if c == 0:
            break
print(f"finished: {n_ok} rows, {(time.time()-T0)/3600:.2f} h", flush=True)
