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

PART = 4
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["ds007522__sub-HML0946__236", "ds007522/sub-HML0946/ses-01/anat/sub-HML0946_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0952__237", "ds007522/sub-HML0952/ses-01/anat/sub-HML0952_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0963__238", "ds007522/sub-HML0963/ses-01/anat/sub-HML0963_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0964__239", "ds007522/sub-HML0964/ses-01/anat/sub-HML0964_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1031__240", "ds007522/sub-HML1031/ses-01/anat/sub-HML1031_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1040__241", "ds007522/sub-HML1040/ses-02/anat/sub-HML1040_ses-02_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1064__242", "ds007522/sub-HML1064/ses-01/anat/sub-HML1064_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1099__243", "ds007522/sub-HML1099/ses-01/anat/sub-HML1099_ses-01_rec-defaced_T1w.nii.gz"], ["ds003592__sub-126__263", "ds003592/sub-126/ses-1/anat/sub-126_ses-1_T1w.nii.gz"], ["ds003592__sub-14__264", "ds003592/sub-14/ses-1/anat/sub-14_ses-1_T1w.nii.gz"], ["ds003592__sub-193__265", "ds003592/sub-193/ses-1/anat/sub-193_ses-1_T1w.nii.gz"], ["ds003592__sub-286__266", "ds003592/sub-286/ses-1/anat/sub-286_ses-1_T1w.nii.gz"], ["ds003592__sub-293__267", "ds003592/sub-293/ses-1/anat/sub-293_ses-1_T1w.nii.gz"], ["ds003592__sub-66__268", "ds003592/sub-66/ses-1/anat/sub-66_ses-1_T1w.nii.gz"], ["ds005270__sub-025__269", "ds005270/sub-025/anat/sub-025_T1w.nii.gz"], ["ds005270__sub-108__270", "ds005270/sub-108/anat/sub-108_T1w.nii.gz"], ["ds005270__sub-139__271", "ds005270/sub-139/anat/sub-139_T1w.nii.gz"], ["ds004711__sub-140__272", "ds004711/sub-140/anat/sub-140_T1w.nii.gz"], ["ds007831__sub-YSN__273", "ds007831/sub-YSN/anat/sub-YSN_T1w.nii.gz"], ["ds004711__sub-169__274", "ds004711/sub-169/anat/sub-169_T1w.nii.gz"], ["ds007831__sub-VDX__275", "ds007831/sub-VDX/anat/sub-VDX_T1w.nii.gz"], ["ds007831__sub-CPV__276", "ds007831/sub-CPV/anat/sub-CPV_T1w.nii.gz"], ["ds004711__sub-131__277", "ds004711/sub-131/anat/sub-131_T1w.nii.gz"], ["ds004725__sub-22043__278", "ds004725/sub-22043/anat/sub-22043_T1w.nii"], ["ds007831__sub-OEH__279", "ds007831/sub-OEH/anat/sub-OEH_T1w.nii.gz"], ["ds004711__sub-165__280", "ds004711/sub-165/anat/sub-165_T1w.nii.gz"], ["ds007522__sub-HML0230__281", "ds007522/sub-HML0230/ses-01/anat/sub-HML0230_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0336__282", "ds007522/sub-HML0336/ses-01/anat/sub-HML0336_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0366__283", "ds007522/sub-HML0366/ses-01/anat/sub-HML0366_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0398__284", "ds007522/sub-HML0398/ses-01/anat/sub-HML0398_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0471__285", "ds007522/sub-HML0471/ses-01/anat/sub-HML0471_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0493__286", "ds007522/sub-HML0493/ses-01/anat/sub-HML0493_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0522__287", "ds007522/sub-HML0522/ses-01/anat/sub-HML0522_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0534__288", "ds007522/sub-HML0534/ses-01/anat/sub-HML0534_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0603__289", "ds007522/sub-HML0603/ses-01/anat/sub-HML0603_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0622__290", "ds007522/sub-HML0622/ses-01/anat/sub-HML0622_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0740__291", "ds007522/sub-HML0740/ses-01/anat/sub-HML0740_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0875__292", "ds007522/sub-HML0875/ses-01/anat/sub-HML0875_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0894__293", "ds007522/sub-HML0894/ses-01/anat/sub-HML0894_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0915__294", "ds007522/sub-HML0915/ses-02/anat/sub-HML0915_ses-02_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0981__295", "ds007522/sub-HML0981/ses-01/anat/sub-HML0981_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0984__296", "ds007522/sub-HML0984/ses-01/anat/sub-HML0984_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1060__297", "ds007522/sub-HML1060/ses-01/anat/sub-HML1060_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1071__298", "ds007522/sub-HML1071/ses-02/anat/sub-HML1071_ses-02_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1089__299", "ds007522/sub-HML1089/ses-01/anat/sub-HML1089_ses-01_rec-defaced_T1w.nii.gz"], ["ds003592__sub-109__318", "ds003592/sub-109/ses-1/anat/sub-109_ses-1_T1w.nii.gz"], ["ds003592__sub-11__319", "ds003592/sub-11/ses-1/anat/sub-11_ses-1_T1w.nii.gz"], ["ds003592__sub-183__320", "ds003592/sub-183/ses-1/anat/sub-183_ses-1_T1w.nii.gz"], ["ds005270__sub-104__321", "ds005270/sub-104/anat/sub-104_T1w.nii.gz"], ["ds005270__sub-109__322", "ds005270/sub-109/anat/sub-109_T1w.nii.gz"], ["ds004725__sub-22042__323", "ds004725/sub-22042/anat/sub-22042_T1w.nii"], ["ds004725__sub-21026__324", "ds004725/sub-21026/anat/sub-21026_T1w.nii"], ["ds004725__sub-22049__325", "ds004725/sub-22049/anat/sub-22049_T1w.nii"], ["ds004725__sub-21010__326", "ds004725/sub-21010/anat/sub-21010_T1w.nii"], ["ds004711__sub-093__327", "ds004711/sub-093/anat/sub-093_T1w.nii.gz"], ["ds007831__sub-EOV__328", "ds007831/sub-EOV/anat/sub-EOV_T1w.nii.gz"], ["ds004711__sub-047__329", "ds004711/sub-047/anat/sub-047_T1w.nii.gz"], ["ds007522__sub-HML0038__330", "ds007522/sub-HML0038/ses-01/anat/sub-HML0038_ses-01_rec-defaced_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
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
