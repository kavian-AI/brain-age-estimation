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

PART = 5
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["ds007522__sub-HML0074__331", "ds007522/sub-HML0074/ses-01/anat/sub-HML0074_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0103__332", "ds007522/sub-HML0103/ses-01/anat/sub-HML0103_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0104__333", "ds007522/sub-HML0104/ses-01/anat/sub-HML0104_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0111__334", "ds007522/sub-HML0111/ses-01/anat/sub-HML0111_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0231__335", "ds007522/sub-HML0231/ses-01/anat/sub-HML0231_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0233__336", "ds007522/sub-HML0233/ses-01/anat/sub-HML0233_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0360__337", "ds007522/sub-HML0360/ses-01/anat/sub-HML0360_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0372__338", "ds007522/sub-HML0372/ses-01/anat/sub-HML0372_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0388__339", "ds007522/sub-HML0388/ses-01/anat/sub-HML0388_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0417__340", "ds007522/sub-HML0417/ses-01/anat/sub-HML0417_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0446__341", "ds007522/sub-HML0446/ses-01/anat/sub-HML0446_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0506__342", "ds007522/sub-HML0506/ses-01/anat/sub-HML0506_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0592__343", "ds007522/sub-HML0592/ses-01/anat/sub-HML0592_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0637__344", "ds007522/sub-HML0637/ses-01/anat/sub-HML0637_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0669__345", "ds007522/sub-HML0669/ses-01/anat/sub-HML0669_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0697__346", "ds007522/sub-HML0697/ses-01/anat/sub-HML0697_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0744__347", "ds007522/sub-HML0744/ses-01/anat/sub-HML0744_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0863__348", "ds007522/sub-HML0863/ses-01/anat/sub-HML0863_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0868__349", "ds007522/sub-HML0868/ses-01/anat/sub-HML0868_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0928__350", "ds007522/sub-HML0928/ses-01/anat/sub-HML0928_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0937__351", "ds007522/sub-HML0937/ses-01/anat/sub-HML0937_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0953__352", "ds007522/sub-HML0953/ses-01/anat/sub-HML0953_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1083__353", "ds007522/sub-HML1083/ses-01/anat/sub-HML1083_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1120__354", "ds007522/sub-HML1120/ses-01/anat/sub-HML1120_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1131__355", "ds007522/sub-HML1131/ses-01/anat/sub-HML1131_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1134__356", "ds007522/sub-HML1134/ses-01/anat/sub-HML1134_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1136__357", "ds007522/sub-HML1136/ses-01/anat/sub-HML1136_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1155__358", "ds007522/sub-HML1155/ses-01/anat/sub-HML1155_ses-01_rec-defaced_T1w.nii.gz"], ["ds003592__sub-02__366", "ds003592/sub-02/ses-1/anat/sub-02_ses-1_T1w.nii.gz"], ["ds003592__sub-110__367", "ds003592/sub-110/ses-1/anat/sub-110_ses-1_T1w.nii.gz"], ["ds003592__sub-298__368", "ds003592/sub-298/ses-1/anat/sub-298_ses-1_T1w.nii.gz"], ["ds005270__sub-130__369", "ds005270/sub-130/anat/sub-130_T1w.nii.gz"], ["ds005270__sub-149__370", "ds005270/sub-149/anat/sub-149_T1w.nii.gz"], ["ds005123__sub-10617__371", "ds005123/sub-10617/anat/sub-10617_T1w.nii.gz"], ["ds005405__sub-056__372", "ds005405/sub-056/ses-bhb/anat/sub-056_ses-bhb_run-1_T1w.nii"], ["ds007831__sub-GNJ__373", "ds007831/sub-GNJ/anat/sub-GNJ_T1w.nii.gz"], ["ds007831__sub-LXR__374", "ds007831/sub-LXR/anat/sub-LXR_T1w.nii.gz"], ["ds004711__sub-041__375", "ds004711/sub-041/anat/sub-041_T1w.nii.gz"], ["ds004725__sub-21002__376", "ds004725/sub-21002/anat/sub-21002_T1w.nii"], ["ds007522__sub-HML0027__377", "ds007522/sub-HML0027/ses-01/anat/sub-HML0027_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0046__378", "ds007522/sub-HML0046/ses-01/anat/sub-HML0046_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0054__379", "ds007522/sub-HML0054/ses-01/anat/sub-HML0054_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0062__380", "ds007522/sub-HML0062/ses-01/anat/sub-HML0062_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0066__381", "ds007522/sub-HML0066/ses-01/anat/sub-HML0066_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0069__382", "ds007522/sub-HML0069/ses-01/anat/sub-HML0069_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0077__383", "ds007522/sub-HML0077/ses-01/anat/sub-HML0077_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0164__384", "ds007522/sub-HML0164/ses-01/anat/sub-HML0164_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0171__385", "ds007522/sub-HML0171/ses-01/anat/sub-HML0171_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0173__386", "ds007522/sub-HML0173/ses-01/anat/sub-HML0173_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0212__387", "ds007522/sub-HML0212/ses-01/anat/sub-HML0212_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0214__388", "ds007522/sub-HML0214/ses-01/anat/sub-HML0214_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0329__389", "ds007522/sub-HML0329/ses-01/anat/sub-HML0329_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0430__390", "ds007522/sub-HML0430/ses-01/anat/sub-HML0430_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0514__391", "ds007522/sub-HML0514/ses-01/anat/sub-HML0514_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0525__392", "ds007522/sub-HML0525/ses-01/anat/sub-HML0525_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0565__393", "ds007522/sub-HML0565/ses-01/anat/sub-HML0565_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0619__394", "ds007522/sub-HML0619/ses-01/anat/sub-HML0619_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0671__395", "ds007522/sub-HML0671/ses-01/anat/sub-HML0671_ses-01_rec-defaced_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
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
