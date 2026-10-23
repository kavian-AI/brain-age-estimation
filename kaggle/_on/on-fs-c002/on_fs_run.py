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

PART = 2
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["ds007522__sub-HML0786__89", "ds007522/sub-HML0786/ses-01/anat/sub-HML0786_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0845__90", "ds007522/sub-HML0845/ses-01/anat/sub-HML0845_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0857__91", "ds007522/sub-HML0857/ses-01/anat/sub-HML0857_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0865__92", "ds007522/sub-HML0865/ses-01/anat/sub-HML0865_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0924__93", "ds007522/sub-HML0924/ses-01/anat/sub-HML0924_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1042__94", "ds007522/sub-HML1042/ses-01/anat/sub-HML1042_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1063__95", "ds007522/sub-HML1063/ses-01/anat/sub-HML1063_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1068__96", "ds007522/sub-HML1068/ses-02/anat/sub-HML1068_ses-02_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1181__97", "ds007522/sub-HML1181/ses-01/anat/sub-HML1181_ses-01_rec-defaced_T1w.nii.gz"], ["ds003592__sub-10__107", "ds003592/sub-10/ses-1/anat/sub-10_ses-1_T1w.nii.gz"], ["ds003592__sub-195__108", "ds003592/sub-195/ses-1/anat/sub-195_ses-1_T1w.nii.gz"], ["ds003592__sub-201__109", "ds003592/sub-201/ses-1/anat/sub-201_ses-1_T1w.nii.gz"], ["ds003592__sub-291__110", "ds003592/sub-291/ses-1/anat/sub-291_ses-1_T1w.nii.gz"], ["ds003592__sub-294__111", "ds003592/sub-294/ses-1/anat/sub-294_ses-1_T1w.nii.gz"], ["ds005270__sub-004__112", "ds005270/sub-004/anat/sub-004_T1w.nii.gz"], ["ds005270__sub-076__113", "ds005270/sub-076/anat/sub-076_T1w.nii.gz"], ["ds005026__sub-15__114", "ds005026/sub-15/ses-01/anat/sub-15_ses-01_T1w.nii.gz"], ["ds004725__sub-22056__115", "ds004725/sub-22056/anat/sub-22056_T1w.nii"], ["ds007831__sub-CUI__116", "ds007831/sub-CUI/anat/sub-CUI_T1w.nii.gz"], ["ds004725__sub-21024__117", "ds004725/sub-21024/anat/sub-21024_T1w.nii"], ["ds007831__sub-NOX__118", "ds007831/sub-NOX/anat/sub-NOX_T1w.nii.gz"], ["ds007522__sub-HML0033__119", "ds007522/sub-HML0033/ses-01/anat/sub-HML0033_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0223__120", "ds007522/sub-HML0223/ses-01/anat/sub-HML0223_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0237__121", "ds007522/sub-HML0237/ses-01/anat/sub-HML0237_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0240__122", "ds007522/sub-HML0240/ses-01/anat/sub-HML0240_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0252__123", "ds007522/sub-HML0252/ses-01/anat/sub-HML0252_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0437__124", "ds007522/sub-HML0437/ses-01/anat/sub-HML0437_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0444__125", "ds007522/sub-HML0444/ses-01/anat/sub-HML0444_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0458__126", "ds007522/sub-HML0458/ses-01/anat/sub-HML0458_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0466__127", "ds007522/sub-HML0466/ses-01/anat/sub-HML0466_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0485__128", "ds007522/sub-HML0485/ses-01/anat/sub-HML0485_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0562__129", "ds007522/sub-HML0562/ses-01/anat/sub-HML0562_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0633__130", "ds007522/sub-HML0633/ses-01/anat/sub-HML0633_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0728__131", "ds007522/sub-HML0728/ses-01/anat/sub-HML0728_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0754__132", "ds007522/sub-HML0754/ses-01/anat/sub-HML0754_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0759__133", "ds007522/sub-HML0759/ses-01/anat/sub-HML0759_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0787__134", "ds007522/sub-HML0787/ses-01/anat/sub-HML0787_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0822__135", "ds007522/sub-HML0822/ses-01/anat/sub-HML0822_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0832__136", "ds007522/sub-HML0832/ses-01/anat/sub-HML0832_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0905__137", "ds007522/sub-HML0905/ses-01/anat/sub-HML0905_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1082__138", "ds007522/sub-HML1082/ses-02/anat/sub-HML1082_ses-02_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1147__139", "ds007522/sub-HML1147/ses-01/anat/sub-HML1147_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1170__140", "ds007522/sub-HML1170/ses-01/anat/sub-HML1170_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1182__141", "ds007522/sub-HML1182/ses-01/anat/sub-HML1182_ses-01_rec-defaced_T1w.nii.gz"], ["ds003592__sub-03__148", "ds003592/sub-03/ses-1/anat/sub-03_ses-1_T1w.nii.gz"], ["ds003592__sub-269__149", "ds003592/sub-269/ses-1/anat/sub-269_ses-1_T1w.nii.gz"], ["ds003592__sub-272__150", "ds003592/sub-272/ses-1/anat/sub-272_ses-1_T1w.nii.gz"], ["ds003592__sub-53__151", "ds003592/sub-53/ses-1/anat/sub-53_ses-1_T1w.nii.gz"], ["ds005270__sub-026__152", "ds005270/sub-026/anat/sub-026_T1w.nii.gz"], ["ds005270__sub-046__153", "ds005270/sub-046/anat/sub-046_T1w.nii.gz"], ["ds005405__sub-076__154", "ds005405/sub-076/ses-bhb/anat/sub-076_ses-bhb_run-1_T1w.nii"], ["ds004725__sub-22036__155", "ds004725/sub-22036/anat/sub-22036_T1w.nii"], ["ds007831__sub-DZG__156", "ds007831/sub-DZG/anat/sub-DZG_T1w.nii.gz"], ["ds004711__sub-179__157", "ds004711/sub-179/anat/sub-179_T1w.nii.gz"], ["ds007831__sub-JFV__158", "ds007831/sub-JFV/anat/sub-JFV_T1w.nii.gz"], ["ds004711__sub-184__159", "ds004711/sub-184/anat/sub-184_T1w.nii.gz"], ["ds007831__sub-SEJ__160", "ds007831/sub-SEJ/anat/sub-SEJ_T1w.nii.gz"], ["ds004725__sub-22051__161", "ds004725/sub-22051/anat/sub-22051_T1w.nii"]]''')          # [[scan_id, s3_key], ...] for this part
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
