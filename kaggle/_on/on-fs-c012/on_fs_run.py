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

PART = 12
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["obhb__870653878584", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-870653878584/ses-1/sub-870653878584_T1w.nii.gz"], ["ds004711__sub-063__2008", "ds004711/sub-063/anat/sub-063_T1w.nii.gz"], ["ds004711__sub-026__2009", "ds004711/sub-026/anat/sub-026_T1w.nii.gz"], ["ds004711__sub-142__2010", "ds004711/sub-142/anat/sub-142_T1w.nii.gz"], ["obhb__353103739610", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-353103739610/ses-1/sub-353103739610_T1w.nii.gz"], ["obhb__317793059909", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-317793059909/ses-1/sub-317793059909_T1w.nii.gz"], ["ds005752__sub-ON13986__2011", "ds005752/sub-ON13986/ses-01/anat/sub-ON13986_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON33159__2012", "ds005752/sub-ON33159/ses-01/anat/sub-ON33159_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON50636__2013", "ds005752/sub-ON50636/ses-01/anat/sub-ON50636_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON86779__2014", "ds005752/sub-ON86779/ses-01/anat/sub-ON86779_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON87362__2015", "ds005752/sub-ON87362/ses-01/anat/sub-ON87362_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON87743__2016", "ds005752/sub-ON87743/ses-01/anat/sub-ON87743_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON95742__2017", "ds005752/sub-ON95742/ses-01/anat/sub-ON95742_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON97526__2018", "ds005752/sub-ON97526/ses-01/anat/sub-ON97526_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds003592__sub-98__2019", "ds003592/sub-98/ses-1/anat/sub-98_ses-1_T1w.nii.gz"], ["ds005270__sub-051__2020", "ds005270/sub-051/anat/sub-051_T1w.nii.gz"], ["ds005270__sub-124__2021", "ds005270/sub-124/anat/sub-124_T1w.nii.gz"], ["ds005123__sub-10589__2022", "ds005123/sub-10589/anat/sub-10589_T1w.nii.gz"], ["ds005123__sub-10685__2023", "ds005123/sub-10685/anat/sub-10685_T1w.nii.gz"], ["ds005123__sub-10801__2024", "ds005123/sub-10801/anat/sub-10801_T1w.nii.gz"], ["ds005123__sub-10958__2025", "ds005123/sub-10958/anat/sub-10958_T1w.nii.gz"], ["ds005405__sub-012__2026", "ds005405/sub-012/ses-bhb/anat/sub-012_ses-bhb_run-1_T1w.nii"], ["obhb__372796067335", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-372796067335/ses-1/sub-372796067335_T1w.nii.gz"], ["obhb__441895637368", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-441895637368/ses-1/sub-441895637368_T1w.nii.gz"], ["obhb__480168790227", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-480168790227/ses-1/sub-480168790227_T1w.nii.gz"], ["obhb__805388577387", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-805388577387/ses-1/sub-805388577387_T1w.nii.gz"], ["obhb__939910199671", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-939910199671/ses-1/sub-939910199671_T1w.nii.gz"], ["ds004711__sub-079__2027", "ds004711/sub-079/anat/sub-079_T1w.nii.gz"], ["ds004711__sub-100__2028", "ds004711/sub-100/anat/sub-100_T1w.nii.gz"], ["obhb__804202082081", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-804202082081/ses-1/sub-804202082081_T1w.nii.gz"], ["ds004711__sub-068__2029", "ds004711/sub-068/anat/sub-068_T1w.nii.gz"], ["ds004711__sub-075__2030", "ds004711/sub-075/anat/sub-075_T1w.nii.gz"], ["obhb__976182490642", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-976182490642/ses-1/sub-976182490642_T1w.nii.gz"], ["obhb__516034188736", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-516034188736/ses-1/sub-516034188736_T1w.nii.gz"], ["ds004711__sub-151__2031", "ds004711/sub-151/anat/sub-151_T1w.nii.gz"], ["obhb__686436805646", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-686436805646/ses-1/sub-686436805646_T1w.nii.gz"], ["ds005752__sub-ON05258__2032", "ds005752/sub-ON05258/ses-01/anat/sub-ON05258_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON52662__2033", "ds005752/sub-ON52662/ses-01/anat/sub-ON52662_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON58053__2034", "ds005752/sub-ON58053/ses-01/anat/sub-ON58053_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON93503__2035", "ds005752/sub-ON93503/ses-01/anat/sub-ON93503_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON96530__2036", "ds005752/sub-ON96530/ses-01/anat/sub-ON96530_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON97252__2037", "ds005752/sub-ON97252/ses-01/anat/sub-ON97252_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON97765__2038", "ds005752/sub-ON97765/ses-01/anat/sub-ON97765_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005270__sub-092__2039", "ds005270/sub-092/anat/sub-092_T1w.nii.gz"], ["ds005270__sub-107__2040", "ds005270/sub-107/anat/sub-107_T1w.nii.gz"], ["ds005270__sub-153__2041", "ds005270/sub-153/anat/sub-153_T1w.nii.gz"], ["ds005123__sub-10674__2042", "ds005123/sub-10674/anat/sub-10674_T1w.nii.gz"], ["ds005123__sub-10677__2043", "ds005123/sub-10677/anat/sub-10677_T1w.nii.gz"], ["ds005123__sub-10720__2044", "ds005123/sub-10720/anat/sub-10720_T1w.nii.gz"], ["ds005123__sub-10748__2045", "ds005123/sub-10748/anat/sub-10748_T1w.nii.gz"], ["ds005405__sub-127__2046", "ds005405/sub-127/ses-bhb/anat/sub-127_ses-bhb_run-1_T1w.nii"], ["ds006395__sub-C01721__2047", "ds006395/sub-C01721/anat/sub-C01721_T1w.nii.gz"], ["ds006395__sub-C01808__2048", "ds006395/sub-C01808/anat/sub-C01808_T1w.nii.gz"], ["obhb__177231880980", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-177231880980/ses-1/sub-177231880980_T1w.nii.gz"], ["obhb__306640245722", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-306640245722/ses-1/sub-306640245722_T1w.nii.gz"], ["obhb__334925895861", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-334925895861/ses-1/sub-334925895861_T1w.nii.gz"], ["obhb__349851138030", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-349851138030/ses-1/sub-349851138030_T1w.nii.gz"], ["obhb__363459366347", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-363459366347/ses-1/sub-363459366347_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
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
            url = key if key.startswith("http") else S3 + key       # OpenBHB lives on HuggingFace
            if key.endswith(".nii.gz"):
                urllib.request.urlretrieve(url, p)
            else:
                urllib.request.urlretrieve(url, f"{d}/{sid}.nii")
                with open(f"{d}/{sid}.nii", "rb") as fi, gzip.open(p, "wb", compresslevel=1) as fo:
                    shutil.copyfileobj(fi, fo)
                os.remove(f"{d}/{sid}.nii")
            meta = {}
            ds = key.split("/")[0]
            for side in (() if key.startswith("http") else (key.rsplit(".nii", 1)[0] + ".json", f"{ds}/T1w.json")):
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
