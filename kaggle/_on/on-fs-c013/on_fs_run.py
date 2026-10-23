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

PART = 13
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["obhb__403893360607", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-403893360607/ses-1/sub-403893360607_T1w.nii.gz"], ["obhb__423076478965", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-423076478965/ses-1/sub-423076478965_T1w.nii.gz"], ["obhb__455375044946", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-455375044946/ses-1/sub-455375044946_T1w.nii.gz"], ["obhb__470124848081", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-470124848081/ses-1/sub-470124848081_T1w.nii.gz"], ["obhb__499165373448", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-499165373448/ses-1/sub-499165373448_T1w.nii.gz"], ["obhb__750841455509", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-750841455509/ses-1/sub-750841455509_T1w.nii.gz"], ["obhb__851134973595", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-851134973595/ses-1/sub-851134973595_T1w.nii.gz"], ["obhb__900348386642", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-900348386642/ses-1/sub-900348386642_T1w.nii.gz"], ["obhb__917008309577", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-917008309577/ses-1/sub-917008309577_T1w.nii.gz"], ["obhb__974416298113", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-974416298113/ses-1/sub-974416298113_T1w.nii.gz"], ["obhb__978703528815", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-978703528815/ses-1/sub-978703528815_T1w.nii.gz"], ["obhb__225798660070", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-225798660070/ses-1/sub-225798660070_T1w.nii.gz"], ["obhb__361321219772", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-361321219772/ses-1/sub-361321219772_T1w.nii.gz"], ["obhb__515731060094", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-515731060094/ses-1/sub-515731060094_T1w.nii.gz"], ["obhb__780945670746", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-780945670746/ses-1/sub-780945670746_T1w.nii.gz"], ["obhb__117730404599", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-117730404599/ses-1/sub-117730404599_T1w.nii.gz"], ["obhb__121236619259", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-121236619259/ses-1/sub-121236619259_T1w.nii.gz"], ["obhb__192969634654", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-192969634654/ses-1/sub-192969634654_T1w.nii.gz"], ["obhb__336832105241", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-336832105241/ses-1/sub-336832105241_T1w.nii.gz"], ["obhb__345201265289", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-345201265289/ses-1/sub-345201265289_T1w.nii.gz"], ["obhb__431389684208", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-431389684208/ses-1/sub-431389684208_T1w.nii.gz"], ["obhb__464956463990", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-464956463990/ses-1/sub-464956463990_T1w.nii.gz"], ["obhb__505669140091", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-505669140091/ses-1/sub-505669140091_T1w.nii.gz"], ["obhb__617548875008", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-617548875008/ses-1/sub-617548875008_T1w.nii.gz"], ["obhb__746032496734", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-746032496734/ses-1/sub-746032496734_T1w.nii.gz"], ["obhb__766124719459", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-766124719459/ses-1/sub-766124719459_T1w.nii.gz"], ["obhb__768852153668", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-768852153668/ses-1/sub-768852153668_T1w.nii.gz"], ["obhb__791276074550", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-791276074550/ses-1/sub-791276074550_T1w.nii.gz"], ["obhb__808051274484", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-808051274484/ses-1/sub-808051274484_T1w.nii.gz"], ["ds004711__sub-085__2049", "ds004711/sub-085/anat/sub-085_T1w.nii.gz"], ["ds004711__sub-112__2050", "ds004711/sub-112/anat/sub-112_run-01_T1w.nii.gz"], ["ds005752__sub-ON09681__2051", "ds005752/sub-ON09681/ses-01/anat/sub-ON09681_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON09766__2052", "ds005752/sub-ON09766/ses-01/anat/sub-ON09766_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON10965__2053", "ds005752/sub-ON10965/ses-01/anat/sub-ON10965_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON64410__2054", "ds005752/sub-ON64410/ses-01/anat/sub-ON64410_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON76625__2055", "ds005752/sub-ON76625/ses-01/anat/sub-ON76625_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON77753__2056", "ds005752/sub-ON77753/ses-01/anat/sub-ON77753_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON85010__2057", "ds005752/sub-ON85010/ses-01/anat/sub-ON85010_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON85305__2058", "ds005752/sub-ON85305/ses-01/anat/sub-ON85305_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON98018__2059", "ds005752/sub-ON98018/ses-01/anat/sub-ON98018_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds003592__sub-15__2060", "ds003592/sub-15/ses-1/anat/sub-15_ses-1_T1w.nii.gz"], ["ds005270__sub-074__2061", "ds005270/sub-074/anat/sub-074_T1w.nii.gz"], ["ds005123__sub-10369__2062", "ds005123/sub-10369/anat/sub-10369_T1w.nii.gz"], ["ds005123__sub-10642__2063", "ds005123/sub-10642/anat/sub-10642_T1w.nii.gz"], ["ds005123__sub-10850__2064", "ds005123/sub-10850/anat/sub-10850_T1w.nii.gz"], ["ds005123__sub-10875__2065", "ds005123/sub-10875/anat/sub-10875_T1w.nii.gz"], ["ds005123__sub-10924__2066", "ds005123/sub-10924/anat/sub-10924_T1w.nii.gz"], ["ds005123__sub-10952__2067", "ds005123/sub-10952/anat/sub-10952_T1w.nii.gz"], ["ds005405__sub-150__2068", "ds005405/sub-150/ses-bhb/anat/sub-150_ses-bhb_run-1_T1w.nii"], ["ds006395__sub-C01715__2069", "ds006395/sub-C01715/anat/sub-C01715_T1w.nii.gz"], ["ds006395__sub-C01810__2070", "ds006395/sub-C01810/anat/sub-C01810_T1w.nii.gz"], ["obhb__158500314736", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-158500314736/ses-1/sub-158500314736_T1w.nii.gz"], ["obhb__299690677871", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-299690677871/ses-1/sub-299690677871_T1w.nii.gz"], ["obhb__300521812714", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-300521812714/ses-1/sub-300521812714_T1w.nii.gz"], ["obhb__506541958125", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-506541958125/ses-1/sub-506541958125_T1w.nii.gz"], ["obhb__535399597427", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-535399597427/ses-1/sub-535399597427_T1w.nii.gz"], ["obhb__550394965552", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-550394965552/ses-1/sub-550394965552_T1w.nii.gz"], ["obhb__762537788840", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-762537788840/ses-1/sub-762537788840_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
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
