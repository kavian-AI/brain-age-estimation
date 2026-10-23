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

PART = 14
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["obhb__907846695925", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-907846695925/ses-1/sub-907846695925_T1w.nii.gz"], ["obhb__862964317430", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-862964317430/ses-1/sub-862964317430_T1w.nii.gz"], ["obhb__940165370546", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-940165370546/ses-1/sub-940165370546_T1w.nii.gz"], ["obhb__975835056753", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-975835056753/ses-1/sub-975835056753_T1w.nii.gz"], ["ds007831__sub-UJA__2071", "ds007831/sub-UJA/anat/sub-UJA_T1w.nii.gz"], ["ds004711__sub-135__2072", "ds004711/sub-135/anat/sub-135_T1w.nii.gz"], ["ds004711__sub-073__2073", "ds004711/sub-073/anat/sub-073_T1w.nii.gz"], ["obhb__310792111263", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-310792111263/ses-1/sub-310792111263_T1w.nii.gz"], ["ds004711__sub-128__2074", "ds004711/sub-128/anat/sub-128_T1w.nii.gz"], ["ds004711__sub-057__2075", "ds004711/sub-057/anat/sub-057_T1w.nii.gz"], ["ds004711__sub-150__2076", "ds004711/sub-150/anat/sub-150_T1w.nii.gz"], ["ds004725__sub-11022__2077", "ds004725/sub-11022/anat/sub-11022_T1w.nii"], ["ds005752__sub-ON26854__2078", "ds005752/sub-ON26854/ses-01/anat/sub-ON26854_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON76510__2079", "ds005752/sub-ON76510/ses-01/anat/sub-ON76510_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON78603__2080", "ds005752/sub-ON78603/ses-01/anat/sub-ON78603_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON85400__2081", "ds005752/sub-ON85400/ses-01/anat/sub-ON85400_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON87054__2082", "ds005752/sub-ON87054/ses-01/anat/sub-ON87054_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON88512__2083", "ds005752/sub-ON88512/ses-01/anat/sub-ON88512_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON98826__2084", "ds005752/sub-ON98826/ses-01/anat/sub-ON98826_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds003592__sub-135__2085", "ds003592/sub-135/ses-1/anat/sub-135_ses-1_T1w.nii.gz"], ["ds003592__sub-43__2086", "ds003592/sub-43/ses-1/anat/sub-43_ses-1_T1w.nii.gz"], ["ds005270__sub-035__2087", "ds005270/sub-035/anat/sub-035_T1w.nii.gz"], ["ds005270__sub-118__2088", "ds005270/sub-118/anat/sub-118_T1w.nii.gz"], ["ds005270__sub-150__2089", "ds005270/sub-150/anat/sub-150_T1w.nii.gz"], ["ds005270__sub-158__2090", "ds005270/sub-158/anat/sub-158_T1w.nii.gz"], ["ds005123__sub-10640__2091", "ds005123/sub-10640/anat/sub-10640_T1w.nii.gz"], ["ds005123__sub-10897__2092", "ds005123/sub-10897/anat/sub-10897_T1w.nii.gz"], ["ds005405__sub-058__2093", "ds005405/sub-058/ses-bhb/anat/sub-058_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-105__2094", "ds005405/sub-105/ses-bhb/anat/sub-105_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-122__2095", "ds005405/sub-122/ses-bhb/anat/sub-122_ses-bhb_run-1_T1w.nii"], ["ds006395__sub-C01809__2096", "ds006395/sub-C01809/anat/sub-C01809_T1w.nii.gz"], ["obhb__101404752059", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-101404752059/ses-1/sub-101404752059_T1w.nii.gz"], ["obhb__107912335979", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-107912335979/ses-1/sub-107912335979_T1w.nii.gz"], ["obhb__142608681206", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-142608681206/ses-1/sub-142608681206_T1w.nii.gz"], ["obhb__145580633313", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-145580633313/ses-1/sub-145580633313_T1w.nii.gz"], ["obhb__174036160594", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-174036160594/ses-1/sub-174036160594_T1w.nii.gz"], ["obhb__189320820344", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-189320820344/ses-1/sub-189320820344_T1w.nii.gz"], ["obhb__369885949666", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-369885949666/ses-1/sub-369885949666_T1w.nii.gz"], ["obhb__431876074723", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-431876074723/ses-1/sub-431876074723_T1w.nii.gz"], ["obhb__530934590183", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-530934590183/ses-1/sub-530934590183_T1w.nii.gz"], ["obhb__561734337718", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-561734337718/ses-1/sub-561734337718_T1w.nii.gz"], ["obhb__585134564411", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-585134564411/ses-1/sub-585134564411_T1w.nii.gz"], ["obhb__624448003003", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-624448003003/ses-1/sub-624448003003_T1w.nii.gz"], ["obhb__657153755669", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-657153755669/ses-1/sub-657153755669_T1w.nii.gz"], ["obhb__781171966594", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-781171966594/ses-1/sub-781171966594_T1w.nii.gz"], ["obhb__851549490177", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-851549490177/ses-1/sub-851549490177_T1w.nii.gz"], ["obhb__890940320423", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-890940320423/ses-1/sub-890940320423_T1w.nii.gz"], ["obhb__933451667612", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-933451667612/ses-1/sub-933451667612_T1w.nii.gz"], ["obhb__962095900736", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-962095900736/ses-1/sub-962095900736_T1w.nii.gz"], ["obhb__971873286741", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-971873286741/ses-1/sub-971873286741_T1w.nii.gz"], ["obhb__739971221380", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-739971221380/ses-1/sub-739971221380_T1w.nii.gz"], ["obhb__841748968258", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-841748968258/ses-1/sub-841748968258_T1w.nii.gz"], ["obhb__872361387502", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-872361387502/ses-1/sub-872361387502_T1w.nii.gz"], ["obhb__892723361287", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-892723361287/ses-1/sub-892723361287_T1w.nii.gz"], ["ds004711__sub-053__2097", "ds004711/sub-053/anat/sub-053_T1w.nii.gz"], ["obhb__219466815546", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-219466815546/ses-1/sub-219466815546_T1w.nii.gz"], ["obhb__935090080624", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-935090080624/ses-1/sub-935090080624_T1w.nii.gz"], ["ds007831__sub-SEW__2098", "ds007831/sub-SEW/anat/sub-SEW_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
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
