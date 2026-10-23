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

PART = 27
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["obhb__762537788840", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-762537788840/ses-1/sub-762537788840_T1w.nii.gz"], ["sald__sub-031534", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031534/anat/sub-031534_T1w.nii.gz"], ["sald__sub-031694", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031694/anat/sub-031694_T1w.nii.gz"], ["obhb__189320820344", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-189320820344/ses-1/sub-189320820344_T1w.nii.gz"], ["obhb__369885949666", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-369885949666/ses-1/sub-369885949666_T1w.nii.gz"], ["obhb__431876074723", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-431876074723/ses-1/sub-431876074723_T1w.nii.gz"], ["obhb__530934590183", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-530934590183/ses-1/sub-530934590183_T1w.nii.gz"], ["obhb__561734337718", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-561734337718/ses-1/sub-561734337718_T1w.nii.gz"], ["obhb__585134564411", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-585134564411/ses-1/sub-585134564411_T1w.nii.gz"], ["obhb__624448003003", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-624448003003/ses-1/sub-624448003003_T1w.nii.gz"], ["obhb__657153755669", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-657153755669/ses-1/sub-657153755669_T1w.nii.gz"], ["obhb__781171966594", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-781171966594/ses-1/sub-781171966594_T1w.nii.gz"], ["obhb__851549490177", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-851549490177/ses-1/sub-851549490177_T1w.nii.gz"], ["obhb__890940320423", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-890940320423/ses-1/sub-890940320423_T1w.nii.gz"], ["obhb__933451667612", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-933451667612/ses-1/sub-933451667612_T1w.nii.gz"], ["obhb__962095900736", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-962095900736/ses-1/sub-962095900736_T1w.nii.gz"], ["obhb__971873286741", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-971873286741/ses-1/sub-971873286741_T1w.nii.gz"], ["obhb__739971221380", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-739971221380/ses-1/sub-739971221380_T1w.nii.gz"], ["obhb__841748968258", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-841748968258/ses-1/sub-841748968258_T1w.nii.gz"], ["obhb__872361387502", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-872361387502/ses-1/sub-872361387502_T1w.nii.gz"], ["obhb__892723361287", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-892723361287/ses-1/sub-892723361287_T1w.nii.gz"], ["sald__sub-031517", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031517/anat/sub-031517_T1w.nii.gz"], ["oasis1__OAS1_0192_MR1__mpr-1", "oasis1:disc6:OAS1_0192_MR1:mpr-1"], ["oasis1__OAS1_0319_MR1__mpr-1", "oasis1:disc9:OAS1_0319_MR1:mpr-1"], ["oasis1__OAS1_0376_MR1__mpr-1", "oasis1:disc10:OAS1_0376_MR1:mpr-1"], ["ds004711__sub-053__2097", "ds004711/sub-053/anat/sub-053_T1w.nii.gz"], ["obhb__219466815546", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-219466815546/ses-1/sub-219466815546_T1w.nii.gz"], ["obhb__935090080624", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-935090080624/ses-1/sub-935090080624_T1w.nii.gz"], ["ds007831__sub-SEW__2098", "ds007831/sub-SEW/anat/sub-SEW_T1w.nii.gz"], ["obhb__199603028032", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-199603028032/ses-1/sub-199603028032_T1w.nii.gz"], ["obhb__970977623265", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-970977623265/ses-1/sub-970977623265_T1w.nii.gz"], ["ds004725__sub-12051__2099", "ds004725/sub-12051/anat/sub-12051_T1w.nii"], ["ds005752__sub-ON01802__2100", "ds005752/sub-ON01802/ses-01/anat/sub-ON01802_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON02693__2101", "ds005752/sub-ON02693/ses-01/anat/sub-ON02693_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON07392__2102", "ds005752/sub-ON07392/ses-01/anat/sub-ON07392_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON26309__2103", "ds005752/sub-ON26309/ses-01/anat/sub-ON26309_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON33827__2104", "ds005752/sub-ON33827/ses-01/anat/sub-ON33827_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON88753__2105", "ds005752/sub-ON88753/ses-01/anat/sub-ON88753_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON88762__2106", "ds005752/sub-ON88762/ses-01/anat/sub-ON88762_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON93222__2107", "ds005752/sub-ON93222/ses-01/anat/sub-ON93222_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds003592__sub-265__2108", "ds003592/sub-265/ses-1/anat/sub-265_ses-1_T1w.nii.gz"], ["ds003592__sub-80__2109", "ds003592/sub-80/ses-1/anat/sub-80_ses-1_T1w.nii.gz"], ["ds005270__sub-010__2110", "ds005270/sub-010/anat/sub-010_T1w.nii.gz"], ["ds005270__sub-054__2111", "ds005270/sub-054/anat/sub-054_T1w.nii.gz"], ["ds005270__sub-062__2112", "ds005270/sub-062/anat/sub-062_T1w.nii.gz"], ["ds005270__sub-075__2113", "ds005270/sub-075/anat/sub-075_T1w.nii.gz"], ["ds005270__sub-082__2114", "ds005270/sub-082/anat/sub-082_T1w.nii.gz"], ["ds005270__sub-084__2115", "ds005270/sub-084/anat/sub-084_T1w.nii.gz"], ["ds005270__sub-105__2116", "ds005270/sub-105/anat/sub-105_T1w.nii.gz"], ["ds005270__sub-146__2117", "ds005270/sub-146/anat/sub-146_T1w.nii.gz"], ["ds005270__sub-157__2118", "ds005270/sub-157/anat/sub-157_T1w.nii.gz"], ["ds005405__sub-085__2119", "ds005405/sub-085/ses-bhb/anat/sub-085_ses-bhb_run-1_T1w.nii"], ["ds006395__sub-C01738__2120", "ds006395/sub-C01738/anat/sub-C01738_T1w.nii.gz"], ["obhb__104756934188", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-104756934188/ses-1/sub-104756934188_T1w.nii.gz"], ["obhb__169297286201", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-169297286201/ses-1/sub-169297286201_T1w.nii.gz"], ["obhb__211479171648", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-211479171648/ses-1/sub-211479171648_T1w.nii.gz"], ["obhb__213536717598", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-213536717598/ses-1/sub-213536717598_T1w.nii.gz"], ["obhb__238319148726", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-238319148726/ses-1/sub-238319148726_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
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


O1_LOCK = {}
O1_GUARD = threading.Lock()


def oasis1_file(key):
    """key = oasis1:disc<N>:<SESSION>:mpr-<k>. Streams the WashU disc tarball once per kernel and
    keeps only the RAW MPRAGE files of the sessions this kernel needs; returns the .hdr path."""
    _, disc, sess, mpr = key.split(":")
    out = f"/tmp/o1/{sess}_{mpr}_anon.hdr"
    with O1_GUARD:
        lock = O1_LOCK.setdefault(disc, threading.Lock())
    with lock:
        if not os.path.exists(f"/tmp/o1/.{disc}.done"):
            import tarfile
            need = {(k.split(":")[2], k.split(":")[3]) for _, k in JOBS if k.startswith(f"oasis1:{disc}:")}
            os.makedirs("/tmp/o1", exist_ok=True)
            url = f"https://download.nrg.wustl.edu/data/oasis_cross-sectional_{disc}.tar.gz"
            with urllib.request.urlopen(url, timeout=600) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tf:
                for m in tf:
                    b = os.path.basename(m.name)
                    parts = b.split("_")
                    if m.isfile() and "/RAW/" in m.name and len(parts) >= 4:
                        if (f"{parts[0]}_{parts[1]}_{parts[2]}", parts[3]) in need:
                            with open(f"/tmp/o1/{b}", "wb") as fo:
                                fo.write(tf.extractfile(m).read())
            open(f"/tmp/o1/.{disc}.done", "w").write("ok")
    return out


def fetch_one(job, d):
    sid, key = job
    if key.startswith("oasis1:"):
        try:
            h = oasis1_file(key)
            im = nib.load(h)
            nib.save(nib.Nifti1Image(np.asanyarray(im.dataobj).astype(np.float32).squeeze(), im.affine),
                     f"{d}/{sid}.nii.gz")
            json.dump({"Manufacturer": "Siemens", "ManufacturersModelName": "Vision", "MagneticFieldStrength": 1.5},
                      open(f"{d}/{sid}.meta.json", "w"))
            return True
        except Exception as e:
            print(f"  oasis1 fail {sid}: {type(e).__name__} {e}", flush=True)
            return False
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
