# FastSurfer + SynthStrip on a Kaggle TPU VM -- using its CPUs, not the TPU chip.
#
# The TPU itself cannot run FastSurfer/SynthStrip (PyTorch models without XLA support),
# but a Kaggle TPU VM is a large CPU machine. We run WORKERS independent FastSurfer
# processes side by side (THREADS CPU threads each), one scan at a time per worker:
# download -> FastSurfer (CPU) -> SynthStrip (CPU) -> one JSON row, appended under a lock.
# Same outputs as the GPU/CPU kernels (on_fs_template.py), so results pool directly.
# New scans stop being started at BUDGET_S; running ones finish (TPU sessions end at 9 h).
import concurrent.futures as cf
import gzip, json, os, shutil, subprocess, threading, time, urllib.request

PART = 0
T0 = time.time()
BUDGET_S = 7.6 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''__JOBS__''')
NCPU = os.cpu_count() or 8
THREADS = 4
try:
    RAM_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
except Exception:
    RAM_GB = 32
WORKERS = max(1, min(48, NCPU // THREADS, int(RAM_GB // 7)))     # ~7 GB RAM per FastSurfer CPU process
LOCK = threading.Lock()


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)


print("PART", PART, "scans", len(JOBS), "cpus", NCPU, "RAM GB", round(RAM_GB), "workers", WORKERS, "x threads", THREADS, flush=True)
print(sh("free -g | head -2").stdout, flush=True)
r = sh("git clone --depth 1 --branch v2.4.2 https://github.com/Deep-MI/FastSurfer.git /tmp/FastSurfer")
print("fastsurfer clone rc", r.returncode, flush=True)
if sh("python -c 'import torch'").returncode != 0:
    r = sh("pip install -q torch --index-url https://download.pytorch.org/whl/cpu")
    print("torch (cpu) install rc", r.returncode, r.stderr[-300:], flush=True)
r = sh("pip install -q yacs simpleitk nibabel scikit-image h5py lapy pandas surfa")
print("pip rc", r.returncode, r.stderr[-300:], flush=True)
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
# warm-up: download FastSurfer's checkpoints once, before the workers start in parallel
r = sh("cd /tmp/FastSurfer && PYTHONPATH=/tmp/FastSurfer python FastSurferCNN/download_checkpoints.py --vinn")
print("checkpoint download rc", r.returncode, r.stderr[-300:], flush=True)

import nibabel as nib
import numpy as np

O1_LOCK = {}
O1_GUARD = threading.Lock()


def oasis1_file(key):
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


def fetch(sid, key, d):
    p = f"{d}/{sid}.nii.gz"
    meta = {}
    if key.startswith("oasis1:"):
        im = nib.load(oasis1_file(key))
        arr = np.asanyarray(im.dataobj).astype(np.float32).squeeze()
        # OASIS-1 RAW: Analyze orient=2 (sagittal), ignored by nibabel -> set the orientation explicitly
        aff = np.array([[0, 0, -1.25, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
        aff[:3, 3] = -aff[:3, :3] @ (np.array(arr.shape) / 2.0)
        nib.save(nib.Nifti1Image(arr, aff), p)
        meta = {"Manufacturer": "Siemens", "ManufacturersModelName": "Vision", "MagneticFieldStrength": 1.5,
                "o1_orient": "ASL-v2"}
    else:
        url = key if key.startswith("http") else S3 + key
        for attempt in range(4):
            try:
                if key.endswith(".nii.gz"):
                    urllib.request.urlretrieve(url, p)
                else:
                    urllib.request.urlretrieve(url, f"{d}/{sid}.nii")
                    with open(f"{d}/{sid}.nii", "rb") as fi, gzip.open(p, "wb", compresslevel=1) as fo:
                        shutil.copyfileobj(fi, fo)
                    os.remove(f"{d}/{sid}.nii")
                break
            except Exception as e:
                print(f"  download retry {sid}: {type(e).__name__}", flush=True)
                time.sleep(10)
        if not key.startswith("http"):
            ds = key.split("/")[0]
            for side in (key.rsplit(".nii", 1)[0] + ".json", f"{ds}/T1w.json"):
                try:
                    meta = json.loads(urllib.request.urlopen(S3 + side, timeout=60).read())
                    break
                except Exception:
                    pass
    return p, {k: meta.get(k) for k in ("Manufacturer", "ManufacturersModelName", "MagneticFieldStrength",
                                       "SoftwareVersions", "InstitutionName", "o1_orient") if k in meta}


def work(job):
    sid, key = job
    if time.time() - T0 > BUDGET_S:
        return "skipped"
    t1 = time.time()
    d = f"/tmp/w/{sid}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    try:
        p, meta = fetch(sid, key, d)
        r = sh(f"cd /tmp/FastSurfer && PYTHONPATH=/tmp/FastSurfer OMP_NUM_THREADS={THREADS} "
               f"python FastSurferCNN/run_prediction.py --t1 {p} --sid {sid} --sd {d}/out "
               f"--device cpu --viewagg_device cpu --threads {THREADS} --vox_size 1")
        segp = f"{d}/out/{sid}/mri/aparc.DKTatlas+aseg.deep.mgz"
        if not os.path.exists(segp):
            print(f"  FAIL seg {sid}: {r.stderr[-300:]}", flush=True)
            return "fail"
        s = nib.load(segp)
        a = np.asanyarray(s.dataobj).astype(np.int32)
        vox = float(np.prod(s.header.get_zooms()[:3]))
        labs, cnt = np.unique(a[a > 0], return_counts=True)
        im = nib.load(p)
        row = {"scan": sid, "key": key, "zooms_in": [float(z) for z in im.header.get_zooms()[:3]],
               "shape_in": [int(x) for x in im.shape[:3]]}
        row.update(meta)
        mp = f"{d}/out/{sid}/mri/mask.mgz"
        if os.path.exists(mp):
            row["mask_vol"] = float((np.asanyarray(nib.load(mp).dataobj) > 0).sum()) * vox
        rs = sh(f"OMP_NUM_THREADS={THREADS} python /tmp/fs/mri_synthstrip -i {p} -m {d}/ss.nii.gz -t {THREADS}")
        if os.path.exists(f"{d}/ss.nii.gz"):
            mm = nib.load(f"{d}/ss.nii.gz")
            row["icv_synthstrip"] = float((np.asanyarray(mm.dataobj) > 0).sum()) * float(np.prod(mm.header.get_zooms()[:3]))
        else:
            print(f"  synthstrip fail {sid}: {rs.stderr[-200:]}", flush=True)
        row.update({f"V{int(l)}": float(n) * vox for l, n in zip(labs, cnt)})
        row["seconds"] = round(time.time() - t1, 1)
        with LOCK:
            with open(OUT, "a") as fh:
                fh.write(json.dumps(row) + chr(10))
        return "ok"
    except Exception as e:
        print(f"  ERROR {sid}: {type(e).__name__} {e}", flush=True)
        return "fail"
    finally:
        shutil.rmtree(d, ignore_errors=True)


done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["scan"] for l in open(OUT)}
todo = [j for j in JOBS if j[0] not in done]
n_ok = 0
n_fail = 0
STOP = False
_work = work


def work(job):                      # early abort: if the first scans all fail, stop the session
    if STOP:
        return "skipped"
    return _work(job)


with cf.ThreadPoolExecutor(WORKERS) as ex:
    for i, res in enumerate(ex.map(work, todo)):
        n_ok += res == "ok"
        n_fail += res == "fail"
        if n_ok == 0 and n_fail >= 6:
            STOP = True
            print("first scans all failed - stopping", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(todo)} handled, {n_ok} ok, elapsed {(time.time()-T0)/3600:.2f} h", flush=True)
print(f"finished: {n_ok} rows, {(time.time()-T0)/3600:.2f} h", flush=True)
