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

PART = 102
T0 = time.time()
BUDGET_S = 11.0 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["ds007522__sub-HML0150__1755", "ds007522/sub-HML0150/ses-01/anat/sub-HML0150_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0158__1756", "ds007522/sub-HML0158/ses-01/anat/sub-HML0158_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0179__1757", "ds007522/sub-HML0179/ses-01/anat/sub-HML0179_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0191__1758", "ds007522/sub-HML0191/ses-01/anat/sub-HML0191_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0206__1759", "ds007522/sub-HML0206/ses-01/anat/sub-HML0206_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0216__1760", "ds007522/sub-HML0216/ses-01/anat/sub-HML0216_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0386__1761", "ds007522/sub-HML0386/ses-01/anat/sub-HML0386_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0488__1762", "ds007522/sub-HML0488/ses-01/anat/sub-HML0488_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0490__1763", "ds007522/sub-HML0490/ses-01/anat/sub-HML0490_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0497__1764", "ds007522/sub-HML0497/ses-01/anat/sub-HML0497_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0527__1765", "ds007522/sub-HML0527/ses-01/anat/sub-HML0527_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0560__1766", "ds007522/sub-HML0560/ses-01/anat/sub-HML0560_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0593__1767", "ds007522/sub-HML0593/ses-01/anat/sub-HML0593_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0636__1768", "ds007522/sub-HML0636/ses-01/anat/sub-HML0636_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0659__1769", "ds007522/sub-HML0659/ses-01/anat/sub-HML0659_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0660__1770", "ds007522/sub-HML0660/ses-01/anat/sub-HML0660_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0685__1771", "ds007522/sub-HML0685/ses-01/anat/sub-HML0685_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0690__1772", "ds007522/sub-HML0690/ses-01/anat/sub-HML0690_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0691__1773", "ds007522/sub-HML0691/ses-01/anat/sub-HML0691_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0715__1774", "ds007522/sub-HML0715/ses-01/anat/sub-HML0715_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0785__1775", "ds007522/sub-HML0785/ses-01/anat/sub-HML0785_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0817__1776", "ds007522/sub-HML0817/ses-01/anat/sub-HML0817_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0893__1777", "ds007522/sub-HML0893/ses-01/anat/sub-HML0893_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0919__1778", "ds007522/sub-HML0919/ses-01/anat/sub-HML0919_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0974__1779", "ds007522/sub-HML0974/ses-01/anat/sub-HML0974_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0988__1780", "ds007522/sub-HML0988/ses-02/anat/sub-HML0988_ses-02_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1001__1781", "ds007522/sub-HML1001/ses-02/anat/sub-HML1001_ses-02_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1039__1782", "ds007522/sub-HML1039/ses-01/anat/sub-HML1039_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1114__1783", "ds007522/sub-HML1114/ses-01/anat/sub-HML1114_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1160__1784", "ds007522/sub-HML1160/ses-01/anat/sub-HML1160_ses-01_rec-defaced_T1w.nii.gz"], ["ds005752__sub-ON09540__1785", "ds005752/sub-ON09540/ses-01/anat/sub-ON09540_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005270__sub-052__1786", "ds005270/sub-052/anat/sub-052_T1w.nii.gz"], ["ds005123__sub-10584__1787", "ds005123/sub-10584/anat/sub-10584_T1w.nii.gz"], ["ds005123__sub-10661__1788", "ds005123/sub-10661/anat/sub-10661_T1w.nii.gz"], ["ds005405__sub-079__1789", "ds005405/sub-079/ses-bhb/anat/sub-079_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-139__1790", "ds005405/sub-139/ses-bhb/anat/sub-139_ses-bhb_run-1_T1w.nii"], ["ds006395__sub-C01771__1791", "ds006395/sub-C01771/anat/sub-C01771_T1w.nii.gz"], ["ds005026__sub-12__1792", "ds005026/sub-12/ses-01/anat/sub-12_ses-01_T1w.nii.gz"], ["obhb__430905405415", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-430905405415/ses-1/sub-430905405415_T1w.nii.gz"], ["sald__sub-031356", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031356/anat/sub-031356_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
CHUNK = 3


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
# batch runner: one python process per chunk calls SynthStrip's own main() for every image
# (no repeated interpreter start-up / torch import / CUDA initialisation)
open("/tmp/fs/ss_batch.py", "w").write(
    "import sys, runpy" + chr(10) +
    "ns = runpy.run_path('/tmp/fs/mri_synthstrip', run_name='synthstrip_lib')" + chr(10) +
    "extra = sys.argv[2:]" + chr(10) +
    "for line in open(sys.argv[1]).read().splitlines():" + chr(10) +
    "    if not line.strip(): continue" + chr(10) +
    "    img, msk = line.split(chr(9))" + chr(10) +
    "    sys.argv = ['mri_synthstrip', '-i', img, '-m', msk] + extra" + chr(10) +
    "    try: ns['main']()" + chr(10) +
    "    except SystemExit: pass" + chr(10) +
    "    except Exception as e: print('SYNTHSTRIP FAILED', img, e, flush=True)" + chr(10))
os.environ["FREESURFER_HOME"] = "/tmp/fs"

import nibabel as nib
import numpy as np
try:
    import torch
    NGPU = max(1, torch.cuda.device_count()) if False else 1
except Exception:
    NGPU = 1
print("GPUs used:", NGPU, flush=True)

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
            arr = np.asanyarray(im.dataobj).astype(np.float32).squeeze()
            # OASIS-1 RAW is Analyze with orient=2 (sagittal), which nibabel ignores (it assumes axial):
            # axis0 posterior->anterior, axis1 inferior->superior, axis2 sagittal slices (1.25 mm, R->L: checked on the cohort's L/R asymmetries)
            aff = np.array([[0, 0, -1.25, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
            aff[:3, 3] = -aff[:3, :3] @ (np.array(arr.shape) / 2.0)
            nib.save(nib.Nifti1Image(arr, aff), f"{d}/{sid}.nii.gz")
            json.dump({"Manufacturer": "Siemens", "ManufacturersModelName": "Vision", "MagneticFieldStrength": 1.5,
                       "o1_orient": "ASL-v2"}, open(f"{d}/{sid}.meta.json", "w"))
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
last_dt = 0.0
for c, ch in enumerate(chunks):
    if time.time() - T0 + 1.15 * last_dt > BUDGET_S:
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
    # one FastSurfer process per GPU (Kaggle "NvidiaTeslaT4" = 2 x T4), scans split between them
    imgs = sorted(f for f in os.listdir(d) if f.endswith(".nii.gz"))
    groups = [imgs[g::NGPU] for g in range(NGPU)]
    procs = []
    for g, grp in enumerate(groups):
        if not grp:
            continue
        gd = f"{d}/gpu{g}"
        os.makedirs(gd, exist_ok=True)
        for f in grp:
            if not os.path.exists(f"{gd}/{f}"):
                os.symlink(f"{d}/{f}", f"{gd}/{f}")
        cmd = ("cd /tmp/FastSurfer && PYTHONPATH=/tmp/FastSurfer python FastSurferCNN/run_prediction.py "
               f"--in_dir {gd} --tag '*.nii.gz' --remove_suffix .nii.gz --sd /tmp/out "
               f"--device cpu --viewagg_device cpu --threads {max(1, 4 // NGPU)} --vox_size 1 --batch_size 8")
        procs.append(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}))
    outs = [p.communicate() for p in procs]
    r = subprocess.CompletedProcess("fastsurfer", 0, "".join(o[0] for o in outs), "".join(o[1] for o in outs))
    with open(f"{d}/ss_list.txt", "w") as fh:
        for sid, key in ch:
            if os.path.exists(f"{d}/{sid}.nii.gz"):
                fh.write(f"{d}/{sid}.nii.gz" + chr(9) + f"{d}/{sid}.ssmask.nii.gz" + chr(10))
    rs = sh(f"python /tmp/fs/ss_batch.py {d}/ss_list.txt")
    if not any(os.path.exists(f"{d}/{sid}.ssmask.nii.gz") for sid, _ in ch):   # fallback: per scan
        for sid, _ in ch:
            if os.path.exists(f"{d}/{sid}.nii.gz"):
                rs = sh(f"python /tmp/fs/mri_synthstrip -i {d}/{sid}.nii.gz -m {d}/{sid}.ssmask.nii.gz")
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
            mpath = f"{d}/{sid}.ssmask.nii.gz"
            if os.path.exists(mpath):
                mm = nib.load(mpath)
                row["icv_synthstrip"] = float((np.asanyarray(mm.dataobj) > 0).sum()) * float(np.prod(mm.header.get_zooms()[:3]))
            else:
                print(f"  synthstrip missing {sid}: {rs.stdout[-200:]} {rs.stderr[-200:]}", flush=True)
            row.update({f"V{int(l)}": float(n) * vox for l, n in zip(labs, cnt)})
            rows.append(row)
        except Exception as e:
            print(f"  ERROR {sid}: {type(e).__name__} {e}", flush=True)
    with open(OUT, "a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + chr(10))
    n_ok += len(rows)
    dt = time.time() - t1
    last_dt = dt
    print(f"chunk {c+1}/{len(chunks)}: {len(rows)}/{len(ch)} ok, {dt:.0f}s ({dt/max(len(ch),1):.1f}s/scan), "
          f"total {n_ok}/{len(JOBS)}, elapsed {(time.time()-T0)/3600:.2f} h", flush=True)
    if len(rows) == 0:
        print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
        if c == 0:
            break
print(f"finished: {n_ok} rows, {(time.time()-T0)/3600:.2f} h", flush=True)
