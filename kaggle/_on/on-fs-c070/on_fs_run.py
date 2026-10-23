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

PART = 70
T0 = time.time()
BUDGET_S = 11.0 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["ds007522__sub-HML0789__722", "ds007522/sub-HML0789/ses-01/anat/sub-HML0789_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0791__723", "ds007522/sub-HML0791/ses-01/anat/sub-HML0791_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0820__724", "ds007522/sub-HML0820/ses-01/anat/sub-HML0820_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0866__725", "ds007522/sub-HML0866/ses-01/anat/sub-HML0866_ses-01_rec-defaced_T1w.nii.gz"], ["sald__sub-031466", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031466/anat/sub-031466_T1w.nii.gz"], ["sald__sub-031471", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031471/anat/sub-031471_T1w.nii.gz"], ["sald__sub-031472", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031472/anat/sub-031472_T1w.nii.gz"], ["sald__sub-031476", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031476/anat/sub-031476_T1w.nii.gz"], ["sald__sub-031527", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031527/anat/sub-031527_T1w.nii.gz"], ["sald__sub-031551", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031551/anat/sub-031551_T1w.nii.gz"], ["sald__sub-031554", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031554/anat/sub-031554_T1w.nii.gz"], ["sald__sub-031564", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031564/anat/sub-031564_T1w.nii.gz"], ["sald__sub-031567", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031567/anat/sub-031567_T1w.nii.gz"], ["sald__sub-031571", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031571/anat/sub-031571_T1w.nii.gz"], ["sald__sub-031574", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031574/anat/sub-031574_T1w.nii.gz"], ["sald__sub-031580", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031580/anat/sub-031580_T1w.nii.gz"], ["sald__sub-031590", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031590/anat/sub-031590_T1w.nii.gz"], ["sald__sub-031601", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031601/anat/sub-031601_T1w.nii.gz"], ["sald__sub-031602", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031602/anat/sub-031602_T1w.nii.gz"], ["sald__sub-031603", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031603/anat/sub-031603_T1w.nii.gz"], ["sald__sub-031604", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031604/anat/sub-031604_T1w.nii.gz"], ["sald__sub-031605", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031605/anat/sub-031605_T1w.nii.gz"], ["sald__sub-031606", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031606/anat/sub-031606_T1w.nii.gz"], ["sald__sub-031612", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031612/anat/sub-031612_T1w.nii.gz"], ["sald__sub-031615", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031615/anat/sub-031615_T1w.nii.gz"], ["sald__sub-031617", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031617/anat/sub-031617_T1w.nii.gz"], ["sald__sub-031618", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031618/anat/sub-031618_T1w.nii.gz"], ["sald__sub-031619", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031619/anat/sub-031619_T1w.nii.gz"], ["sald__sub-031620", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031620/anat/sub-031620_T1w.nii.gz"], ["sald__sub-031624", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031624/anat/sub-031624_T1w.nii.gz"], ["sald__sub-031630", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031630/anat/sub-031630_T1w.nii.gz"], ["sald__sub-031737", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031737/anat/sub-031737_T1w.nii.gz"], ["sald__sub-031742", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031742/anat/sub-031742_T1w.nii.gz"], ["sald__sub-031748", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031748/anat/sub-031748_T1w.nii.gz"], ["sald__sub-031749", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031749/anat/sub-031749_T1w.nii.gz"], ["sald__sub-031757", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031757/anat/sub-031757_T1w.nii.gz"], ["sald__sub-031760", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031760/anat/sub-031760_T1w.nii.gz"], ["sald__sub-031763", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031763/anat/sub-031763_T1w.nii.gz"], ["oasis1__OAS1_0038_MR1__mpr-1", "oasis1:disc1:OAS1_0038_MR1:mpr-1"], ["oasis1__OAS1_0097_MR1__mpr-1", "oasis1:disc3:OAS1_0097_MR1:mpr-1"]]''')          # [[scan_id, s3_key], ...] for this part
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
            # axis0 posterior->anterior, axis1 inferior->superior, axis2 sagittal slices (1.25 mm, L->R)
            aff = np.array([[0, 0, 1.25, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
            aff[:3, 3] = -aff[:3, :3] @ (np.array(arr.shape) / 2.0)
            nib.save(nib.Nifti1Image(arr, aff), f"{d}/{sid}.nii.gz")
            json.dump({"Manufacturer": "Siemens", "ManufacturersModelName": "Vision", "MagneticFieldStrength": 1.5,
                       "o1_orient": "ASR-v1"}, open(f"{d}/{sid}.meta.json", "w"))
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
