# IXI through FastSurfer + the USC local-brain-age (LBA) model.
#
# Two jobs in one pass over ~589 healthy adults aged 20-86 (three London sites:
# Guy's Philips 1.5T, Hammersmith Philips 3T, IoP GE 1.5T):
#   1. FastSurfer segmentation (aseg + DKT) -> regional volumes. Same software as
#      our OASIS-2 and DLBS runs, so the three pool into one healthy reference.
#   2. The Irimia-lab V-Net (PNAS 2026, public ONNX weights) -> voxel LBA map,
#      summarised per FastSurfer label, to test prediction P4 (regional "age" ranks
#      follow predictability and flip sign around the training mean age).
#
# The LBA model wants a FreeSurfer brain.mgz (conformed 256^3, WM normalised to 110,
# skull-stripped, uint8). Without recon-all we build stand-ins from FastSurfer's
# conformed image after N4 bias correction, and record LBA for three of them:
#   cap  WM mean scaled to 110, capped at 110 (brain.mgz saturates WM at 110)
#   hm   histogram-matched to the intensity distribution of the repo's sample brain.mgz
#   raw  WM mean scaled to 110, uncapped (every third scan)
# A local test on IXI002 (age 35.8) gave 58.6 / 63.7 / 59.0, so the stand-in choice
# moves LBA by a few years at most; the cohort will show whether the model is biased.
#
# FastSurfer (GPU) runs in the main loop; N4 + LBA (CPU + GPU) run in a worker thread
# on the previous scan, so the slow N4 step costs no wall time.
# Rows are appended after EVERY scan, so a timeout loses nothing.
import concurrent.futures as cf
import glob, json, os, shutil, subprocess, threading, time

PART = 1
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/ixi_lba_part{PART}.jsonl"
MAPS = "/kaggle/working/maps"
os.makedirs(MAPS, exist_ok=True)
REF_Q = None                           # filled below after numpy import
REF_Q_TXT = ("1,2,4,5,6,8,10,12,14,17,20,24,27,29,31,32,34,35,36,38,39,41,42,44,45,47,48,50,52,54,"
             "56,57,59,61,63,65,67,69,70,72,74,75,77,78,79,81,82,83,84,85,86,87,88,89,90,91,91,92,93,"
             "94,95,95,96,97,98,98,99,100,101,101,102,103,104,104,105,106,107,108,108,110,110,110,110,"
             "110,110,110,110,110,110,110,110,110,110,110,110,110,110,110,110,110,163")


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)


print("PART", PART, flush=True)
r = sh("git clone --depth 1 --branch v2.4.2 https://github.com/Deep-MI/FastSurfer.git /tmp/FastSurfer")
print("fastsurfer clone rc", r.returncode, flush=True)
# NumPy 2 removed ndarray.newbyteorder(); FastSurfer 2.4.2's image_io still calls it
r = sh("sed -i 's/data = data.byteswap().newbyteorder()/data = data.byteswap().view(data.dtype.newbyteorder())/' "
       "/tmp/FastSurfer/recon_surf/image_io.py && grep -c 'view(data.dtype.newbyteorder' /tmp/FastSurfer/recon_surf/image_io.py")
print("numpy2 patch:", r.stdout.strip(), flush=True)
r = sh("git clone --depth 1 https://github.com/irimia-laboratory/USC_LBA_estimator.git /tmp/LBA")
print("lba clone rc", r.returncode, os.path.getsize("/tmp/LBA/LBAmodel.onnx") if os.path.exists("/tmp/LBA/LBAmodel.onnx") else -1, flush=True)
r = sh("pip install -q yacs simpleitk nibabel scikit-image h5py lapy pandas onnxruntime-gpu")
print("pip rc", r.returncode, r.stderr[-300:], flush=True)

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
import onnxruntime as ort

REF_Q = np.array([float(v) for v in REF_Q_TXT.split(",")])
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", "numpy", np.__version__, flush=True)
try:
    ort.preload_dlls()          # use the CUDA/cuDNN that ships with torch
except Exception as e:
    print("preload_dlls:", type(e).__name__, e, flush=True)


def make_session(providers):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    s = ort.InferenceSession("/tmp/LBA/LBAmodel.onnx", so, providers=providers)
    s.run(None, {s.get_inputs()[0].name: np.zeros((1, 1, 128, 128, 128), np.float32)})
    return s


try:
    SESS = make_session(["CUDAExecutionProvider", "CPUExecutionProvider"])
except Exception as e:
    print("GPU session failed, CPU fallback:", type(e).__name__, str(e)[:300], flush=True)
    SESS = make_session(["CPUExecutionProvider"])
print("ORT providers", SESS.get_providers(), flush=True)
INAME = SESS.get_inputs()[0].name
A_COEF, B_COEF = 0.67661, 21.6     # the published semiglobal correction
LOCK = threading.Lock()


def lba_of(brain256):
    """Exactly the repository's preprocessing: cubic zoom 0.5, raw 0-255 floats."""
    b = ndi.zoom(brain256.astype(np.float64), (0.5, 0.5, 0.5))
    out = SESS.run(None, {INAME: b[None, None].astype(np.float32)})[0].squeeze()
    return out * A_COEF + B_COEF, b > 0.5, b


def summarise(lba, m128, seg128, tag):
    v = lba[m128]
    row = {f"{tag}_mean": float(v.mean()), f"{tag}_median": float(np.median(v)),
           f"{tag}_sd": float(v.std())}
    for q in (5, 25, 75, 95):
        row[f"{tag}_p{q}"] = float(np.percentile(v, q))
    labs = seg128[m128]
    sums = np.bincount(labs, weights=v)
    cnts = np.bincount(labs)
    for l in np.nonzero(cnts)[0]:
        if l > 0 and cnts[l] >= 5:
            row[f"{tag}_L{int(l)}"] = float(sums[l] / cnts[l])
    return row


def post(name, f, i, t_seg):
    """N4 + volumes + LBA variants for one segmented scan; runs in the worker thread."""
    t1 = time.time()
    mri = f"/tmp/out/{name}/mri"
    try:
        segp = f"{mri}/aparc.DKTatlas+aseg.deep.mgz"
        r = sh(f"cd /tmp/FastSurfer && PYTHONPATH=/tmp/FastSurfer python recon_surf/N4_bias_correct.py "
               f"--in {mri}/orig.mgz --rescale {mri}/orig_nu.mgz --aseg {segp} --threads 2")
        if not os.path.exists(f"{mri}/orig_nu.mgz"):
            print(f"  FAIL N4 {name}: {r.stdout[-300:]} {r.stderr[-600:]}", flush=True)
            return
        s = nib.load(segp)
        seg = np.asanyarray(s.dataobj).astype(np.int32)
        nu = np.asanyarray(nib.load(f"{mri}/orig_nu.mgz").dataobj).astype(np.float32)
        if os.path.exists(f"{mri}/mask.mgz"):
            mask = np.asanyarray(nib.load(f"{mri}/mask.mgz").dataobj) > 0
        else:
            mask = ndi.binary_fill_holes(ndi.binary_closing(seg > 0, iterations=3))
        assert seg.shape == (256, 256, 256) and nu.shape == seg.shape, seg.shape
        wm = np.isin(seg, (2, 41))
        gm = (seg >= 1000) | np.isin(seg, (3, 42))
        csf = np.isin(seg, (4, 43, 24))
        x = nu * (110.0 / nu[wm].mean()) * mask
        row = {"file": name, "zooms_in": [float(z) for z in nib.load(f).header.get_zooms()[:3]],
               "wm_mean_nu": float(nu[wm].mean()), "gm_over_wm": float(nu[gm].mean() / nu[wm].mean()),
               "csf_over_wm": float(np.median(nu[csf]) / nu[wm].mean()) if csf.any() else None,
               "mask_vox": int(mask.sum())}
        vox = float(np.prod(s.header.get_zooms()[:3]))
        labs, cnt = np.unique(seg[seg > 0], return_counts=True)
        row.update({f"V{int(l)}": float(c) * vox for l, c in zip(labs, cnt)})
        seg128 = ndi.zoom(seg, (0.5, 0.5, 0.5), order=0)

        lba, m128, b128 = lba_of(np.clip(x, 0, 110))
        row.update(summarise(lba, m128, seg128, "cap"))
        small = lba.reshape(64, 2, 64, 2, 64, 2).mean(axis=(1, 3, 5))
        msmall = m128.reshape(64, 2, 64, 2, 64, 2).mean(axis=(1, 3, 5)) > 0.5
        np.savez_compressed(f"{MAPS}/{name}.npz", lba=np.where(msmall, small, 0).astype(np.float16))
        if i < 6:      # keep a few model inputs for visual checks
            np.savez_compressed(f"{MAPS}/{name}_input128.npz", x=np.clip(b128, 0, 255).astype(np.uint8))
        # histogram matching to the sample brain.mgz
        v = x[mask]
        ranks = np.empty(len(v))
        ranks[np.argsort(v, kind="stable")] = np.linspace(0, 1, len(v))
        hm = np.zeros_like(x)
        hm[mask] = np.interp(ranks, np.linspace(0, 1, len(REF_Q)), REF_Q)
        lba_h, m_h, _ = lba_of(hm)
        row.update(summarise(lba_h, m_h, seg128, "hm"))
        if i % 3 == 0:
            lba_r, m_r, _ = lba_of(np.clip(x, 0, 255))
            row.update(summarise(lba_r, m_r, seg128, "raw"))
        row["seconds_seg"] = round(t_seg, 1)
        row["seconds_post"] = round(time.time() - t1, 1)
        with LOCK:
            with open(OUT, "a") as fh:
                fh.write(json.dumps(row) + chr(10))
    except Exception as e:
        print(f"  ERROR post {name}: {type(e).__name__} {e}", flush=True)
    finally:
        shutil.rmtree(f"/tmp/out/{name}", ignore_errors=True)


files = sorted(glob.glob("/kaggle/input/**/IXI*-T1.nii*", recursive=True))
print("IXI T1 files found", len(files), flush=True)
assert len(files) > 500, "wrong dataset resolve"
jobs = files[PART::2]
print("jobs for this part", len(jobs), flush=True)

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        done.add(json.loads(line)["file"])
pool = cf.ThreadPoolExecutor(1)
pending = []
for i, f in enumerate(jobs):
    name = os.path.basename(f).split(".")[0]
    if name in done:
        continue
    if time.time() - T0 > BUDGET_S:
        print("time budget reached, stopping", flush=True)
        break
    while sum(not p.done() for p in pending) >= 2:     # bound the backlog
        time.sleep(1)
    t1 = time.time()
    r = sh("cd /tmp/FastSurfer && PYTHONPATH=/tmp/FastSurfer python FastSurferCNN/run_prediction.py "
           f"--t1 {f} --sid {name} --sd /tmp/out --device cuda --viewagg_device cuda "
           "--threads 2 --vox_size 1")
    if not os.path.exists(f"/tmp/out/{name}/mri/aparc.DKTatlas+aseg.deep.mgz"):
        print(f"  FAIL seg {name}: {r.stderr[-600:]}", flush=True)
        continue
    pending.append(pool.submit(post, name, f, i, time.time() - t1))
    if i % 10 == 0 or i < 3:
        n_rows = sum(1 for _ in open(OUT)) if os.path.exists(OUT) else 0
        last = json.loads(open(OUT).readlines()[-1]) if n_rows else {}
        print(f"  [{i+1}/{len(jobs)}] {name} seg {time.time()-t1:.0f}s | rows {n_rows} | last: "
              f"cap {last.get('cap_mean', float('nan')):.1f} hm {last.get('hm_mean', float('nan')):.1f} "
              f"sd {last.get('cap_sd', float('nan')):.2f} post {last.get('seconds_post', 0)}s | "
              f"elapsed {(time.time()-T0)/3600:.2f} h", flush=True)
pool.shutdown(wait=True)
n_rows = sum(1 for _ in open(OUT)) if os.path.exists(OUT) else 0
print(f"finished: {n_rows} rows, {(time.time()-T0)/3600:.2f} h", flush=True)
