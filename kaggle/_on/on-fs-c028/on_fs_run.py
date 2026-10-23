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

PART = 28
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["obhb__245306557719", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-245306557719/ses-1/sub-245306557719_T1w.nii.gz"], ["obhb__275841492998", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-275841492998/ses-1/sub-275841492998_T1w.nii.gz"], ["obhb__299666419729", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-299666419729/ses-1/sub-299666419729_T1w.nii.gz"], ["obhb__347451361275", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-347451361275/ses-1/sub-347451361275_T1w.nii.gz"], ["obhb__399014895585", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-399014895585/ses-1/sub-399014895585_T1w.nii.gz"], ["obhb__686082388541", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-686082388541/ses-1/sub-686082388541_T1w.nii.gz"], ["obhb__777893483077", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-777893483077/ses-1/sub-777893483077_T1w.nii.gz"], ["obhb__809580291106", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-809580291106/ses-1/sub-809580291106_T1w.nii.gz"], ["obhb__823640829061", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-823640829061/ses-1/sub-823640829061_T1w.nii.gz"], ["obhb__826455898290", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-826455898290/ses-1/sub-826455898290_T1w.nii.gz"], ["obhb__917842166945", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-917842166945/ses-1/sub-917842166945_T1w.nii.gz"], ["obhb__237746771571", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-237746771571/ses-1/sub-237746771571_T1w.nii.gz"], ["obhb__792041257894", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-792041257894/ses-1/sub-792041257894_T1w.nii.gz"], ["sald__sub-031437", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031437/anat/sub-031437_T1w.nii.gz"], ["sald__sub-031485", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031485/anat/sub-031485_T1w.nii.gz"], ["sald__sub-031518", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031518/anat/sub-031518_T1w.nii.gz"], ["sald__sub-031519", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031519/anat/sub-031519_T1w.nii.gz"], ["sald__sub-031520", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031520/anat/sub-031520_T1w.nii.gz"], ["sald__sub-031529", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031529/anat/sub-031529_T1w.nii.gz"], ["sald__sub-031533", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031533/anat/sub-031533_T1w.nii.gz"], ["sald__sub-031729", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031729/anat/sub-031729_T1w.nii.gz"], ["oasis1__OAS1_0012_MR1__mpr-1", "oasis1:disc1:OAS1_0012_MR1:mpr-1"], ["oasis1__OAS1_0127_MR1__mpr-1", "oasis1:disc4:OAS1_0127_MR1:mpr-1"], ["oasis1__OAS1_0417_MR1__mpr-1", "oasis1:disc11:OAS1_0417_MR1:mpr-1"], ["oasis1__OAS1_0444_MR1__mpr-1", "oasis1:disc12:OAS1_0444_MR1:mpr-1"], ["obhb__671852511180", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-671852511180/ses-1/sub-671852511180_T1w.nii.gz"], ["obhb__981966417728", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-981966417728/ses-1/sub-981966417728_T1w.nii.gz"], ["ds004711__sub-018__2121", "ds004711/sub-018/anat/sub-018_T1w.nii.gz"], ["obhb__573188297121", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-573188297121/ses-1/sub-573188297121_T1w.nii.gz"], ["obhb__389237925618", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-389237925618/ses-1/sub-389237925618_T1w.nii.gz"], ["obhb__871409555490", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-871409555490/ses-1/sub-871409555490_T1w.nii.gz"], ["ds004711__sub-069__2122", "ds004711/sub-069/anat/sub-069_T1w.nii.gz"], ["obhb__619612744941", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-619612744941/ses-1/sub-619612744941_T1w.nii.gz"], ["obhb__219201052528", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-219201052528/ses-1/sub-219201052528_T1w.nii.gz"], ["ds005752__sub-ON05311__2123", "ds005752/sub-ON05311/ses-01/anat/sub-ON05311_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON05530__2124", "ds005752/sub-ON05530/ses-01/anat/sub-ON05530_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON09474__2125", "ds005752/sub-ON09474/ses-01/anat/sub-ON09474_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON12666__2126", "ds005752/sub-ON12666/ses-01/anat/sub-ON12666_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON23664__2127", "ds005752/sub-ON23664/ses-01/anat/sub-ON23664_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON30080__2128", "ds005752/sub-ON30080/ses-01/anat/sub-ON30080_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON71932__2129", "ds005752/sub-ON71932/ses-01/anat/sub-ON71932_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON80038__2130", "ds005752/sub-ON80038/ses-01/anat/sub-ON80038_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON96440__2131", "ds005752/sub-ON96440/ses-01/anat/sub-ON96440_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds003592__sub-159__2132", "ds003592/sub-159/ses-1/anat/sub-159_ses-1_T1w.nii.gz"], ["ds003592__sub-257__2133", "ds003592/sub-257/ses-1/anat/sub-257_ses-1_T1w.nii.gz"], ["ds003592__sub-258__2134", "ds003592/sub-258/ses-1/anat/sub-258_ses-1_T1w.nii.gz"], ["ds003592__sub-259__2135", "ds003592/sub-259/ses-1/anat/sub-259_ses-1_T1w.nii.gz"], ["ds003592__sub-74__2136", "ds003592/sub-74/ses-1/anat/sub-74_ses-1_T1w.nii.gz"], ["ds005123__sub-10608__2137", "ds005123/sub-10608/anat/sub-10608_T1w.nii.gz"], ["ds005405__sub-094__2138", "ds005405/sub-094/ses-bhb/anat/sub-094_ses-bhb_run-1_T1w.nii"], ["obhb__133292484570", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-133292484570/ses-1/sub-133292484570_T1w.nii.gz"], ["obhb__144658033075", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-144658033075/ses-1/sub-144658033075_T1w.nii.gz"], ["obhb__150529102662", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-150529102662/ses-1/sub-150529102662_T1w.nii.gz"], ["obhb__183469803882", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-183469803882/ses-1/sub-183469803882_T1w.nii.gz"], ["obhb__206099391777", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-206099391777/ses-1/sub-206099391777_T1w.nii.gz"], ["obhb__216691637184", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-216691637184/ses-1/sub-216691637184_T1w.nii.gz"], ["obhb__251630004450", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-251630004450/ses-1/sub-251630004450_T1w.nii.gz"], ["obhb__255360929670", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-255360929670/ses-1/sub-255360929670_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
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
