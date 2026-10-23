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

PART = 19
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["ds007522__sub-HML0787__134", "ds007522/sub-HML0787/ses-01/anat/sub-HML0787_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0822__135", "ds007522/sub-HML0822/ses-01/anat/sub-HML0822_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0832__136", "ds007522/sub-HML0832/ses-01/anat/sub-HML0832_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0905__137", "ds007522/sub-HML0905/ses-01/anat/sub-HML0905_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1082__138", "ds007522/sub-HML1082/ses-02/anat/sub-HML1082_ses-02_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1147__139", "ds007522/sub-HML1147/ses-01/anat/sub-HML1147_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1170__140", "ds007522/sub-HML1170/ses-01/anat/sub-HML1170_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1182__141", "ds007522/sub-HML1182/ses-01/anat/sub-HML1182_ses-01_rec-defaced_T1w.nii.gz"], ["ds003592__sub-03__148", "ds003592/sub-03/ses-1/anat/sub-03_ses-1_T1w.nii.gz"], ["ds003592__sub-269__149", "ds003592/sub-269/ses-1/anat/sub-269_ses-1_T1w.nii.gz"], ["ds003592__sub-272__150", "ds003592/sub-272/ses-1/anat/sub-272_ses-1_T1w.nii.gz"], ["ds003592__sub-53__151", "ds003592/sub-53/ses-1/anat/sub-53_ses-1_T1w.nii.gz"], ["ds005270__sub-026__152", "ds005270/sub-026/anat/sub-026_T1w.nii.gz"], ["ds005270__sub-046__153", "ds005270/sub-046/anat/sub-046_T1w.nii.gz"], ["ds005405__sub-076__154", "ds005405/sub-076/ses-bhb/anat/sub-076_ses-bhb_run-1_T1w.nii"], ["ds004725__sub-22036__155", "ds004725/sub-22036/anat/sub-22036_T1w.nii"], ["ds007831__sub-DZG__156", "ds007831/sub-DZG/anat/sub-DZG_T1w.nii.gz"], ["ds004711__sub-179__157", "ds004711/sub-179/anat/sub-179_T1w.nii.gz"], ["ds007831__sub-JFV__158", "ds007831/sub-JFV/anat/sub-JFV_T1w.nii.gz"], ["ds004711__sub-184__159", "ds004711/sub-184/anat/sub-184_T1w.nii.gz"], ["ds007831__sub-SEJ__160", "ds007831/sub-SEJ/anat/sub-SEJ_T1w.nii.gz"], ["ds004725__sub-22051__161", "ds004725/sub-22051/anat/sub-22051_T1w.nii"], ["ds004711__sub-050__1826", "ds004711/sub-050/anat/sub-050_T1w.nii.gz"], ["ds005752__sub-ON59735__1827", "ds005752/sub-ON59735/ses-01/anat/sub-ON59735_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON63221__1828", "ds005752/sub-ON63221/ses-01/anat/sub-ON63221_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005270__sub-016__1829", "ds005270/sub-016/anat/sub-016_T1w.nii.gz"], ["ds005270__sub-066__1830", "ds005270/sub-066/anat/sub-066_T1w.nii.gz"], ["ds005405__sub-116__1831", "ds005405/sub-116/ses-bhb/anat/sub-116_ses-bhb_run-1_T1w.nii"], ["ds006395__sub-C01741__1832", "ds006395/sub-C01741/anat/sub-C01741_T1w.nii.gz"], ["obhb__841148850958", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-841148850958/ses-1/sub-841148850958_T1w.nii.gz"], ["obhb__886465425609", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-886465425609/ses-1/sub-886465425609_T1w.nii.gz"], ["sald__sub-031377", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031377/anat/sub-031377_T1w.nii.gz"], ["sald__sub-031491", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031491/anat/sub-031491_T1w.nii.gz"], ["sald__sub-031525", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031525/anat/sub-031525_T1w.nii.gz"], ["sald__sub-031530", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031530/anat/sub-031530_T1w.nii.gz"], ["sald__sub-031536", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031536/anat/sub-031536_T1w.nii.gz"], ["sald__sub-031549", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031549/anat/sub-031549_T1w.nii.gz"], ["sald__sub-031627", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031627/anat/sub-031627_T1w.nii.gz"], ["oasis1__OAS1_0044_MR1__mpr-1", "oasis1:disc2:OAS1_0044_MR1:mpr-1"], ["oasis1__OAS1_0086_MR1__mpr-1", "oasis1:disc3:OAS1_0086_MR1:mpr-1"], ["oasis1__OAS1_0096_MR1__mpr-1", "oasis1:disc3:OAS1_0096_MR1:mpr-1"], ["ds007831__sub-WKP__1833", "ds007831/sub-WKP/anat/sub-WKP_T1w.nii.gz"], ["obhb__657669884728", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-657669884728/ses-1/sub-657669884728_T1w.nii.gz"], ["ds004711__sub-109__1834", "ds004711/sub-109/anat/sub-109_T1w.nii.gz"], ["ds007831__sub-CQE__1835", "ds007831/sub-CQE/anat/sub-CQE_T1w.nii.gz"], ["ds007831__sub-NHL__1836", "ds007831/sub-NHL/anat/sub-NHL_T1w.nii.gz"], ["ds005752__sub-ON08710__1837", "ds005752/sub-ON08710/ses-01/anat/sub-ON08710_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON75100__1838", "ds005752/sub-ON75100/ses-01/anat/sub-ON75100_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON96353__1839", "ds005752/sub-ON96353/ses-01/anat/sub-ON96353_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005270__sub-106__1840", "ds005270/sub-106/anat/sub-106_T1w.nii.gz"], ["ds005123__sub-10854__1841", "ds005123/sub-10854/anat/sub-10854_T1w.nii.gz"], ["ds005123__sub-10954__1842", "ds005123/sub-10954/anat/sub-10954_T1w.nii.gz"], ["ds005405__sub-100__1843", "ds005405/sub-100/ses-bhb/anat/sub-100_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-104__1844", "ds005405/sub-104/ses-bhb/anat/sub-104_ses-bhb_run-1_T1w.nii"], ["obhb__196146731599", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-196146731599/ses-1/sub-196146731599_T1w.nii.gz"], ["obhb__561752414596", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-561752414596/ses-1/sub-561752414596_T1w.nii.gz"], ["obhb__715730829823", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-715730829823/ses-1/sub-715730829823_T1w.nii.gz"], ["obhb__894511350563", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-894511350563/ses-1/sub-894511350563_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
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
