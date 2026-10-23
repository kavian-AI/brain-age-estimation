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

PART = 20
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["ds007522__sub-HML0860__60", "ds007522/sub-HML0860/ses-01/anat/sub-HML0860_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0942__61", "ds007522/sub-HML0942/ses-01/anat/sub-HML0942_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0962__62", "ds007522/sub-HML0962/ses-01/anat/sub-HML0962_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1080__63", "ds007522/sub-HML1080/ses-01/anat/sub-HML1080_ses-01_rec-defaced_T1w.nii.gz"], ["ds003592__sub-124__71", "ds003592/sub-124/ses-1/anat/sub-124_ses-1_T1w.nii.gz"], ["ds003592__sub-292__72", "ds003592/sub-292/ses-1/anat/sub-292_ses-1_T1w.nii.gz"], ["ds003592__sub-296__73", "ds003592/sub-296/ses-1/anat/sub-296_ses-1_T1w.nii.gz"], ["ds005270__sub-014__74", "ds005270/sub-014/anat/sub-014_T1w.nii.gz"], ["ds005270__sub-113__75", "ds005270/sub-113/anat/sub-113_T1w.nii.gz"], ["ds005405__sub-107__76", "ds005405/sub-107/ses-bhb/anat/sub-107_ses-bhb_run-1_T1w.nii"], ["ds004725__sub-21020__77", "ds004725/sub-21020/anat/sub-21020_T1w.nii"], ["ds007831__sub-LGQ__78", "ds007831/sub-LGQ/anat/sub-LGQ_T1w.nii.gz"], ["ds007831__sub-YFE__79", "ds007831/sub-YFE/anat/sub-YFE_T1w.nii.gz"], ["ds007522__sub-HML0165__80", "ds007522/sub-HML0165/ses-01/anat/sub-HML0165_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0277__81", "ds007522/sub-HML0277/ses-01/anat/sub-HML0277_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0362__82", "ds007522/sub-HML0362/ses-01/anat/sub-HML0362_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0390__83", "ds007522/sub-HML0390/ses-01/anat/sub-HML0390_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0439__84", "ds007522/sub-HML0439/ses-01/anat/sub-HML0439_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0521__85", "ds007522/sub-HML0521/ses-01/anat/sub-HML0521_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0646__86", "ds007522/sub-HML0646/ses-01/anat/sub-HML0646_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0682__87", "ds007522/sub-HML0682/ses-01/anat/sub-HML0682_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0726__88", "ds007522/sub-HML0726/ses-01/anat/sub-HML0726_ses-01_rec-defaced_T1w.nii.gz"], ["sald__sub-031344", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031344/anat/sub-031344_T1w.nii.gz"], ["sald__sub-031504", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031504/anat/sub-031504_T1w.nii.gz"], ["sald__sub-031526", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031526/anat/sub-031526_T1w.nii.gz"], ["sald__sub-031558", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031558/anat/sub-031558_T1w.nii.gz"], ["sald__sub-031638", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031638/anat/sub-031638_T1w.nii.gz"], ["sald__sub-031700", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031700/anat/sub-031700_T1w.nii.gz"], ["oasis1__OAS1_0058_MR1__mpr-1", "oasis1:disc2:OAS1_0058_MR1:mpr-1"], ["oasis1__OAS1_0262_MR1__mpr-1", "oasis1:disc7:OAS1_0262_MR1:mpr-1"], ["oasis1__OAS1_0367_MR1__mpr-1", "oasis1:disc10:OAS1_0367_MR1:mpr-1"], ["oasis1__OAS1_0375_MR1__mpr-1", "oasis1:disc10:OAS1_0375_MR1:mpr-1"], ["ds004711__sub-120__1845", "ds004711/sub-120/anat/sub-120_T1w.nii.gz"], ["ds004711__sub-052__1846", "ds004711/sub-052/anat/sub-052_T1w.nii.gz"], ["ds007831__sub-PWJ__1847", "ds007831/sub-PWJ/anat/sub-PWJ_T1w.nii.gz"], ["ds007831__sub-QTR__1848", "ds007831/sub-QTR/anat/sub-QTR_T1w.nii.gz"], ["ds007831__sub-PQX__1849", "ds007831/sub-PQX/anat/sub-PQX_T1w.nii.gz"], ["ds007831__sub-VRT__1850", "ds007831/sub-VRT/anat/sub-VRT_T1w.nii.gz"], ["ds004711__sub-015__1851", "ds004711/sub-015/anat/sub-015_T1w.nii.gz"], ["ds004711__sub-046__1852", "ds004711/sub-046/anat/sub-046_T1w.nii.gz"], ["ds004711__sub-058__1853", "ds004711/sub-058/anat/sub-058_T1w.nii.gz"], ["ds005752__sub-ON23776__1854", "ds005752/sub-ON23776/ses-01/anat/sub-ON23776_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON96555__1855", "ds005752/sub-ON96555/ses-01/anat/sub-ON96555_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005270__sub-064__1856", "ds005270/sub-064/anat/sub-064_T1w.nii.gz"], ["ds005405__sub-053__1857", "ds005405/sub-053/ses-bhb/anat/sub-053_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-063__1858", "ds005405/sub-063/ses-bhb/anat/sub-063_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-135__1859", "ds005405/sub-135/ses-bhb/anat/sub-135_ses-bhb_run-1_T1w.nii"], ["ds006395__sub-C01750__1860", "ds006395/sub-C01750/anat/sub-C01750_T1w.nii.gz"], ["obhb__401672562533", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-401672562533/ses-1/sub-401672562533_T1w.nii.gz"], ["sald__sub-031291", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031291/anat/sub-031291_T1w.nii.gz"], ["sald__sub-031293", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031293/anat/sub-031293_T1w.nii.gz"], ["sald__sub-031296", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031296/anat/sub-031296_T1w.nii.gz"], ["sald__sub-031349", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031349/anat/sub-031349_T1w.nii.gz"], ["sald__sub-031432", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031432/anat/sub-031432_T1w.nii.gz"], ["sald__sub-031480", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031480/anat/sub-031480_T1w.nii.gz"], ["sald__sub-031495", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031495/anat/sub-031495_T1w.nii.gz"], ["sald__sub-031528", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031528/anat/sub-031528_T1w.nii.gz"], ["sald__sub-031671", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031671/anat/sub-031671_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
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
