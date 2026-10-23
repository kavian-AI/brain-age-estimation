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

PART = 25
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["sald__sub-031674", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031674/anat/sub-031674_T1w.nii.gz"], ["oasis1__OAS1_0235_MR1__mpr-1", "oasis1:disc7:OAS1_0235_MR1:mpr-1"], ["obhb__927033363456", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-927033363456/ses-1/sub-927033363456_T1w.nii.gz"], ["sald__sub-031647", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031647/anat/sub-031647_T1w.nii.gz"], ["ds004711__sub-186__1985", "ds004711/sub-186/anat/sub-186_T1w.nii.gz"], ["ds004711__sub-005__1986", "ds004711/sub-005/anat/sub-005_T1w.nii.gz"], ["ds004711__sub-126__1987", "ds004711/sub-126/anat/sub-126_T1w.nii.gz"], ["obhb__610598919254", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-610598919254/ses-1/sub-610598919254_T1w.nii.gz"], ["ds004711__sub-065__1988", "ds004711/sub-065/anat/sub-065_T1w.nii.gz"], ["ds004711__sub-185__1989", "ds004711/sub-185/anat/sub-185_T1w.nii.gz"], ["ds005752__sub-ON08392__1990", "ds005752/sub-ON08392/ses-01/anat/sub-ON08392_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON28693__1991", "ds005752/sub-ON28693/ses-01/anat/sub-ON28693_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON35773__1992", "ds005752/sub-ON35773/ses-01/anat/sub-ON35773_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON66199__1993", "ds005752/sub-ON66199/ses-01/anat/sub-ON66199_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON87642__1994", "ds005752/sub-ON87642/ses-01/anat/sub-ON87642_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON87777__1995", "ds005752/sub-ON87777/ses-01/anat/sub-ON87777_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON98130__1996", "ds005752/sub-ON98130/ses-01/anat/sub-ON98130_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON98806__1997", "ds005752/sub-ON98806/ses-01/anat/sub-ON98806_ses-01_acq-SagittalMPRAGE_T1w.nii.gz"], ["ds005270__sub-148__1998", "ds005270/sub-148/anat/sub-148_T1w.nii.gz"], ["ds005123__sub-10887__1999", "ds005123/sub-10887/anat/sub-10887_T1w.nii.gz"], ["ds005123__sub-10926__2000", "ds005123/sub-10926/anat/sub-10926_T1w.nii.gz"], ["ds005405__sub-006__2001", "ds005405/sub-006/ses-bhb/anat/sub-006_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-013__2002", "ds005405/sub-013/ses-bhb/anat/sub-013_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-015__2003", "ds005405/sub-015/ses-bhb/anat/sub-015_ses-bhb_run-1_T1w.nii"], ["ds006395__sub-C01765__2004", "ds006395/sub-C01765/anat/sub-C01765_T1w.nii.gz"], ["ds006395__sub-C01854__2005", "ds006395/sub-C01854/anat/sub-C01854_T1w.nii.gz"], ["obhb__188694833436", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-188694833436/ses-1/sub-188694833436_T1w.nii.gz"], ["obhb__403901799268", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-403901799268/ses-1/sub-403901799268_T1w.nii.gz"], ["obhb__789333583005", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-789333583005/ses-1/sub-789333583005_T1w.nii.gz"], ["obhb__838089558550", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-838089558550/ses-1/sub-838089558550_T1w.nii.gz"], ["obhb__669618721362", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-669618721362/ses-1/sub-669618721362_T1w.nii.gz"], ["obhb__824381268107", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-824381268107/ses-1/sub-824381268107_T1w.nii.gz"], ["obhb__847931561162", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-847931561162/ses-1/sub-847931561162_T1w.nii.gz"], ["sald__sub-031493", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031493/anat/sub-031493_T1w.nii.gz"], ["sald__sub-031583", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031583/anat/sub-031583_T1w.nii.gz"], ["sald__sub-031644", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031644/anat/sub-031644_T1w.nii.gz"], ["sald__sub-031678", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031678/anat/sub-031678_T1w.nii.gz"], ["sald__sub-031692", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031692/anat/sub-031692_T1w.nii.gz"], ["oasis1__OAS1_0173_MR1__mpr-1", "oasis1:disc5:OAS1_0173_MR1:mpr-1"], ["ds007831__sub-CHR__2006", "ds007831/sub-CHR/anat/sub-CHR_T1w.nii.gz"], ["obhb__511979172515", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-511979172515/ses-1/sub-511979172515_T1w.nii.gz"], ["obhb__119019619320", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-119019619320/ses-1/sub-119019619320_T1w.nii.gz"], ["ds004711__sub-132__2007", "ds004711/sub-132/anat/sub-132_T1w.nii.gz"], ["obhb__480168790227", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-480168790227/ses-1/sub-480168790227_T1w.nii.gz"], ["obhb__805388577387", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-805388577387/ses-1/sub-805388577387_T1w.nii.gz"], ["obhb__939910199671", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-939910199671/ses-1/sub-939910199671_T1w.nii.gz"], ["sald__sub-031457", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031457/anat/sub-031457_T1w.nii.gz"], ["sald__sub-031651", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031651/anat/sub-031651_T1w.nii.gz"], ["sald__sub-031653", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031653/anat/sub-031653_T1w.nii.gz"], ["sald__sub-031654", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031654/anat/sub-031654_T1w.nii.gz"], ["sald__sub-031656", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031656/anat/sub-031656_T1w.nii.gz"], ["sald__sub-031657", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031657/anat/sub-031657_T1w.nii.gz"], ["sald__sub-031688", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031688/anat/sub-031688_T1w.nii.gz"], ["sald__sub-031693", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031693/anat/sub-031693_T1w.nii.gz"], ["sald__sub-031752", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031752/anat/sub-031752_T1w.nii.gz"], ["oasis1__OAS1_0145_MR1__mpr-1", "oasis1:disc4:OAS1_0145_MR1:mpr-1"], ["oasis1__OAS1_0409_MR1__mpr-1", "oasis1:disc11:OAS1_0409_MR1:mpr-1"], ["ds004711__sub-079__2027", "ds004711/sub-079/anat/sub-079_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
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
