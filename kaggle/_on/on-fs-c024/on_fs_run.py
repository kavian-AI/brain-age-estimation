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

PART = 24
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["ds005270__sub-061__1915", "ds005270/sub-061/anat/sub-061_T1w.nii.gz"], ["ds005270__sub-077__1916", "ds005270/sub-077/anat/sub-077_T1w.nii.gz"], ["ds005123__sub-10834__1917", "ds005123/sub-10834/anat/sub-10834_T1w.nii.gz"], ["ds005405__sub-057__1918", "ds005405/sub-057/ses-bhb/anat/sub-057_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-114__1919", "ds005405/sub-114/ses-bhb/anat/sub-114_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-115__1920", "ds005405/sub-115/ses-bhb/anat/sub-115_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-120__1921", "ds005405/sub-120/ses-bhb/anat/sub-120_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-126__1922", "ds005405/sub-126/ses-bhb/anat/sub-126_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-141__1923", "ds005405/sub-141/ses-bhb/anat/sub-141_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-149__1924", "ds005405/sub-149/ses-bhb/anat/sub-149_ses-bhb_run-1_T1w.nii"], ["ds006395__sub-C01713__1925", "ds006395/sub-C01713/anat/sub-C01713_T1w.nii.gz"], ["obhb__619614013878", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-619614013878/ses-1/sub-619614013878_T1w.nii.gz"], ["obhb__725821040249", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-725821040249/ses-1/sub-725821040249_T1w.nii.gz"], ["obhb__820231231679", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-820231231679/ses-1/sub-820231231679_T1w.nii.gz"], ["obhb__395384646781", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-395384646781/ses-1/sub-395384646781_T1w.nii.gz"], ["sald__sub-031545", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031545/anat/sub-031545_T1w.nii.gz"], ["sald__sub-031557", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031557/anat/sub-031557_T1w.nii.gz"], ["sald__sub-031561", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031561/anat/sub-031561_T1w.nii.gz"], ["sald__sub-031655", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031655/anat/sub-031655_T1w.nii.gz"], ["sald__sub-031668", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031668/anat/sub-031668_T1w.nii.gz"], ["sald__sub-031696", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031696/anat/sub-031696_T1w.nii.gz"], ["oasis1__OAS1_0088_MR1__mpr-1", "oasis1:disc3:OAS1_0088_MR1:mpr-1"], ["oasis1__OAS1_0159_MR1__mpr-1", "oasis1:disc5:OAS1_0159_MR1:mpr-1"], ["sald__sub-031555", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031555/anat/sub-031555_T1w.nii.gz"], ["sald__sub-031598", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031598/anat/sub-031598_T1w.nii.gz"], ["sald__sub-031665", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031665/anat/sub-031665_T1w.nii.gz"], ["oasis1__OAS1_0018_MR1__mpr-1", "oasis1:disc1:OAS1_0018_MR1:mpr-1"], ["sald__sub-031375", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031375/anat/sub-031375_T1w.nii.gz"], ["sald__sub-031546", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031546/anat/sub-031546_T1w.nii.gz"], ["sald__sub-031547", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031547/anat/sub-031547_T1w.nii.gz"], ["sald__sub-031634", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031634/anat/sub-031634_T1w.nii.gz"], ["oasis1__OAS1_0040_MR1__mpr-1", "oasis1:disc1:OAS1_0040_MR1:mpr-1"], ["oasis1__OAS1_0384_MR1__mpr-1", "oasis1:disc11:OAS1_0384_MR1:mpr-1"], ["obhb__782755729398", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-782755729398/ses-1/sub-782755729398_T1w.nii.gz"], ["obhb__994666209803", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-994666209803/ses-1/sub-994666209803_T1w.nii.gz"], ["obhb__205481948499", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-205481948499/ses-1/sub-205481948499_T1w.nii.gz"], ["ds007831__sub-TML__1951", "ds007831/sub-TML/anat/sub-TML_T1w.nii.gz"], ["ds005752__sub-ON22662__1952", "ds005752/sub-ON22662/ses-01/anat/sub-ON22662_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON48925__1953", "ds005752/sub-ON48925/ses-01/anat/sub-ON48925_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON75450__1954", "ds005752/sub-ON75450/ses-01/anat/sub-ON75450_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON84201__1955", "ds005752/sub-ON84201/ses-01/anat/sub-ON84201_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON87092__1956", "ds005752/sub-ON87092/ses-01/anat/sub-ON87092_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON97654__1957", "ds005752/sub-ON97654/ses-01/anat/sub-ON97654_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON98502__1958", "ds005752/sub-ON98502/ses-01/anat/sub-ON98502_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON98602__1959", "ds005752/sub-ON98602/ses-01/anat/sub-ON98602_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON99943__1960", "ds005752/sub-ON99943/ses-01/anat/sub-ON99943_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005270__sub-027__1961", "ds005270/sub-027/anat/sub-027_T1w.nii.gz"], ["ds005123__sub-10668__1962", "ds005123/sub-10668/anat/sub-10668_T1w.nii.gz"], ["ds005123__sub-10741__1963", "ds005123/sub-10741/anat/sub-10741_T1w.nii.gz"], ["ds005123__sub-10781__1964", "ds005123/sub-10781/anat/sub-10781_T1w.nii.gz"], ["ds005123__sub-10836__1965", "ds005123/sub-10836/anat/sub-10836_T1w.nii.gz"], ["ds005405__sub-047__1966", "ds005405/sub-047/ses-bhb/anat/sub-047_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-106__1967", "ds005405/sub-106/ses-bhb/anat/sub-106_ses-bhb_run-1_T1w.nii"], ["ds006395__sub-C01783__1968", "ds006395/sub-C01783/anat/sub-C01783_T1w.nii.gz"], ["ds006395__sub-C01786__1969", "ds006395/sub-C01786/anat/sub-C01786_T1w.nii.gz"], ["sald__sub-031522", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031522/anat/sub-031522_T1w.nii.gz"], ["sald__sub-031578", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031578/anat/sub-031578_T1w.nii.gz"], ["sald__sub-031673", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031673/anat/sub-031673_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
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
