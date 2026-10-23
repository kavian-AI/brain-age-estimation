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

PART = 26
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["ds004711__sub-100__2028", "ds004711/sub-100/anat/sub-100_T1w.nii.gz"], ["obhb__804202082081", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-804202082081/ses-1/sub-804202082081_T1w.nii.gz"], ["ds004711__sub-068__2029", "ds004711/sub-068/anat/sub-068_T1w.nii.gz"], ["ds004711__sub-075__2030", "ds004711/sub-075/anat/sub-075_T1w.nii.gz"], ["obhb__976182490642", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-976182490642/ses-1/sub-976182490642_T1w.nii.gz"], ["obhb__516034188736", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-516034188736/ses-1/sub-516034188736_T1w.nii.gz"], ["ds004711__sub-151__2031", "ds004711/sub-151/anat/sub-151_T1w.nii.gz"], ["obhb__686436805646", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-686436805646/ses-1/sub-686436805646_T1w.nii.gz"], ["ds005752__sub-ON05258__2032", "ds005752/sub-ON05258/ses-01/anat/sub-ON05258_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds005752__sub-ON52662__2033", "ds005752/sub-ON52662/ses-01/anat/sub-ON52662_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON58053__2034", "ds005752/sub-ON58053/ses-01/anat/sub-ON58053_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON93503__2035", "ds005752/sub-ON93503/ses-01/anat/sub-ON93503_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON96530__2036", "ds005752/sub-ON96530/ses-01/anat/sub-ON96530_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON97252__2037", "ds005752/sub-ON97252/ses-01/anat/sub-ON97252_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON97765__2038", "ds005752/sub-ON97765/ses-01/anat/sub-ON97765_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005270__sub-092__2039", "ds005270/sub-092/anat/sub-092_T1w.nii.gz"], ["ds005270__sub-107__2040", "ds005270/sub-107/anat/sub-107_T1w.nii.gz"], ["ds005270__sub-153__2041", "ds005270/sub-153/anat/sub-153_T1w.nii.gz"], ["ds005123__sub-10674__2042", "ds005123/sub-10674/anat/sub-10674_T1w.nii.gz"], ["ds005123__sub-10677__2043", "ds005123/sub-10677/anat/sub-10677_T1w.nii.gz"], ["ds005123__sub-10720__2044", "ds005123/sub-10720/anat/sub-10720_T1w.nii.gz"], ["ds005123__sub-10748__2045", "ds005123/sub-10748/anat/sub-10748_T1w.nii.gz"], ["ds005405__sub-127__2046", "ds005405/sub-127/ses-bhb/anat/sub-127_ses-bhb_run-1_T1w.nii"], ["ds006395__sub-C01721__2047", "ds006395/sub-C01721/anat/sub-C01721_T1w.nii.gz"], ["ds006395__sub-C01808__2048", "ds006395/sub-C01808/anat/sub-C01808_T1w.nii.gz"], ["obhb__177231880980", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-177231880980/ses-1/sub-177231880980_T1w.nii.gz"], ["obhb__306640245722", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-306640245722/ses-1/sub-306640245722_T1w.nii.gz"], ["obhb__334925895861", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-334925895861/ses-1/sub-334925895861_T1w.nii.gz"], ["obhb__349851138030", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-349851138030/ses-1/sub-349851138030_T1w.nii.gz"], ["obhb__363459366347", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-363459366347/ses-1/sub-363459366347_T1w.nii.gz"], ["sald__sub-031498", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031498/anat/sub-031498_T1w.nii.gz"], ["sald__sub-031508", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031508/anat/sub-031508_T1w.nii.gz"], ["sald__sub-031676", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031676/anat/sub-031676_T1w.nii.gz"], ["sald__sub-031680", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031680/anat/sub-031680_T1w.nii.gz"], ["sald__sub-031690", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031690/anat/sub-031690_T1w.nii.gz"], ["oasis1__OAS1_0069_MR1__mpr-1", "oasis1:disc2:OAS1_0069_MR1:mpr-1"], ["oasis1__OAS1_0318_MR1__mpr-1", "oasis1:disc9:OAS1_0318_MR1:mpr-1"], ["ds005752__sub-ON77753__2056", "ds005752/sub-ON77753/ses-01/anat/sub-ON77753_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON85010__2057", "ds005752/sub-ON85010/ses-01/anat/sub-ON85010_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON85305__2058", "ds005752/sub-ON85305/ses-01/anat/sub-ON85305_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON98018__2059", "ds005752/sub-ON98018/ses-01/anat/sub-ON98018_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz"], ["ds003592__sub-15__2060", "ds003592/sub-15/ses-1/anat/sub-15_ses-1_T1w.nii.gz"], ["ds005270__sub-074__2061", "ds005270/sub-074/anat/sub-074_T1w.nii.gz"], ["ds005123__sub-10369__2062", "ds005123/sub-10369/anat/sub-10369_T1w.nii.gz"], ["ds005123__sub-10642__2063", "ds005123/sub-10642/anat/sub-10642_T1w.nii.gz"], ["ds005123__sub-10850__2064", "ds005123/sub-10850/anat/sub-10850_T1w.nii.gz"], ["ds005123__sub-10875__2065", "ds005123/sub-10875/anat/sub-10875_T1w.nii.gz"], ["ds005123__sub-10924__2066", "ds005123/sub-10924/anat/sub-10924_T1w.nii.gz"], ["ds005123__sub-10952__2067", "ds005123/sub-10952/anat/sub-10952_T1w.nii.gz"], ["ds005405__sub-150__2068", "ds005405/sub-150/ses-bhb/anat/sub-150_ses-bhb_run-1_T1w.nii"], ["ds006395__sub-C01715__2069", "ds006395/sub-C01715/anat/sub-C01715_T1w.nii.gz"], ["ds006395__sub-C01810__2070", "ds006395/sub-C01810/anat/sub-C01810_T1w.nii.gz"], ["obhb__158500314736", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-158500314736/ses-1/sub-158500314736_T1w.nii.gz"], ["obhb__299690677871", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-299690677871/ses-1/sub-299690677871_T1w.nii.gz"], ["obhb__300521812714", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-300521812714/ses-1/sub-300521812714_T1w.nii.gz"], ["obhb__506541958125", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-506541958125/ses-1/sub-506541958125_T1w.nii.gz"], ["obhb__535399597427", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-535399597427/ses-1/sub-535399597427_T1w.nii.gz"], ["obhb__550394965552", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-550394965552/ses-1/sub-550394965552_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
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
