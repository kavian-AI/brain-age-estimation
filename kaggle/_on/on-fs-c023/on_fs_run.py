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

PART = 23
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["obhb__254729298278", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-254729298278/ses-1/sub-254729298278_T1w.nii.gz"], ["obhb__348573416015", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-348573416015/ses-1/sub-348573416015_T1w.nii.gz"], ["obhb__562839089454", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-562839089454/ses-1/sub-562839089454_T1w.nii.gz"], ["obhb__242741538964", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-242741538964/ses-1/sub-242741538964_T1w.nii.gz"], ["obhb__385846642581", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-385846642581/ses-1/sub-385846642581_T1w.nii.gz"], ["ds004711__sub-123__1884", "ds004711/sub-123/anat/sub-123_T1w.nii.gz"], ["ds005752__sub-ON41090__1885", "ds005752/sub-ON41090/ses-01/anat/sub-ON41090_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON63704__1886", "ds005752/sub-ON63704/ses-01/anat/sub-ON63704_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON85514__1887", "ds005752/sub-ON85514/ses-01/anat/sub-ON85514_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON91906__1888", "ds005752/sub-ON91906/ses-01/anat/sub-ON91906_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON97515__1889", "ds005752/sub-ON97515/ses-01/anat/sub-ON97515_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON98175__1890", "ds005752/sub-ON98175/ses-01/anat/sub-ON98175_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005270__sub-031__1891", "ds005270/sub-031/anat/sub-031_T1w.nii.gz"], ["ds005270__sub-100__1892", "ds005270/sub-100/anat/sub-100_T1w.nii.gz"], ["ds005123__sub-10486__1893", "ds005123/sub-10486/anat/sub-10486_T1w.nii.gz"], ["ds005123__sub-10802__1894", "ds005123/sub-10802/anat/sub-10802_T1w.nii.gz"], ["ds005123__sub-10863__1895", "ds005123/sub-10863/anat/sub-10863_T1w.nii.gz"], ["ds005123__sub-10908__1896", "ds005123/sub-10908/anat/sub-10908_T1w.nii.gz"], ["ds005123__sub-10929__1897", "ds005123/sub-10929/anat/sub-10929_T1w.nii.gz"], ["ds005405__sub-119__1898", "ds005405/sub-119/ses-bhb/anat/sub-119_ses-bhb_run-1_T1w.nii"], ["ds005405__sub-124__1899", "ds005405/sub-124/ses-bhb/anat/sub-124_ses-bhb_run-1_T1w.nii"], ["obhb__133559617458", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-133559617458/ses-1/sub-133559617458_T1w.nii.gz"], ["obhb__141883005255", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-141883005255/ses-1/sub-141883005255_T1w.nii.gz"], ["obhb__190824700808", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-190824700808/ses-1/sub-190824700808_T1w.nii.gz"], ["sald__sub-031357", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031357/anat/sub-031357_T1w.nii.gz"], ["sald__sub-031535", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031535/anat/sub-031535_T1w.nii.gz"], ["sald__sub-031548", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031548/anat/sub-031548_T1w.nii.gz"], ["sald__sub-031560", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031560/anat/sub-031560_T1w.nii.gz"], ["sald__sub-031639", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031639/anat/sub-031639_T1w.nii.gz"], ["sald__sub-031652", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031652/anat/sub-031652_T1w.nii.gz"], ["sald__sub-031660", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031660/anat/sub-031660_T1w.nii.gz"], ["sald__sub-031687", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031687/anat/sub-031687_T1w.nii.gz"], ["sald__sub-031704", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031704/anat/sub-031704_T1w.nii.gz"], ["ds004711__sub-029__1900", "ds004711/sub-029/anat/sub-029_T1w.nii.gz"], ["obhb__353593360279", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-353593360279/ses-1/sub-353593360279_T1w.nii.gz"], ["ds004711__sub-016__1901", "ds004711/sub-016/anat/sub-016_T1w.nii.gz"], ["ds007831__sub-LTV__1902", "ds007831/sub-LTV/anat/sub-LTV_T1w.nii.gz"], ["ds007831__sub-YER__1903", "ds007831/sub-YER/anat/sub-YER_T1w.nii.gz"], ["ds005752__sub-ON85616__1904", "ds005752/sub-ON85616/ses-01/anat/sub-ON85616_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON96510__1905", "ds005752/sub-ON96510/ses-01/anat/sub-ON96510_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005123__sub-10977__1906", "ds005123/sub-10977/anat/sub-10977_T1w.nii.gz"], ["ds005405__sub-051__1907", "ds005405/sub-051/ses-bhb/anat/sub-051_ses-bhb_run-1_T1w.nii"], ["obhb__475942618761", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-475942618761/ses-1/sub-475942618761_T1w.nii.gz"], ["obhb__666694788584", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-666694788584/ses-1/sub-666694788584_T1w.nii.gz"], ["obhb__997144206185", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-997144206185/ses-1/sub-997144206185_T1w.nii.gz"], ["obhb__721147018160", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-721147018160/ses-1/sub-721147018160_T1w.nii.gz"], ["sald__sub-031421", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031421/anat/sub-031421_T1w.nii.gz"], ["sald__sub-031488", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031488/anat/sub-031488_T1w.nii.gz"], ["sald__sub-031489", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031489/anat/sub-031489_T1w.nii.gz"], ["oasis1__OAS1_0167_MR1__mpr-1", "oasis1:disc5:OAS1_0167_MR1:mpr-1"], ["oasis1__OAS1_0336_MR1__mpr-1", "oasis1:disc9:OAS1_0336_MR1:mpr-1"], ["ds005752__sub-ON18047__1908", "ds005752/sub-ON18047/ses-01/anat/sub-ON18047_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON53213__1909", "ds005752/sub-ON53213/ses-01/anat/sub-ON53213_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005752__sub-ON87550__1910", "ds005752/sub-ON87550/ses-01/anat/sub-ON87550_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds005270__sub-003__1911", "ds005270/sub-003/anat/sub-003_T1w.nii.gz"], ["ds005270__sub-011__1912", "ds005270/sub-011/anat/sub-011_T1w.nii.gz"], ["ds005270__sub-012__1913", "ds005270/sub-012/anat/sub-012_T1w.nii.gz"], ["ds005270__sub-029__1914", "ds005270/sub-029/anat/sub-029_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
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
