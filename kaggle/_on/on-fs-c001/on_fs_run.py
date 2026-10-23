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

PART = 1
T0 = time.time()
BUDGET_S = 11.2 * 3600
OUT = f"/kaggle/working/on_fs_part{PART}.jsonl"
S3 = "https://s3.amazonaws.com/openneuro.org/"
JOBS = json.loads(r'''[["ds005270__sub-112__3", "ds005270/sub-112/anat/sub-112_T1w.nii.gz"], ["ds005270__sub-034__6", "ds005270/sub-034/anat/sub-034_T1w.nii.gz"], ["ds007831__sub-PAM__7", "ds007831/sub-PAM/anat/sub-PAM_T1w.nii.gz"], ["ds007831__sub-LTE__13", "ds007831/sub-LTE/anat/sub-LTE_T1w.nii.gz"], ["ds003592__sub-197__16", "ds003592/sub-197/ses-1/anat/sub-197_ses-1_T1w.nii.gz"], ["ds005270__sub-037__17", "ds005270/sub-037/anat/sub-037_T1w.nii.gz"], ["ds007831__sub-LDK__18", "ds007831/sub-LDK/anat/sub-LDK_T1w.nii.gz"], ["ds003592__sub-280__23", "ds003592/sub-280/ses-1/anat/sub-280_ses-1_T1w.nii.gz"], ["ds005270__sub-009__24", "ds005270/sub-009/anat/sub-009_T1w.nii.gz"], ["ds004711__sub-164__25", "ds004711/sub-164/anat/sub-164_T1w.nii.gz"], ["ds005270__sub-002__29", "ds005270/sub-002/anat/sub-002_T1w.nii.gz"], ["ds005270__sub-006__30", "ds005270/sub-006/anat/sub-006_T1w.nii.gz"], ["ds005270__sub-008__31", "ds005270/sub-008/anat/sub-008_T1w.nii.gz"], ["ds004711__sub-183__32", "ds004711/sub-183/anat/sub-183_T1w.nii.gz"], ["ds004725__sub-21011__33", "ds004725/sub-21011/anat/sub-21011_T1w.nii"], ["ds004711__sub-153__34", "ds004711/sub-153/anat/sub-153_T1w.nii.gz"], ["ds004725__sub-22029__35", "ds004725/sub-22029/anat/sub-22029_T1w.nii"], ["ds003592__sub-274__41", "ds003592/sub-274/ses-1/anat/sub-274_ses-1_T1w.nii.gz"], ["ds003592__sub-288__42", "ds003592/sub-288/ses-1/anat/sub-288_ses-1_T1w.nii.gz"], ["ds003592__sub-295__43", "ds003592/sub-295/ses-1/anat/sub-295_ses-1_T1w.nii.gz"], ["ds005270__sub-017__44", "ds005270/sub-017/anat/sub-017_T1w.nii.gz"], ["ds005123__sub-10858__45", "ds005123/sub-10858/anat/sub-10858_T1w.nii.gz"], ["ds005123__sub-10974__46", "ds005123/sub-10974/anat/sub-10974_T1w.nii.gz"], ["ds004725__sub-21013__47", "ds004725/sub-21013/anat/sub-21013_T1w.nii"], ["ds004725__sub-21006__48", "ds004725/sub-21006/anat/sub-21006_T1w.nii"], ["ds007522__sub-HML0078__49", "ds007522/sub-HML0078/ses-01/anat/sub-HML0078_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0169__50", "ds007522/sub-HML0169/ses-01/anat/sub-HML0169_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0189__51", "ds007522/sub-HML0189/ses-01/anat/sub-HML0189_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0250__52", "ds007522/sub-HML0250/ses-01/anat/sub-HML0250_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0339__53", "ds007522/sub-HML0339/ses-01/anat/sub-HML0339_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0406__54", "ds007522/sub-HML0406/ses-01/anat/sub-HML0406_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0427__55", "ds007522/sub-HML0427/ses-01/anat/sub-HML0427_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0541__56", "ds007522/sub-HML0541/ses-01/anat/sub-HML0541_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0654__57", "ds007522/sub-HML0654/ses-01/anat/sub-HML0654_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0689__58", "ds007522/sub-HML0689/ses-01/anat/sub-HML0689_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0774__59", "ds007522/sub-HML0774/ses-01/anat/sub-HML0774_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0860__60", "ds007522/sub-HML0860/ses-01/anat/sub-HML0860_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0942__61", "ds007522/sub-HML0942/ses-01/anat/sub-HML0942_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0962__62", "ds007522/sub-HML0962/ses-01/anat/sub-HML0962_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML1080__63", "ds007522/sub-HML1080/ses-01/anat/sub-HML1080_ses-01_rec-defaced_T1w.nii.gz"], ["ds003592__sub-124__71", "ds003592/sub-124/ses-1/anat/sub-124_ses-1_T1w.nii.gz"], ["ds003592__sub-292__72", "ds003592/sub-292/ses-1/anat/sub-292_ses-1_T1w.nii.gz"], ["ds003592__sub-296__73", "ds003592/sub-296/ses-1/anat/sub-296_ses-1_T1w.nii.gz"], ["ds005270__sub-014__74", "ds005270/sub-014/anat/sub-014_T1w.nii.gz"], ["ds005270__sub-113__75", "ds005270/sub-113/anat/sub-113_T1w.nii.gz"], ["ds005405__sub-107__76", "ds005405/sub-107/ses-bhb/anat/sub-107_ses-bhb_run-1_T1w.nii"], ["ds004725__sub-21020__77", "ds004725/sub-21020/anat/sub-21020_T1w.nii"], ["ds007831__sub-LGQ__78", "ds007831/sub-LGQ/anat/sub-LGQ_T1w.nii.gz"], ["ds007831__sub-YFE__79", "ds007831/sub-YFE/anat/sub-YFE_T1w.nii.gz"], ["ds007522__sub-HML0165__80", "ds007522/sub-HML0165/ses-01/anat/sub-HML0165_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0277__81", "ds007522/sub-HML0277/ses-01/anat/sub-HML0277_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0362__82", "ds007522/sub-HML0362/ses-01/anat/sub-HML0362_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0390__83", "ds007522/sub-HML0390/ses-01/anat/sub-HML0390_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0439__84", "ds007522/sub-HML0439/ses-01/anat/sub-HML0439_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0521__85", "ds007522/sub-HML0521/ses-01/anat/sub-HML0521_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0646__86", "ds007522/sub-HML0646/ses-01/anat/sub-HML0646_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0682__87", "ds007522/sub-HML0682/ses-01/anat/sub-HML0682_ses-01_rec-defaced_T1w.nii.gz"], ["ds007522__sub-HML0726__88", "ds007522/sub-HML0726/ses-01/anat/sub-HML0726_ses-01_rec-defaced_T1w.nii.gz"]]''')          # [[scan_id, s3_key], ...] for this part
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


def fetch_one(job, d):
    sid, key = job
    for attempt in range(4):
        try:
            p = f"{d}/{sid}.nii.gz"
            if key.endswith(".nii.gz"):
                urllib.request.urlretrieve(S3 + key, p)
            else:
                urllib.request.urlretrieve(S3 + key, f"{d}/{sid}.nii")
                with open(f"{d}/{sid}.nii", "rb") as fi, gzip.open(p, "wb", compresslevel=1) as fo:
                    shutil.copyfileobj(fi, fo)
                os.remove(f"{d}/{sid}.nii")
            meta = {}
            ds = key.split("/")[0]
            for side in (key.rsplit(".nii", 1)[0] + ".json", f"{ds}/T1w.json"):   # BIDS inheritance
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
