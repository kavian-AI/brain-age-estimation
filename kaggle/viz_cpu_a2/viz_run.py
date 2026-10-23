# Visual QC of the preprocessing: re-segment a few scans per dataset with the exact pipeline
# (FastSurfer v2.4.2 on GPU) and draw the labels over the conformed T1 in three planes, through
# the left hippocampus. One PNG per scan + a contact sheet. Read-only check, nothing else changes.
import gzip, glob, json, os, shutil, subprocess, tarfile, urllib.request

PICKS = json.loads(r'''[["ds003592 typical (age 26)", "ds003592/sub-149/ses-1/anat/sub-149_ses-1_T1w.nii.gz"], ["ds004711 typical (age 46)", "ds004711/sub-052/anat/sub-052_T1w.nii.gz"], ["ds004725 typical (age 31)", "ds004725/sub-11022/anat/sub-11022_T1w.nii"], ["ds005026 typical (age 59)", "ds005026/sub-05/ses-01/anat/sub-05_ses-01_T1w.nii.gz"], ["ds005123 typical (age 37)", "ds005123/sub-10836/anat/sub-10836_T1w.nii.gz"], ["ds005270 typical (age 49)", "ds005270/sub-126/anat/sub-126_T1w.nii.gz"], ["ds005405 typical (age 44)", "ds005405/sub-101/ses-bhb/anat/sub-101_ses-bhb_run-1_T1w.nii"], ["ds005752 typical (age 31)", "ds005752/sub-ON98826/ses-01/anat/sub-ON98826_ses-01_acq-MPRAGE_T1w.nii.gz"], ["ds006395 typical (age 37)", "ds006395/sub-C01789/anat/sub-C01789_T1w.nii.gz"], ["ds007522 typical (age 64)", "ds007522/sub-HML0699/ses-01/anat/sub-HML0699_ses-01_rec-defaced_T1w.nii.gz"], ["ds007831 typical (age 62)", "ds007831/sub-SNJ/anat/sub-SNJ_T1w.nii.gz"], ["oasis1 typical (age 55)", "oasis1:disc11:OAS1_0389_MR1:mpr-1"], ["obhb typical (age 21)", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-958050974251/ses-1/sub-958050974251_T1w.nii.gz"], ["sald typical (age 48)", "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_BIDS/sub-031663/anat/sub-031663_T1w.nii.gz"], ["oasis1 age 25", "oasis1:disc11:OAS1_0406_MR1:mpr-1"], ["oasis1 age 80", "oasis1:disc5:OAS1_0166_MR1:mpr-1"], ["ds005405 cerebellum pair run-1 (cb 13 cm3)", "ds005405/sub-037/ses-glc/anat/sub-037_ses-glc_run-1_T1w.nii"], ["ds005405 cerebellum pair run-2 (cb 79 cm3)", "ds005405/sub-037/ses-glc/anat/sub-037_ses-glc_run-2_T1w.nii"], ["OpenBHB QC failure 347451361275", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-347451361275/ses-1/sub-347451361275_T1w.nii.gz"], ["OpenBHB QC failure 792041257894", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/val/rawdata/sub-792041257894/ses-1/sub-792041257894_T1w.nii.gz"], ["IXI typical (IXI002, age 36)", "kaggle:IXI002-Guys-0828-T1.nii"], ["DLBS typical (sub-1003 wave 1, age 54)", "ds004856/sub-1003/ses-wave1/anat/sub-1003_ses-wave1_acq-MPRAGE_run-1_T1w.nii.gz"], ["OASIS-2 (OAS2_0001 visit 1, age 87)", "kaggle:OAS2_0001_MR1/RAW/mpr-1.nifti.hdr"]]''')
S3 = "https://s3.amazonaws.com/openneuro.org/"
OUT = "/kaggle/working/viz"
os.makedirs(OUT, exist_ok=True)


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)


print("clone", sh("git clone -q --depth 1 --branch v2.4.2 https://github.com/Deep-MI/FastSurfer.git /tmp/FastSurfer").returncode, flush=True)
print("pip", sh("pip install -q yacs simpleitk nibabel scikit-image h5py lapy pandas").returncode, flush=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap

GROUPS = [("hippocampus", (17, 53), "#ffd400"), ("amygdala", (18, 54), "#00e5ff"), ("thalamus", (10, 49), "#36c24a"),
          ("lateral ventricles", (4, 43, 5, 44), "#2f6bff"), ("cerebellum cortex", (8, 47), "#ff8c1a"),
          ("brainstem", (16,), "#b04dff"), ("putamen/caudate", (11, 50, 12, 51), "#ff4fa3"), ("cortex", "ctx", "#e8352e")]


def fetch(label, key, path):
    if key.startswith("oasis1:"):
        _, disc, sess, mpr = key.split(":")
        url = f"https://download.nrg.wustl.edu/data/oasis_cross-sectional_{disc}.tar.gz"
        with urllib.request.urlopen(url, timeout=900) as r, tarfile.open(fileobj=r, mode="r|gz") as tf:
            for m in tf:
                b = os.path.basename(m.name)
                if m.isfile() and "/RAW/" in m.name and b.startswith(f"{sess}_{mpr}_anon."):
                    open(f"/tmp/{b}", "wb").write(tf.extractfile(m).read())
        im = nib.load(f"/tmp/{sess}_{mpr}_anon.hdr")
        arr = np.asanyarray(im.dataobj).astype(np.float32).squeeze()
        aff = np.array([[0, 0, -1.25, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)   # = pipeline (ASL-v2)
        aff[:3, 3] = -aff[:3, :3] @ (np.array(arr.shape) / 2.0)
        nib.save(nib.Nifti1Image(arr, aff), path)
        return
    if key.startswith("kaggle:"):
        hits = sorted(glob.glob("/kaggle/input/**/" + key[7:], recursive=True))
        im = nib.load(hits[0])
        nib.save(nib.Nifti1Image(np.asanyarray(im.dataobj).astype(np.float32).squeeze(), im.affine), path)
        return
    url = key if key.startswith("http") else S3 + key
    if key.endswith(".nii.gz"):
        urllib.request.urlretrieve(url, path)
    else:
        urllib.request.urlretrieve(url, "/tmp/x.nii")
        with open("/tmp/x.nii", "rb") as fi, gzip.open(path, "wb") as fo:
            shutil.copyfileobj(fi, fo)


def plot(label, orig, seg, png):
    t1 = np.asanyarray(nib.load(orig).dataobj).astype(float)
    lab = np.asanyarray(nib.load(seg).dataobj).astype(int)
    col = np.zeros(lab.shape, int)
    for i, (_, ids, _) in enumerate(GROUPS, start=1):
        col[(lab >= 1000) if ids == "ctx" else np.isin(lab, ids)] = i
    cmap = ListedColormap(["none"] + [g[2] for g in GROUPS])
    hip = np.argwhere(lab == 17)
    c = hip.mean(0).astype(int) if len(hip) else np.array(lab.shape) // 2
    vols = {g[0]: (int((lab >= 1000).sum()) if g[1] == "ctx" else int(np.isin(lab, g[1]).sum())) for g in GROUPS}
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    for a, (sl, nm) in zip(ax, ((np.s_[c[0], :, :], "sagittal"), (np.s_[:, :, c[2]], "coronal"), (np.s_[:, c[1], :], "axial"))):
        img, cc = t1[sl], col[sl]
        a.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=np.percentile(t1[t1 > 0], 99.5))
        a.imshow(np.rot90(np.ma.masked_equal(cc, 0)), cmap=cmap, vmin=0, vmax=len(GROUPS), alpha=0.45, interpolation="nearest")
        a.set_title(nm)
        a.axis("off")
    fig.suptitle(f"{label}\nhippocampus {vols['hippocampus']/1e3:.1f} cm3 | lat. ventricles {vols['lateral ventricles']/1e3:.1f} | "
                 f"cerebellum {vols['cerebellum cortex']/1e3:.0f} | cortex {vols['cortex']/1e3:.0f} cm3", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=g[2]) for g in GROUPS]
    fig.legend(handles, [g[0] for g in GROUPS], loc="lower center", ncol=8, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(png, dpi=80)
    plt.close(fig)


done = []
for i, (label, key) in list(enumerate(PICKS))[16:23]:
    try:
        os.makedirs("/tmp/in", exist_ok=True)
        p = f"/tmp/in/s{i}.nii.gz"
        fetch(label, key, p)
        shutil.rmtree("/tmp/out", ignore_errors=True)
        r = sh("cd /tmp/FastSurfer && PYTHONPATH=/tmp/FastSurfer python FastSurferCNN/run_prediction.py "
               f"--t1 {p} --sid s{i} --sd /tmp/out --device cpu --viewagg_device cpu --threads 4 --vox_size 1")
        seg = f"/tmp/out/s{i}/mri/aparc.DKTatlas+aseg.deep.mgz"
        if not os.path.exists(seg):
            print("FAIL", label, r.stderr[-400:], flush=True)
            continue
        png = f"{OUT}/{i:02d}.png"
        plot(label, f"/tmp/out/s{i}/mri/orig.mgz", seg, png)
        done.append(png)
        print("ok", i, label, flush=True)
    except Exception as e:
        print("ERROR", label, type(e).__name__, e, flush=True)
print("plots:", len(done), "of", len(PICKS), flush=True)
