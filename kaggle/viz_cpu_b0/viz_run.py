# Visual QC, batch 2 (suspected cut-off / outlier scans), with the SynthStrip head-size mask outline.
# Visual QC of the preprocessing: re-segment a few scans per dataset with the exact pipeline
# (FastSurfer v2.4.2 on GPU) and draw the labels over the conformed T1 in three planes, through
# the left hippocampus. One PNG per scan + a contact sheet. Read-only check, nothing else changes.
import gzip, glob, json, os, shutil, subprocess, tarfile, urllib.request

PICKS = json.loads(r'''[["cut-off: worst (cerebellum 32 cm3, brainstem 3)", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-901046779045/ses-1/sub-901046779045_T1w.nii.gz"], ["cut-off: brainstem 0.9 cm3", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-923642697698/ses-1/sub-923642697698_T1w.nii.gz"], ["cut-off: borderline (cb 99 cm3, labels missing)", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-933451667612/ses-1/sub-933451667612_T1w.nii.gz"], ["child 11y: brain/ICV z -7.9", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-924333698510/ses-1/sub-924333698510_T1w.nii.gz"], ["child 12y: frontal z -5.6, brain/ICV -6.4", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-213073713974/ses-1/sub-213073713974_T1w.nii.gz"], ["child 12y: top cortex z -6.0", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-784746686469/ses-1/sub-784746686469_T1w.nii.gz"], ["adult 55y: inf.temporal z -7.0, brain/ICV -8.9", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-264782182812/ses-1/sub-264782182812_T1w.nii.gz"], ["adult 22y: inf.temporal z -7.4", "https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main/train/rawdata/sub-719423973532/ses-1/sub-719423973532_T1w.nii.gz"], ["ds005405 sub-127: inf.temporal z -6.2", "ds005405/sub-127/ses-bhb/anat/sub-127_ses-bhb_run-1_T1w.nii"], ["OASIS-1 84y: inf.temporal z -5.4", "oasis1:disc3:OAS1_0110_MR1:mpr-1"]]''')
S3 = "https://s3.amazonaws.com/openneuro.org/"
OUT = "/kaggle/working/viz"
os.makedirs(OUT, exist_ok=True)


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)


print("clone", sh("git clone -q --depth 1 --branch v2.4.2 https://github.com/Deep-MI/FastSurfer.git /tmp/FastSurfer").returncode, flush=True)
print("pip", sh("pip install -q yacs simpleitk nibabel scikit-image h5py lapy pandas surfa").returncode, flush=True)
if sh("python -c 'import surfa'").returncode != 0:
    print("surfa from GitHub rc", sh("pip install -q git+https://github.com/freesurfer/surfa.git").returncode, flush=True)
os.makedirs("/tmp/fs/models", exist_ok=True)
urllib.request.urlretrieve("https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthstrip/mri_synthstrip", "/tmp/fs/mri_synthstrip")
urllib.request.urlretrieve("https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt", "/tmp/fs/models/synthstrip.1.pt")
_s = open("/tmp/fs/mri_synthstrip").read().replace("torch.load(modelfile, map_location=device)", "torch.load(modelfile, map_location=device, weights_only=False)")
open("/tmp/fs/mri_synthstrip", "w").write(_s)
os.environ["FREESURFER_HOME"] = "/tmp/fs"
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


def plot(label, orig, seg, mask, png):
    from nibabel.processing import resample_from_to
    oi = nib.load(orig)
    t1 = np.asanyarray(oi.dataobj).astype(float)
    lab = np.asanyarray(nib.load(seg).dataobj).astype(int)
    msk = None
    if mask and os.path.exists(mask):
        mi = nib.load(mask)
        msk = np.asanyarray(resample_from_to(nib.Nifti1Image((np.asanyarray(mi.dataobj) > 0).astype(np.float32), mi.affine),
                                             (oi.shape, oi.affine), order=0).dataobj) > 0.5
    col = np.zeros(lab.shape, int)
    for i, (_, ids, _) in enumerate(GROUPS, start=1):
        col[(lab >= 1000) if ids == "ctx" else np.isin(lab, ids)] = i
    cmap = ListedColormap(["none"] + [g[2] for g in GROUPS])
    hip = np.argwhere(lab == 17)
    c = hip.mean(0).astype(int) if len(hip) else np.array(lab.shape) // 2
    bs = np.argwhere(lab == 16)
    cb = bs.mean(0).astype(int) if len(bs) else np.array(lab.shape) // 2
    vols = {g[0]: (int((lab >= 1000).sum()) if g[1] == "ctx" else int(np.isin(lab, g[1]).sum())) for g in GROUPS}
    fig, ax = plt.subplots(1, 4, figsize=(19, 5.4))
    views = ((np.s_[cb[0], :, :], "mid-sagittal (through brainstem)"), (np.s_[c[0], :, :], "sagittal (left hippocampus)"),
             (np.s_[:, :, c[2]], "coronal"), (np.s_[:, c[1], :], "axial"))
    for a, (sl, nm) in zip(ax, views):
        img, cc = t1[sl], col[sl]
        a.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=np.percentile(t1[t1 > 0], 99.5))
        a.imshow(np.rot90(np.ma.masked_equal(cc, 0)), cmap=cmap, vmin=0, vmax=len(GROUPS), alpha=0.45, interpolation="nearest")
        if msk is not None:
            a.contour(np.rot90(msk[sl].astype(float)), levels=[0.5], colors="white", linewidths=0.9)
        a.set_title(nm)
        a.axis("off")
    icv = f" | head-size mask (white line) {msk.sum()/1e3:.0f} cm3" if msk is not None else ""
    fig.suptitle(f"{label}\nhippocampus {vols['hippocampus']/1e3:.1f} cm3 | lat. ventricles {vols['lateral ventricles']/1e3:.1f} | "
                 f"cerebellum {vols['cerebellum cortex']/1e3:.0f} | brainstem {vols['brainstem']/1e3:.1f} | cortex {vols['cortex']/1e3:.0f} cm3" + icv, fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=g[2]) for g in GROUPS]
    fig.legend(handles, [g[0] for g in GROUPS], loc="lower center", ncol=8, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(png, dpi=80)
    plt.close(fig)


done = []
for i, (label, key) in list(enumerate(PICKS))[0:5]:
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
        rs = sh(f"python /tmp/fs/mri_synthstrip -i {p} -m /tmp/in/s{i}.mask.nii.gz")
        if rs.returncode: print("synthstrip rc", rs.returncode, rs.stderr[-300:], flush=True)
        png = f"{OUT}/{i:02d}.png"
        plot(label, f"/tmp/out/s{i}/mri/orig.mgz", seg, f"/tmp/in/s{i}.mask.nii.gz", png)
        done.append(png)
        print("ok", i, label, flush=True)
    except Exception as e:
        print("ERROR", label, type(e).__name__, e, flush=True)
print("plots:", len(done), "of", len(PICKS), flush=True)
