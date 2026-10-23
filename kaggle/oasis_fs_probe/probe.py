# Probe: can FastSurfer segment a raw OASIS-2 MPRAGE on a Kaggle T4, and are the
# volumes anatomically sane? Run once before committing hours of GPU to 746 scans.
import glob, os, subprocess, sys, time
t0 = time.time()
def sh(c, check=True):
    print(">>", c, flush=True)
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if check and r.returncode != 0: print("FAILED rc", r.returncode, flush=True)
    return r
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
imgs = sorted(glob.glob("/kaggle/input/**/OAS2_0001_MR1/RAW/mpr-1.nifti.hdr", recursive=True))
print("probe image:", imgs, flush=True)
if not imgs:
    for d in glob.glob("/kaggle/input/*/*")[:10]: print("  ", d)
    sys.exit(1)
import nibabel as nib, numpy as np
im = nib.load(imgs[0])
print("shape", im.shape, "zooms", im.header.get_zooms(), "axcodes", nib.aff2axcodes(im.affine), flush=True)
print(im.affine, flush=True)
os.makedirs("/kaggle/working/in", exist_ok=True)
nib.save(nib.Nifti1Image(np.asanyarray(im.dataobj).astype(np.float32).squeeze(), im.affine), "/kaggle/working/in/t1.nii.gz")
sh("git clone --depth 1 --branch v2.4.2 https://github.com/Deep-MI/FastSurfer.git /tmp/FastSurfer || git clone --depth 1 https://github.com/Deep-MI/FastSurfer.git /tmp/FastSurfer")
sh("pip install -q yacs simpleitk nibabel scikit-image h5py lapy pandas 2>&1 | tail -3")
os.chdir("/tmp/FastSurfer"); os.environ["PYTHONPATH"] = "/tmp/FastSurfer"
sh("python FastSurferCNN/run_prediction.py --help | head -60", check=False)
r = sh("python FastSurferCNN/run_prediction.py --t1 /kaggle/working/in/t1.nii.gz --sid probe --sd /tmp/out --device cuda --viewagg_device cuda --threads 2", check=False)
segs = glob.glob("/tmp/out/probe/mri/*.mgz")
print("outputs:", segs, flush=True)
seg = [s for s in segs if "aparc" in s and "aseg" in s]
if seg:
    s = nib.load(seg[0]); a = np.asanyarray(s.dataobj).astype(int)
    vox = float(np.prod(s.header.get_zooms()))
    lab = {17: "L-Hippocampus", 53: "R-Hippocampus", 4: "L-LatVent", 43: "R-LatVent",
           10: "L-Thalamus", 49: "R-Thalamus", 1006: "L-entorhinal", 2006: "R-entorhinal"}
    for k, v in lab.items(): print(f"   {v:15s} {int((a==k).sum()*vox):8d} mm3", flush=True)
    print("   total labelled brain", int((a > 0).sum() * vox), "mm3", flush=True)
    print("   distinct labels", len(np.unique(a)), flush=True)
print("elapsed", round(time.time() - t0), "s", flush=True)
