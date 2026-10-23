import subprocess, numpy
print("numpy", numpy.__version__, flush=True)
r = subprocess.run("pip install -q yacs simpleitk nibabel scikit-image h5py lapy pandas onnxruntime-gpu 2>&1 | tail -3; python -c 'import numpy; print(\"numpy after pip\", numpy.__version__)'", shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr, flush=True)
