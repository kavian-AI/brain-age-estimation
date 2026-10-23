import os, subprocess, sys, time
print("python", sys.version.split()[0], "cpus", os.cpu_count(), flush=True)
print(subprocess.run("free -g | head -2; nproc; lscpu | grep -i 'model name\|socket\|thread\|core'; df -h /tmp /kaggle/working | tail -2; pip list 2>/dev/null | grep -i '^torch\|jax\|numpy '", shell=True, capture_output=True, text=True).stdout, flush=True)
t = time.time()
import numpy as np
a = np.random.rand(3000, 3000); [a @ a for _ in range(5)]
print("numpy matmul 5x3000^2: %.1fs" % (time.time() - t), flush=True)
try:
    import torch
    print("torch", torch.__version__, "threads", torch.get_num_threads(), flush=True)
except Exception as e:
    print("no torch:", e, flush=True)
