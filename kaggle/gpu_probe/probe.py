import subprocess, os, torch
print("cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
print("cpus", os.cpu_count(), flush=True)
print(subprocess.run("nvidia-smi -L; free -g | head -2", shell=True, capture_output=True, text=True).stdout, flush=True)
