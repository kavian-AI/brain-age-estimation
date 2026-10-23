# Waits for both OASIS FastSurfer kernels and pulls their JSONL results.
# Part 1 is only valid once the watcher has pushed its FIXED version, so the
# watcher's log is checked before part 1's status is trusted.
import os
import time

from kaggle.api.kaggle_api_extended import KaggleApi

OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\kaggle\oasis_fs_results"
WATCHER = r"C:\Users\SIBETA~1\AppData\Local\Temp\claude\C--Users-Sibetalaee-Desktop-recovery\aa6723fd-15d5-4904-b3b4-95d9c86e89bc\tasks\b9a5aaasx.output"
os.makedirs(OUT, exist_ok=True)
a = KaggleApi()
a.authenticate()


def status(slug):
    return str(a.kernels_status("mohammadparsakavian/" + slug))


def fetch(slug, part):
    d = os.path.join(OUT, f"p{part}")
    os.makedirs(d, exist_ok=True)
    a.kernels_output("mohammadparsakavian/" + slug, path=d, quiet=True)
    src = os.path.join(d, f"oasis_fs_part{part}.jsonl")
    if os.path.exists(src):
        n = sum(1 for _ in open(src))
        os.replace(src, os.path.join(OUT, f"oasis_fs_part{part}.jsonl"))
        print(f"{slug}: {n} scans fetched", flush=True)
    else:
        print(f"{slug}: no jsonl in output; files: {os.listdir(d)}", flush=True)


done = {0: False, 1: False}
t0 = time.time()
while not all(done.values()) and time.time() - t0 < 14 * 3600:
    s0 = status("oasis-fs-run-p0")
    if not done[0] and ('"COMPLETE"' in s0 or '"ERROR"' in s0 or '"CANCEL' in s0):
        print("p0 finished:", s0, flush=True)
        fetch("oasis-fs-run-p0", 0)
        done[0] = True
    pushed = os.path.exists(WATCHER) and "pushed p1" in open(WATCHER, errors="ignore").read()
    if pushed and not done[1]:
        s1 = status("oasis-fs-run-p1")
        if '"COMPLETE"' in s1 or '"ERROR"' in s1 or '"CANCEL' in s1:
            print("p1 finished:", s1, flush=True)
            fetch("oasis-fs-run-p1", 1)
            done[1] = True
    time.sleep(300)
print("all done" if all(done.values()) else "gave up waiting", flush=True)
