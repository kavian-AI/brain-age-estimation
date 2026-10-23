# Chains the lifespan kernels through Kaggle's 2-GPU-session limit:
#   dlbs-fs-p0 / p1 run first; each time one finishes its output is fetched and
#   one IXI part (ixi-lba-p0 / p1) is pushed into the freed slot; IXI outputs
#   (JSONL + maps/) are fetched when they finish.
# Network hiccups are retried; the whole chain gives up after 30 hours.
import os
import shutil
import time

from kaggle.api.kaggle_api_extended import KaggleApi

KAG = os.path.dirname(os.path.abspath(__file__))
USER = "mohammadparsakavian/"
a = KaggleApi()
a.authenticate()


def retry(fn, *args, **kw):
    for _ in range(10):
        try:
            return fn(*args, **kw)
        except Exception as e:
            print("  retry:", type(e).__name__, str(e)[:120], flush=True)
            time.sleep(60)
    return None


def status(slug):
    s = retry(a.kernels_status, USER + slug)
    return str(s) if s is not None else "UNKNOWN"


def finished(s):
    return any(k in s for k in ('"COMPLETE"', '"ERROR"', '"CANCEL'))


def fetch(slug, dest):
    tmp = os.path.join(KAG, "_fetch_" + slug)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    retry(a.kernels_output, USER + slug, path=tmp, quiet=True)
    os.makedirs(dest, exist_ok=True)
    moved = []
    for root, _, files in os.walk(tmp):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, tmp)
            if f.endswith(".log"):
                rel = slug + ".log"
            dst = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(src, dst)
            moved.append(rel)
    shutil.rmtree(tmp, ignore_errors=True)
    jl = [m for m in moved if m.endswith(".jsonl")]
    n = sum(sum(1 for _ in open(os.path.join(dest, j))) for j in jl)
    print(f"{slug}: fetched {len(moved)} files, {n} JSONL rows", flush=True)


def push(folder):
    r = retry(a.kernels_push, os.path.join(KAG, folder))
    err = getattr(r, "error", None) if r is not None else "no response"
    print(f"push {folder}: {err or 'ok'}", flush=True)
    return not err


state = {"dlbs-fs-p0": "run", "dlbs-fs-p1": "run", "ixi-lba-p0": "wait", "ixi-lba-p1": "wait"}
t0 = time.time()
while time.time() - t0 < 30 * 3600:
    for k in ("dlbs-fs-p0", "dlbs-fs-p1"):
        if state[k] == "run":
            s = status(k)
            if finished(s):
                print(time.strftime("%H:%M"), k, "finished:", s, flush=True)
                fetch(k, os.path.join(KAG, "dlbs_fs_results"))
                state[k] = "done"
    for k, folder in (("ixi-lba-p0", "ixi_lba_p0"), ("ixi-lba-p1", "ixi_lba_p1")):
        free = sum(state[x] == "run" for x in state) < 2
        if state[k] == "wait" and free:
            if push(folder):
                state[k] = "run"
                time.sleep(120)
        elif state[k] == "run":
            s = status(k)
            if finished(s):
                print(time.strftime("%H:%M"), k, "finished:", s, flush=True)
                fetch(k, os.path.join(KAG, "ixi_lba_results"))
                state[k] = "done"
    if all(v == "done" for v in state.values()):
        break
    time.sleep(300)
print("watcher exit:", state, f"{(time.time()-t0)/3600:.1f} h", flush=True)
