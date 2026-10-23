# Processes every scan in data/all_jobs.csv (OpenNeuro healthy adults + OpenBHB) through
# FastSurfer + SynthStrip on Kaggle, using every account, CPU and GPU, without doing any
# scan twice.
#   - each account keeps up to N_CPU CPU kernels cycling (no weekly quota, ~11 min/scan,
#     CPU_BATCH scans each) and up to N_GPU GPU kernels (~45 s/scan, GPU_BATCH scans)
#     whenever its weekly GPU quota allows (a quota refusal is retried after 2 h);
#   - every kernel takes the highest-priority scans (older people first) not yet done or
#     assigned; finished kernels are fetched into on_fs_results/<kernel>/, their scans
#     marked done, and unreached scans return to the pool.
# State: on_fs_state.json (restart-safe; also adopts kernels started by the older
# single-account watcher).
import json
import os
import shutil
import socket
import time

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# the Kaggle client sets no timeout: through the local proxy a dropped connection hung the
# watcher for 6.5 h (14 Oct 11:07-17:34). Every socket now gives up after 3 minutes and retry() retries.
socket.setdefaulttimeout(180)

KAG = os.path.dirname(os.path.abspath(__file__))
ACCOUNTS = {"mohammadparsakavian": "access_token", "kavianasli": "token_acc2"}
N_CPU, CPU_BATCH = 5, 32
# GPU per account: kavianasli's weekly GPU allowance is used up until 24 Oct (a push there sits QUEUED
# forever holding its scans); the main account has ~2.6 h left -> one T4x2 kernel of ~230 scans (36 s/scan)
N_GPU, GPU_BATCH = {"mohammadparsakavian": 1, "kavianasli": 0}, 230
N_TPU, TPU_BATCH = 0, 900                  # TPU disabled: API-pushed TPU kernels run as plain 4-CPU sessions (kaggle-cli #1197), 0 TPU hours charged
TPU_SHAPES = ["TpuV5E8", "TpuV6E8"]       # Tpu1VmV38 (v3-8) is retired
TPU_TEMPLATE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "on_fs_tpu_template.py"),
                    encoding="utf-8").read()
JOBS = pd.read_csv(os.path.join(KAG, "..", "data", "all_jobs.csv"))
TEMPLATE = open(os.path.join(KAG, "on_fs_template.py"), encoding="utf-8").read()
STATE_F = os.path.join(KAG, "on_fs_state.json")
RES = os.path.join(KAG, "on_fs_results")

API = {}
for user, tok in ACCOUNTS.items():
    os.environ["KAGGLE_API_TOKEN"] = os.path.expanduser(f"~/.kaggle/{tok}")
    a = KaggleApi()
    a.authenticate()
    API[user] = a


def retry(fn, *args, **kw):
    for _ in range(10):
        try:
            return fn(*args, **kw)
        except Exception as e:
            print("  retry:", type(e).__name__, str(e)[:100], flush=True)
            time.sleep(60)
    return None


def status(user, name):
    s = retry(API[user].kernels_status, f"{user}/{name}")
    return str(s) if s is not None else "UNKNOWN"


def finished(s):
    return any(k in s for k in ('"COMPLETE"', '"ERROR"', '"CANCEL'))


def fetch(user, name):
    tmp = os.path.join(KAG, "_fetch_" + name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    retry(API[user].kernels_output, f"{user}/{name}", path=tmp, quiet=True)
    dest = os.path.join(RES, name)
    os.makedirs(dest, exist_ok=True)
    for root, _, files in os.walk(tmp):
        for f in files:
            os.replace(os.path.join(root, f), os.path.join(dest, f if not f.endswith(".log") else name + ".log"))
    shutil.rmtree(tmp, ignore_errors=True)


def valid(row):
    # OASIS-1 rows made before the orientation fix (Analyze sagittal read as axial) are invalid
    # ASR-v1 rows: correct volumes but hemispheres swapped (fixed when loading); ASL-v2: fully correct
    return not row["scan"].startswith("oasis1__") or row.get("o1_orient") in ("ASR-v1", "ASL-v2")


def done_from_results():
    ids = set()
    for root, _, files in os.walk(RES):
        for f in files:
            if f.endswith(".jsonl"):
                for l in open(os.path.join(root, f)):
                    r = json.loads(l)
                    if valid(r):
                        ids.add(r["scan"])
    return ids


def make_kernel(user, name, jobs, gpu, part, tpu_shape=None):
    folder = os.path.join(KAG, "_on", name)
    os.makedirs(folder, exist_ok=True)
    if tpu_shape:
        t = TPU_TEMPLATE.replace("PART = 0", f"PART = {part}", 1).replace("__JOBS__", json.dumps(jobs))
        open(os.path.join(folder, "on_fs_run.py"), "w", encoding="utf-8").write(t)
        meta = {"id": f"{user}/{name}", "title": name, "code_file": "on_fs_run.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": True,
                "machine_shape": tpu_shape, "docker_image_pinning_type": "latest",
                "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
        json.dump(meta, open(os.path.join(folder, "kernel-metadata.json"), "w"), indent=2)
        return folder
    t = TEMPLATE.replace("PART = 0\n", f"PART = {part}\n", 1).replace("__JOBS__", json.dumps(jobs))
    if not gpu:
        t = t.replace("--device cuda --viewagg_device cuda", "--device cpu --viewagg_device cpu")
        t = t.replace('if "cuda" in "--device cuda" else 1', 'if False else 1')
        t = t.replace("ss_list.txt -g", "ss_list.txt").replace(".ssmask.nii.gz -g", ".ssmask.nii.gz")
        t = t.replace("BUDGET_S = 9.4 * 3600", "BUDGET_S = 11.0 * 3600").replace("CHUNK = 12", "CHUNK = 3")
    open(os.path.join(folder, "on_fs_run.py"), "w", encoding="utf-8").write(t)
    meta = {"id": f"{user}/{name}", "title": name, "code_file": "on_fs_run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    json.dump(meta, open(os.path.join(folder, "kernel-metadata.json"), "w"), indent=2)
    return folder


def next_jobs(S, n):
    busy = set(S["done"]) | {j for k in S["running"].values() for j in k["scans"]}
    todo = JOBS[~JOBS.scan_id.isin(busy)]
    return [[a, b] for a, b in zip(todo.scan_id.head(n), todo.key.head(n))]


def push(S, user, gpu, tpu=False):
    jobs = next_jobs(S, TPU_BATCH if tpu else (GPU_BATCH if gpu else CPU_BATCH))
    if not jobs:
        return False
    S["counter"] += 1
    name = f"on-fs-{'t' if tpu else ('g' if gpu else 'c')}{S['counter']:03d}"
    err = ""
    for shape in (TPU_SHAPES if tpu else [None]):
        folder = make_kernel(user, name, jobs, gpu, S["counter"], tpu_shape=shape)
        r = retry(API[user].kernels_push, folder)
        err = (getattr(r, "error", None) or "") if r is not None else "no response"
        if not err:
            break
        print(time.strftime("%m-%d %H:%M"), user, "push", name, shape or "", "refused:", err.strip()[:100],
              flush=True)
    if err:
        # a real refusal (quota / session limit) waits; a network failure ("no response", e.g. the local
        # proxy being down) is retried at the next 10-minute cycle instead of blocking for hours
        wait = 0 if err == "no response" else (2 * 3600 if (tpu or gpu) else 1800)
        if tpu:
            S["tpu_blocked"][user] = time.time() + wait
        elif gpu:
            S["gpu_blocked"][user] = time.time() + wait
        else:
            S["cpu_blocked"][user] = time.time() + wait
        if err == "no response":
            print(time.strftime("%m-%d %H:%M"), "  (network failure - check the proxy; retrying next cycle)", flush=True)
        return False
    S["running"][name] = {"user": user, "scans": [j[0] for j in jobs], "gpu": gpu, "tpu": tpu, "t": time.time()}
    print(time.strftime("%m-%d %H:%M"), user, "pushed", name, len(jobs), "scans",
          "(TPU)" if tpu else ("(GPU)" if gpu else "(CPU)"), flush=True)
    save_state(S)
    time.sleep(60)
    return True


def load_state():
    S = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {"running": {}, "done": [], "counter": 0}
    for k in S["running"].values():
        k.setdefault("user", "mohammadparsakavian")
    S.setdefault("gpu_blocked", {})
    S.setdefault("cpu_blocked", {})
    S.setdefault("tpu_blocked", {})
    for k in S["running"].values():
        k.setdefault("tpu", False)
    S.pop("gpu_blocked_until", None)
    return S


def save_state(S):
    json.dump(S, open(STATE_F, "w"), indent=1)


S = load_state()
S["done"] = sorted(done_from_results())          # recomputed: invalid rows are re-queued
save_state(S)
print("start: done", len(S["done"]), "of", len(JOBS), "| running", {k: v["user"] for k, v in S["running"].items()},
      flush=True)
t0 = time.time()
while time.time() - t0 < 21 * 24 * 3600:
    for name in list(S["running"]):
        k = S["running"][name]
        st = status(k["user"], name)
        if finished(st):
            print(time.strftime("%m-%d %H:%M"), name, st, flush=True)
            fetch(k["user"], name)
            S["done"] = sorted(done_from_results())
            del S["running"][name]
            save_state(S)
            print(f"   done {len(S['done'])}/{len(JOBS)}", flush=True)
    if len(S["done"]) >= len(JOBS) and not S["running"]:
        break
    for user in ACCOUNTS:
        mine = [v for v in S["running"].values() if v["user"] == user]
        if time.time() > S["gpu_blocked"].get(user, 0):
            for _ in range(N_GPU.get(user, 0) - sum(v["gpu"] for v in mine)):
                if not push(S, user, gpu=True):
                    break
        mine = [v for v in S["running"].values() if v["user"] == user]
        if time.time() > S["tpu_blocked"].get(user, 0):
            for _ in range(N_TPU - sum(v.get("tpu", False) for v in mine)):
                if not push(S, user, gpu=False, tpu=True):
                    break
        mine = [v for v in S["running"].values() if v["user"] == user]
        if time.time() > S["cpu_blocked"].get(user, 0):
            for _ in range(N_CPU - sum(not v["gpu"] and not v.get("tpu", False) for v in mine)):
                if not push(S, user, gpu=False):
                    break
    save_state(S)
    with open(os.path.join(KAG, "_queue_heartbeat.txt"), "w") as hb:    # proof of life for checks
        hb.write(f"{time.strftime('%m-%d %H:%M:%S')} done {len(S['done'])} running {len(S['running'])}\n")
    time.sleep(600)
print("watcher exit; done", len(S["done"]), flush=True)
