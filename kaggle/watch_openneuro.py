# Processes every scan in data/openneuro_jobs.csv through FastSurfer + SynthStrip on
# Kaggle, as fast as the account allows, without doing any scan twice:
#   - up to N_CPU CPU kernels cycle continuously (no weekly quota; ~11 min/scan),
#     each taking the next CPU_BATCH highest-priority scans not yet done or assigned;
#   - when the weekly GPU quota is available, up to 2 GPU kernels take GPU_BATCH
#     scans each (~45 s/scan); a quota refusal is retried every 2 hours;
#   - finished kernels are fetched into on_fs_results/<kernel>/ and their scans
#     marked done; scans a kernel did not reach return to the pool.
# State lives in on_fs_state.json, so the watcher can be restarted safely.
import json
import os
import time

import pandas as pd

KAG = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(KAG, "watch_lifespan.py"), encoding="utf-8").read()
exec(compile(src.split("state = {")[0], "w", "exec"))          # a, retry, status, finished, fetch

JOBS = pd.read_csv(os.path.join(KAG, "..", "data", "openneuro_jobs.csv"))
JOBS = JOBS[JOBS.ds != "ds007857"]                             # MiND: files access-restricted (403)
TEMPLATE = open(os.path.join(KAG, "on_fs_template.py"), encoding="utf-8").read()
STATE_F = os.path.join(KAG, "on_fs_state.json")
RES = os.path.join(KAG, "on_fs_results")
N_CPU, CPU_BATCH = 5, 58
N_GPU, GPU_BATCH = 2, 760


def load_state():
    if os.path.exists(STATE_F):
        return json.load(open(STATE_F))
    return {"running": {}, "done": [], "counter": 0, "gpu_blocked_until": 0}


def save_state(S):
    json.dump(S, open(STATE_F, "w"), indent=1)


def done_from_results():
    ids = set()
    for root, _, files in os.walk(RES):
        for f in files:
            if f.endswith(".jsonl"):
                ids |= {json.loads(l)["scan"] for l in open(os.path.join(root, f))}
    return ids


def make_kernel(name, jobs, gpu, part):
    folder = os.path.join(KAG, "_on", name)
    os.makedirs(folder, exist_ok=True)
    t = TEMPLATE.replace("PART = 0\n", f"PART = {part}\n", 1).replace("__JOBS__", json.dumps(jobs))
    if not gpu:
        t = t.replace("--device cuda --viewagg_device cuda", "--device cpu --viewagg_device cpu")
        t = t.replace("-m /tmp/ssmask.nii.gz -g", "-m /tmp/ssmask.nii.gz")
        t = t.replace("BUDGET_S = 9.4 * 3600", "BUDGET_S = 11.2 * 3600")
    open(os.path.join(folder, "on_fs_run.py"), "w", encoding="utf-8").write(t)
    meta = {"id": f"mohammadparsakavian/{name}", "title": name, "code_file": "on_fs_run.py", "language": "python",
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


def push(S, gpu):
    n = GPU_BATCH if gpu else CPU_BATCH
    jobs = next_jobs(S, n)
    if not jobs:
        return False
    S["counter"] += 1
    name = f"on-fs-{'g' if gpu else 'c'}{S['counter']:03d}"
    folder = make_kernel(name, jobs, gpu, S["counter"])
    r = retry(a.kernels_push, folder)
    err = (getattr(r, "error", None) or "") if r is not None else "no response"
    if err:
        print(time.strftime("%m-%d %H:%M"), "push", name, "refused:", err.strip()[:100], flush=True)
        if gpu and "quota" in err.lower():
            S["gpu_blocked_until"] = time.time() + 2 * 3600
        return False
    S["running"][name] = {"scans": [j[0] for j in jobs], "gpu": gpu, "t": time.time()}
    print(time.strftime("%m-%d %H:%M"), "pushed", name, len(jobs), "scans", "(GPU)" if gpu else "(CPU)", flush=True)
    save_state(S)
    time.sleep(60)
    return True


S = load_state()
S["done"] = sorted(set(S["done"]) | done_from_results())
save_state(S)
print("start: done", len(S["done"]), "of", len(JOBS), "| running", list(S["running"]), flush=True)
t0 = time.time()
while time.time() - t0 < 21 * 24 * 3600:
    for name in list(S["running"]):
        st = status(name)
        if finished(st):
            print(time.strftime("%m-%d %H:%M"), name, st, flush=True)
            fetch(name, os.path.join(RES, name))
            S["done"] = sorted(set(S["done"]) | done_from_results())
            del S["running"][name]
            save_state(S)
            print(f"   done {len(S['done'])}/{len(JOBS)}", flush=True)
    if len(S["done"]) >= len(JOBS) and not S["running"]:
        break
    while sum(not k["gpu"] for k in S["running"].values()) < N_CPU:
        if not push(S, gpu=False):
            break
    if time.time() > S.get("gpu_blocked_until", 0):
        while sum(k["gpu"] for k in S["running"].values()) < N_GPU:
            if not push(S, gpu=True):
                break
    save_state(S)
    time.sleep(600)
print("watcher exit; done", len(S["done"]), flush=True)
