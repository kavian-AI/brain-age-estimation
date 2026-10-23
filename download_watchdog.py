#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Keep the two Kaggle downloads alive.

They died silently once already when the session was disrupted, leaving partial
zips with no running process. This polls file size and restarts any download
that has stalled for STALL_MIN minutes, then exits once both are complete.
"""

import os
import subprocess
import time
from datetime import datetime

BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
TOKEN = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()

JOBS = [
    dict(name="ADNI", slug="fabriziopacheco/adni-mri-tensors",
         d=os.path.join(BASE, "adni_img"), zip="adni-mri-tensors.zip",
         target=5_930_630_765),
    dict(name="OASIS", slug="ninadaithal/imagesoasis",
         d=os.path.join(BASE, "images"), zip="imagesoasis.zip",
         target=1_322_017_985),
]
POLL = 60
STALL_MIN = 4


def size(j):
    p = os.path.join(j["d"], j["zip"])
    return os.path.getsize(p) if os.path.exists(p) else 0


def partial(j):
    return os.path.exists(os.path.join(j["d"], j["zip"] + ".kaggle-partial"))


def done(j):
    return size(j) >= j["target"] * 0.995 and not partial(j)


def launch(j):
    env = dict(os.environ, KAGGLE_API_TOKEN=TOKEN)
    log = open(os.path.join(j["d"], f"{j['name']}_dl.log"), "ab")
    return subprocess.Popen(
        ["python", "-m", "kaggle", "datasets", "download", "-d", j["slug"],
         "-p", ".", "-o"],
        cwd=j["d"], env=env, stdout=log, stderr=log,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def main():
    last = {j["name"]: (size(j), time.time()) for j in JOBS}
    procs = {}
    while True:
        alive = 0
        for j in JOBS:
            n = j["name"]
            if done(j):
                continue
            alive += 1
            s = size(j)
            prev, t0 = last[n]
            if s > prev:
                last[n] = (s, time.time())
            elif time.time() - t0 > STALL_MIN * 60:
                p = procs.get(n)
                if p and p.poll() is None:
                    p.kill()
                print(f"[{datetime.now():%H:%M:%S}] {n} stalled at "
                      f"{s/1e6:.0f} MB - restarting", flush=True)
                procs[n] = launch(j)
                last[n] = (s, time.time())
        stamp = datetime.now().strftime("%H:%M:%S")
        line = "  ".join(
            f"{j['name']} {size(j)/1e6:6.0f}/{j['target']/1e6:.0f} MB"
            f"{' DONE' if done(j) else ''}" for j in JOBS)
        print(f"[{stamp}] {line}", flush=True)
        if alive == 0:
            print("both downloads complete")
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
