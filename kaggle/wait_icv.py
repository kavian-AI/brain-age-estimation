# Waits for the four SynthStrip ICV kernels and fetches their JSONL into icv_results/.
import os, time
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "watch_lifespan.py"), encoding="utf-8").read()
exec(compile(src.split("state = {")[0], "w", "exec"))
left = ["icv-synthstrip-ixi", "icv-synthstrip-oasis", "icv-synthstrip-dlbs0", "icv-synthstrip-dlbs1"]
t0 = time.time()
while left and time.time() - t0 < 13 * 3600:
    for k in list(left):
        s = status(k)
        if finished(s):
            print(time.strftime("%H:%M"), k, s, flush=True)
            fetch(k, os.path.join(KAG, "icv_results"))
            left.remove(k)
    time.sleep(300)
print("done; unfinished:", left, flush=True)
