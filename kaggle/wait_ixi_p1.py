# Waits for ixi-lba-p1 and fetches its output (helpers from watch_lifespan.py).
import os, time
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "watch_lifespan.py"), encoding="utf-8").read()
exec(compile(src.split("state = {")[0], "w", "exec"))
t0 = time.time()
while time.time() - t0 < 6 * 3600:
    s = status("ixi-lba-p1")
    print(time.strftime("%H:%M"), s, flush=True)
    if finished(s):
        fetch("ixi-lba-p1", os.path.join(KAG, "ixi_lba_results"))
        break
    time.sleep(300)
