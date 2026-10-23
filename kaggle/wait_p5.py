# Waits for the cloud analysis kernels p5-cloud-a / p5-cloud-b and downloads their outputs
# (tables + figures only) into pilot/p5_out. Status in kaggle/_p5_wait.log.
import os, shutil, socket, time
socket.setdefaulttimeout(180)
from kaggle.api.kaggle_api_extended import KaggleApi
os.environ["KAGGLE_API_TOKEN"] = os.path.expanduser("~/.kaggle/access_token")
a = KaggleApi(); a.authenticate()
KAG = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(os.path.dirname(KAG), "pilot", "p5_out")
import sys; left = sys.argv[1:] or ["p5-cloud-a", "p5-cloud-b"]
t0 = time.time()
while left and time.time() - t0 < 30 * 3600:
    for k in list(left):
        try:
            s = str(a.kernels_status("mohammadparsakavian/" + k))
        except Exception as e:
            print(time.strftime("%m-%d %H:%M"), k, "status error", type(e).__name__, flush=True); continue
        if any(x in s for x in ('"COMPLETE"', '"ERROR"', '"CANCEL')):
            tmp = os.path.join(KAG, "_fetch_" + k); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
            try:
                a.kernels_output("mohammadparsakavian/" + k, path=tmp, quiet=True)
            except Exception as e:
                print(time.strftime("%m-%d %H:%M"), k, "fetch error", type(e).__name__, flush=True); continue
            os.makedirs(DEST, exist_ok=True)
            n = 0
            for root, _, files in os.walk(tmp):
                for f in files:
                    dst = os.path.join(DEST, f if not f.endswith(".log") else k + ".log")
                    os.replace(os.path.join(root, f), dst); n += 1
            shutil.rmtree(tmp, ignore_errors=True)
            print(time.strftime("%m-%d %H:%M"), k, s, "->", n, "files", flush=True)
            left.remove(k)
    time.sleep(300)
print("done; unfinished:", left, flush=True)

