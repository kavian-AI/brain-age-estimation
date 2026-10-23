# Cloud run of pilot/p5_full_pool.py part B on the private dataset brainage-normative-inputs.
# Copies the project to /kaggle/working, runs the analysis, keeps only the result folder as output.
import glob, os, shutil, subprocess, sys
PART = "B"
src = glob.glob("/kaggle/input/**/pilot/p5_full_pool.py", recursive=True)
assert src, "dataset not mounted"
root = os.path.dirname(os.path.dirname(src[0]))
work = "/kaggle/working/proj"
shutil.copytree(root, work)
print("pip:", subprocess.run("pip install -q lifelines patsy statsmodels openpyxl lightgbm", shell=True).returncode, flush=True)
rc = subprocess.run([sys.executable, "-u", "pilot/p5_full_pool.py", PART], cwd=work).returncode
print("analysis rc", rc, flush=True)
fig = """
import os, sys, json, pandas as pd
sys.path.insert(0, 'pilot')
import p4_figures as F
F.D = os.path.join('pilot', 'p5_out')
if os.path.exists(os.path.join(F.D, 'everyage_table.csv')):
    F.T = pd.read_csv(os.path.join(F.D, 'everyage_table.csv'))
    F.meta = json.load(open(os.path.join(F.D, 'everyage_meta.json')))
    F.fig_charts(); F.fig_resolution()
if os.path.exists(os.path.join(F.D, 'model_comparison.csv')):
    F.M = pd.read_csv(os.path.join(F.D, 'model_comparison.csv')); F.fig_models()
print('figures done')
"""
print(subprocess.run([sys.executable, "-c", fig], cwd=work, capture_output=True, text=True), flush=True)
shutil.copytree(os.path.join(work, "pilot", "p5_out"), f"/kaggle/working/p5_out_{PART}")
shutil.rmtree(work)
