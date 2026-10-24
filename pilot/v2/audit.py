"""Residual audit: can a flexible learner predict a healthy person's held-out z (or z^2) from age, sex,
head size and scanner? If it can, the normative model is missing a term.
  test               atlas v2
  real controls      published atlas, regions WITHOUT an age x sex term (known 0.2-0.37 SD bias after 60)
  planted controls   v2 z with a known bias added for men >= 60 (0.1 / 0.2 / 0.3 SD) -> measures sensitivity
Learners are heavily regularised so they predict 'no effect' when there is none; significance is a paired
one-sided t-test of per-person squared-error improvement over the constant prediction."""
import os, sys, numpy as np, pandas as pd, warnings
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
pub = pd.read_csv(os.path.join(HERE, "published_heldout_z.csv.gz")).rename(columns={"region": "feature"})
v2 = pd.read_csv(os.path.join(HERE, "v2_heldout_z.csv.gz"))
FEATS = ["hippocampus", "amygdala", "entorhinal", "thalamus", "putamen", "lateral ventricles", "inf. lat. ventricles",
         "third ventricle", "brainstem", "cerebellum cortex", "cortex", "cerebral WM", "total brain"]
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
if QUICK: FEATS = ["total brain", "hippocampus"]

def audit(g, target, learner):
    sites = sorted(g.site.unique())
    hs = g.licv.values if "licv" in g else np.log(g.icv.values)
    X = np.column_stack([g.age.values, g.sex.values, hs]); y = target(g.z.values)
    pred, base = np.zeros(len(g)), np.zeros(len(g))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        base[te] = y[tr].mean()
        if learner == "gbm":
            Xa = np.column_stack([X, pd.Categorical(g.site.values, categories=sites).codes])
            m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=200,
                                              l2_regularization=1.0, categorical_features=[3], early_stopping=True,
                                              validation_fraction=0.2, n_iter_no_change=30, random_state=0)
            m.fit(Xa[tr], y[tr]); pred[te] = m.predict(Xa[te])
        else:
            S = (g.site.values[:, None] == np.array(sites)[None, :]).astype(float)
            sc = StandardScaler().fit(X[tr]); Xa = np.column_stack([sc.transform(X), S])
            m = MLPRegressor(hidden_layer_sizes=(32, 32), alpha=1.0, early_stopping=True, validation_fraction=0.2,
                             n_iter_no_change=20, max_iter=400, random_state=0)
            m.fit(Xa[tr], y[tr]); pred[te] = base[te] + 0.0 + (m.predict(Xa[te]) - y[tr].mean()) if False else m.predict(Xa[te])
    d = (y - base) ** 2 - (y - pred) ** 2                          # per-person improvement over the constant
    r2 = d.sum() / ((y - base) ** 2).sum()
    p = stats.ttest_1samp(d, 0.0, alternative="greater").pvalue
    return r2, p

targets = {"z": lambda z: z, "z^2": lambda z: z ** 2}
rows = []
def run(label, Z, feats, transform=None):
    for f in feats:
        g = Z[Z.feature == f].dropna(subset=["z"]).reset_index(drop=True)
        if transform is not None: g = transform(g)
        for tname, t in targets.items():
            for learner in ("gbm", "mlp"):
                r2, p = audit(g, t, learner)
                rows.append({"set": label, "feature": f, "target": tname, "learner": learner, "R2": r2, "p": p})
        print(f"  audited [{label}] {f}", flush=True)

run("v2 (test)", v2, FEATS)
run("published (real control)", pub, [f for f in FEATS if f not in ("hippocampus", "thalamus")])   # these two already had age x sex
for delta in (0.1, 0.2, 0.3):
    plant = lambda g, d=delta: g.assign(z=g.z + d * ((g.sex == 1) & (g.age >= 60)))
    run(f"v2 + planted {delta} SD (men 60+)", v2, ["hippocampus", "cortex", "lateral ventricles"] if not QUICK else ["hippocampus"], plant)
A = pd.DataFrame(rows); A.to_csv(os.path.join(HERE, "audit_results.csv"), index=False)
A["found"] = A.p < 0.01
pd.set_option("display.width", 250)
print("\n=========== detection rate (p < 0.01, either learner) by set and target ===========")
best = A.groupby(["set", "feature", "target"]).agg(found=("found", "max"), R2=("R2", "max"), p=("p", "min")).reset_index()
print(best.groupby(["set", "target"]).agg(detected=("found", "sum"), of=("found", "size"), median_R2=("R2", "median"), max_R2=("R2", "max"))
      .round(4).to_string())
print("\n=========== every detection ===========")
hit = best[best.found].sort_values(["set", "feature"])
print(hit.round(4).to_string(index=False) if len(hit) else "  none")
