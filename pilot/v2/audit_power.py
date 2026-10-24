import os, numpy as np, pandas as pd, warnings
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
v2 = pd.read_csv(os.path.join(HERE, "v2_heldout_z.csv.gz"))
g0 = v2[v2.feature == "hippocampus"].reset_index(drop=True)
X = np.column_stack([g0.age.values, g0.sex.values, g0.licv.values])
SET = {"gbm-A lr.1 leaf4 min300": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=4, min_samples_leaf=300,
                                                             early_stopping=True, validation_fraction=0.25, n_iter_no_change=50, tol=0, random_state=0),
       "gbm-B lr.05 leaf6 min150": lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=6, min_samples_leaf=150,
                                                              early_stopping=True, validation_fraction=0.25, n_iter_no_change=60, tol=0, random_state=0),
       "mlp 16x16 a1": lambda: MLPRegressor(hidden_layer_sizes=(16, 16), alpha=1.0, early_stopping=True, validation_fraction=0.25,
                                            n_iter_no_change=30, max_iter=500, random_state=0)}
def audit(y, mk, reps=3):
    D = np.zeros(len(y))
    for rep in range(reps):
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(X):
            sc = StandardScaler().fit(X[tr]); m = mk().fit(sc.transform(X[tr]), y[tr])
            D[te] += ((y[te] - y[tr].mean()) ** 2 - (y[te] - m.predict(sc.transform(X[te]))) ** 2) / reps
    return D.sum() / ((y - y.mean()) ** 2).sum(), stats.ttest_1samp(D, 0, alternative="greater").pvalue
men60 = ((g0.sex == 1) & (g0.age >= 60)).values
print(f"men 60+ = {men60.mean():.1%} of people")
for name, mk in SET.items():
    out = []
    for delta in (0.0, 0.1, 0.2, 0.3):
        r2, p = audit(g0.z.values + delta * men60, mk); out.append(f"{delta:.1f}SD: R2 {r2:+.4f} p {p:.3f}")
    print(f"  {name:<26} " + " | ".join(out))
