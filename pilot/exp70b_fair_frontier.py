# E70b: the disease-aware brain age against the BEST-TUNED standard model, not alpha=10.
# For every penalty, gamma=0 (standard) vs relevance-weighted gammas; the question is
# whether the frontier moves beyond what tuning the standard penalty alone reaches.
import os, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "pilot")
import exp53_units_and_dimension as E, exp62_rich_atlas as R, exp70_frontier as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
a, m, T, names = R.build("X")
cn = m.DX.values == "CN"; Z, _ = R.normative(m, names, cn)
y = (m.DX.values == "Dementia").astype(int); sel = m.DX.isin(["CN", "Dementia"]).values
half = np.random.RandomState(0).rand(len(m)) < 0.5; A_, B_ = sel & half, sel & ~half
ZcA, ZdA = Z[A_ & (y == 0)], Z[A_ & (y == 1)]
mu = ZdA.mean(0) - ZcA.mean(0); sdA = np.sqrt(0.5 * (ZcA.var(0, ddof=1) + ZdA.var(0, ddof=1)))
pi = np.abs(mu / sdA); pi = np.r_[pi / pi.mean(), 1.0, 1.0]
pool = m[m.DX == "MCI"]; keep, dur, ev = E.survival(a, pool.PTID.values, ["Dementia"])
idx = pool.index[pool.PTID.isin(set(keep))].values; o = {x: i for i, x in enumerate(keep)}
ii = np.array([o[x] for x in m.PTID.values[idx]]); dur, ev = dur[ii], ev[ii]; agC = m.age_visit.values[idx]
conv = lambda g: np.nanmean([R.cidx(np.column_stack([g[idx], agC]), dur, ev, s) for s in range(3)])
res = []
print(f"{'alpha':>8s} {'gamma':>6s} {'MAE':>6s} {'AD/CN(B)':>9s} {'conv C':>7s}")
for al in [31.6, 100, 316, 1000, 3162]:
    for gm in [0, 2, 4, 6, 8]:
        g, mae = F.gap_model(m, names, cn, al, scale=pi ** (gm / 2))
        res.append((al, gm, mae, roc_auc_score(y[B_], g[B_]), conv(g), g))
        print(f"{al:8.1f} {gm:6d} {mae:6.2f} {res[-1][3]:9.3f} {res[-1][4]:7.3f}")
std = [r for r in res if r[1] == 0]; aw = [r for r in res if r[1] > 0]
bs = max(std, key=lambda r: r[3]); ba = max(aw, key=lambda r: r[3])
print(f"\nbest STANDARD (any alpha):       alpha {bs[0]}, MAE {bs[2]:.2f}, AD/CN {bs[3]:.3f}, conv {bs[4]:.3f}")
print(f"best DISEASE-AWARE (any alpha):  alpha {ba[0]}, gamma {ba[1]}, MAE {ba[2]:.2f}, AD/CN {ba[3]:.3f}, conv {ba[4]:.3f}")
rng = np.random.RandomState(0); ib = np.where(B_)[0]; d1, d2 = [], []
for _ in range(1000):
    j = rng.choice(ib, len(ib))
    if y[j].sum() >= 10: d1.append(roc_auc_score(y[j], ba[5][j]) - roc_auc_score(y[j], bs[5][j]))
for _ in range(300):
    i = rng.randint(0, len(dur), len(dur))
    if ev[i].sum() < 20: continue
    p1 = R.cidx(np.column_stack([ba[5][idx][i], agC[i]]), dur[i], ev[i], 0); p2 = R.cidx(np.column_stack([bs[5][idx][i], agC[i]]), dur[i], ev[i], 0)
    if np.isfinite(p1) and np.isfinite(p2): d2.append(p1 - p2)
R.show("AD/CN: best disease-aware minus best standard", d1)
R.show("conversion: best disease-aware minus best standard", d2)
