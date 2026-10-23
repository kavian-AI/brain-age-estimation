# Common vs scanner-specific spread, with the ML spread fit: LR tests + held-out calibration.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import p3_pooled as P
cross, D, O = P.assemble()
rows = []
for r in P.REGIONS:
    d, ly = P.region_data(cross, r)
    a = P.LS(d, ly, site_sd=False, agesex=(r in P.AGESEX)); b = P.LS(d, ly, site_sd=True, agesex=(r in P.AGESEX))
    lr = 2 * (b.loglik - a.loglik); df = b.k_params - a.k_params
    rows.append({"region": r, "LR": lr, "df": df, "p": stats.chi2.sf(max(lr, 0), df),
                 "dAIC_site_minus_common": (-2*b.loglik+2*b.k_params) - (-2*a.loglik+2*a.k_params),
                 "dBIC_site_minus_common": (-2*b.loglik+b.k_params*np.log(len(d))) - (-2*a.loglik+a.k_params*np.log(len(d))),
                 "spread_ratio_max_min": max(b.sigma([70.], site=s)[0] for s in b.sites) / min(b.sigma([70.], site=s)[0] for s in b.sites)})
T = pd.DataFrame(rows); print(T.round(3).to_string(index=False))
print("Bonferroni-significant:", int((T.p < 0.05/len(T)).sum()), "of", len(T))
# held-out calibration by site, both models
folds = pd.Series(cross.subject.unique()).sample(frac=1, random_state=0).reset_index(drop=True)
fold_of = {s: i % 5 for i, s in enumerate(folds)}
out = []
for site_sd in (False, True):
    for r in P.REGIONS:
        d, ly = P.region_data(cross, r); f = d.subject.map(fold_of).values; z = np.full(len(d), np.nan)
        for k in range(5):
            tr, te = f != k, f == k
            m = P.LS(d[tr].reset_index(drop=True), ly[tr], site_sd=site_sd, agesex=(r in P.AGESEX))
            z[te] = m.z(d[te].reset_index(drop=True), ly[te])
        out.append(pd.DataFrame({"model": "site spread" if site_sd else "common spread", "site": d.site, "z": z}))
Z = pd.concat(out)
print(Z.groupby(["model", "site"]).z.agg(sd_z="std", within_196=lambda v: np.mean(np.abs(v) < 1.96)).round(3).unstack(0).to_string())
