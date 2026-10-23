# Which model family describes healthy brain volumes best?
#
# Same data (pooled healthy reference, 1,471 people, 7 sites), same inputs (age, sex,
# log head size, site), same 5 subject folds (stratified by site), same scoring on
# held-out people, for six families:
#   GAUSS  location-scale Gaussian, splines (our current model; exact ML)
#   SHASH  location-scale + skewness + tail weight (sinh-arcsinh; ML)
#   QR     linear quantile regression on the same spline design (distribution-free)
#   MLP    heteroscedastic neural network (mean + log SD heads), 5-net ensemble
#   GBM    gradient-boosted trees: mean, then variance on squared residuals (LightGBM)
#   GPR    Gaussian process on age (classic normative model), homoscedastic
# Scores (held-out):
#   CRPS      proper score for the whole predicted distribution (lower = better), %
#   NLL       negative log-likelihood per person (density models), lower = better
#   cal_err   mean |observed - nominal| over centiles 1..99 (0 = perfect)
#   worst_site_cal  largest cal_err among the 7 sites
#   cov95     share inside the predicted 2.5-97.5 % band (target 0.95)
#   slope70_sd  across-fold SD of the implied aging rate at 70 (%/yr): stability
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import patsy
import torch
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3_pooled as P  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p4_out")
os.makedirs(OUT, exist_ok=True)
QS = np.array([0.01, 0.025] + list(np.arange(0.05, 0.951, 0.05)) + [0.975, 0.99])
REGIONS = ["hippocampus", "entorhinal", "thalamus", "lateral ventricles", "inf. lat. ventricles",
           "putamen", "brainstem", "precuneus"]
torch.set_num_threads(4)


# ------------------------------------------------------------------ shared design
class Design:
    def __init__(self, d):
        self.sites = sorted(d.site.unique())
        self.Bi = patsy.dmatrix("cr(age, df=5, lower_bound=18, upper_bound=98) - 1", d).design_info
        self.Si = patsy.dmatrix("cr(age, df=3, lower_bound=18, upper_bound=98) - 1", d).design_info
        self.sx = d.sex.mean()

    def site(self, d):
        return np.column_stack([(d.site.values == s).astype(float) for s in self.sites[1:]])

    def mean_X(self, d):
        B = patsy.build_design_matrices([self.Bi], d)[0]
        return np.column_stack([B, d.sex.values - self.sx, d.licv.values, self.site(d),
                                ((d.age.values - 60) / 10) * (d.sex.values - self.sx)])

    def sd_X(self, d):
        S = patsy.build_design_matrices([self.Si], d)[0]
        return np.column_stack([S, self.site(d)])


def ref_grid(d, ages, site):
    return pd.DataFrame({"age": ages, "sex": d.sex.mean(), "licv": 0.0, "site": site})


# ------------------------------------------------------------------ models
class Gauss:
    name = "GAUSS"

    def fit(self, d, y):
        self.m = P.LS(d, y, agesex=True, site_sd=True)
        return self

    def dist(self, d):
        mu = self.m._Xm(d) @ self.m.bm
        sd = np.sqrt(np.exp(self.m._Xs(d) @ self.m.bs))
        return mu, sd

    def quantiles(self, d):
        mu, sd = self.dist(d)
        return mu[:, None] + sd[:, None] * stats.norm.ppf(QS)[None, :]

    def nll(self, d, y):
        mu, sd = self.dist(d)
        return -stats.norm.logpdf(y, mu, sd)

    def median_curve(self, d, ages):
        g = ref_grid(d, ages, self.m.sites[0])
        return self.m._Xm(g, zero_site=True) @ self.m.bm


class Shash:
    """y = mu + sigma * sinh((asinh(Z) + nu) / tau),  Z ~ N(0,1)."""
    name = "SHASH"

    def fit(self, d, y):
        g = Gauss().fit(d, y)
        self.D = Design(d)
        Xm, Xs = self.D.mean_X(d), self.D.sd_X(d)
        Xm_t, Xs_t, y_t = map(lambda a: torch.tensor(a, dtype=torch.float64), (Xm, Xs, y))
        b0 = np.linalg.lstsq(Xm, y, rcond=None)[0]
        e = y - Xm @ b0
        s0 = np.linalg.lstsq(Xs, np.log(e ** 2 + 1e-12) + 1.27, rcond=None)[0] / 2
        self.bm = torch.tensor(b0, requires_grad=True)
        self.bs = torch.tensor(s0, requires_grad=True)
        self.nu = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        self.lt = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([self.bm, self.bs, self.nu, self.lt], max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = -self._logpdf(Xm_t, Xs_t, y_t).mean()
            loss.backward()
            return loss
        opt.step(closure)
        return self

    def _logpdf(self, Xm, Xs, y):
        mu = Xm @ self.bm
        ls = Xs @ self.bs
        tau = torch.exp(self.lt)
        z0 = (y - mu) / torch.exp(ls)
        a = tau * torch.asinh(z0) - self.nu
        return (torch.log(tau) - ls + torch.log(torch.cosh(a)) - 0.5 * torch.log1p(z0 ** 2)
                - 0.5 * torch.sinh(a) ** 2 - 0.5 * np.log(2 * np.pi))

    def _params(self, d):
        Xm, Xs = self.D.mean_X(d), self.D.sd_X(d)
        with torch.no_grad():
            return (Xm @ self.bm.numpy(), np.exp(Xs @ self.bs.numpy()), float(self.nu), float(torch.exp(self.lt)))

    def quantiles(self, d):
        mu, sg, nu, tau = self._params(d)
        zq = stats.norm.ppf(QS)
        return mu[:, None] + sg[:, None] * np.sinh((np.arcsinh(zq)[None, :] + nu) / tau)

    def nll(self, d, y):
        Xm, Xs = self.D.mean_X(d), self.D.sd_X(d)
        with torch.no_grad():
            return -self._logpdf(torch.tensor(Xm), torch.tensor(Xs), torch.tensor(y)).numpy()

    def median_curve(self, d, ages):
        g = ref_grid(d, ages, self.D.sites[0])
        Xm = self.D.mean_X(g)
        Xm[:, 5 + 2:5 + 2 + len(self.D.sites) - 1] = 0      # average site (treatment coding -> approx)
        mu, sg, nu, tau = self._params(g)
        return Xm @ self.bm.detach().numpy() + sg * np.sinh(nu / tau)


class QR:
    name = "QR"

    def fit(self, d, y):
        from statsmodels.regression.quantile_regression import QuantReg
        self.D = Design(d)
        X = np.column_stack([np.ones(len(d)), self.D.mean_X(d)])
        self.B = [QuantReg(y, X).fit(q=q, max_iter=5000).params for q in QS]
        return self

    def quantiles(self, d):
        X = np.column_stack([np.ones(len(d)), self.D.mean_X(d)])
        return np.sort(np.column_stack([X @ b for b in self.B]), axis=1)

    nll = None

    def median_curve(self, d, ages):
        g = ref_grid(d, ages, self.D.sites[0])
        X = np.column_stack([np.ones(len(g)), self.D.mean_X(g)])
        return X @ self.B[list(QS).index(0.5) if 0.5 in QS else len(QS) // 2]


class MLP:
    """Heteroscedastic MLP: inputs (age, sex, log ICV, site one-hot) -> (mu, log sd); NLL loss."""
    name = "MLP"

    def _X(self, d):
        S = np.column_stack([(d.site.values == s).astype(float) for s in self.sites])
        return np.column_stack([(d.age.values - 55) / 20, d.sex.values - 0.5, d.licv.values / 0.1, S]).astype(np.float32)

    def fit(self, d, y, n_ens=5, epochs=600):
        self.sites = sorted(d.site.unique())
        self.ym, self.ys = y.mean(), y.std()
        X = torch.tensor(self._X(d))
        t = torch.tensor(((y - self.ym) / self.ys).astype(np.float32))
        self.nets = []
        rng = np.random.default_rng(0)
        for k in range(n_ens):
            torch.manual_seed(k)
            net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.SiLU(),
                                      torch.nn.Linear(32, 32), torch.nn.SiLU(), torch.nn.Linear(32, 2))
            opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
            idx = rng.permutation(len(d))
            va, tr = idx[:len(d) // 8], idx[len(d) // 8:]
            best, best_state, bad = 1e9, None, 0
            for ep in range(epochs):
                net.train()
                opt.zero_grad()
                o = net(X[tr])
                loss = (0.5 * ((t[tr] - o[:, 0]) / torch.exp(o[:, 1])) ** 2 + o[:, 1]).mean()
                loss.backward()
                opt.step()
                net.eval()
                with torch.no_grad():
                    o = net(X[va])
                    vl = float((0.5 * ((t[va] - o[:, 0]) / torch.exp(o[:, 1])) ** 2 + o[:, 1]).mean())
                if vl < best - 1e-4:
                    best, best_state, bad = vl, {k2: v.clone() for k2, v in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad > 60:
                        break
            net.load_state_dict(best_state)
            self.nets.append(net)
        return self

    def dist(self, d):
        X = torch.tensor(self._X(d))
        with torch.no_grad():
            O = np.stack([n(X).numpy() for n in self.nets])        # ens x n x 2
        mus = O[:, :, 0] * self.ys + self.ym
        sds = np.exp(O[:, :, 1]) * self.ys
        mu = mus.mean(0)
        sd = np.sqrt((sds ** 2).mean(0) + mus.var(0))
        return mu, sd

    def quantiles(self, d):
        mu, sd = self.dist(d)
        return mu[:, None] + sd[:, None] * stats.norm.ppf(QS)[None, :]

    def nll(self, d, y):
        mu, sd = self.dist(d)
        return -stats.norm.logpdf(y, mu, sd)

    def median_curve(self, d, ages):
        curves = []
        for s in self.sites:                                        # average over sites
            curves.append(self.dist(ref_grid(d, ages, s))[0])
        return np.mean(curves, axis=0)


class GBM:
    name = "GBM"

    def _X(self, d):
        return pd.DataFrame({"age": d.age.values, "sex": d.sex.values, "licv": d.licv.values,
                             "site": pd.Categorical(d.site.values, categories=self.sites)})

    def fit(self, d, y):
        import lightgbm as lgb
        self.sites = sorted(d.site.unique())
        p = dict(n_estimators=400, learning_rate=0.03, num_leaves=8, min_child_samples=30, subsample=0.8,
                 subsample_freq=1, verbose=-1)
        X = self._X(d)
        # out-of-fold residuals for the variance model (avoid in-sample shrinkage)
        oof = np.zeros(len(d))
        f = np.arange(len(d)) % 5
        for k in range(5):
            m = lgb.LGBMRegressor(**p).fit(X[f != k], y[f != k])
            oof[f == k] = m.predict(X[f == k])
        self.mm = lgb.LGBMRegressor(**p).fit(X, y)
        self.mv = lgb.LGBMRegressor(**{**p, "n_estimators": 200}).fit(X, np.log((y - oof) ** 2 + 1e-8))
        e = y - oof
        lv = self.mv.predict(X)
        self.c = np.log(np.mean(e ** 2 / np.exp(lv)))               # rescale variance
        return self

    def dist(self, d):
        X = self._X(d)
        return self.mm.predict(X), np.sqrt(np.exp(self.mv.predict(X) + self.c))

    quantiles = MLP.quantiles
    nll = MLP.nll
    median_curve = MLP.median_curve


class GPR:
    name = "GPR"

    def fit(self, d, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        self.D = Design(d)
        L = np.column_stack([np.ones(len(d)), d.sex.values - self.D.sx, d.licv.values, self.D.site(d)])
        self.bl = np.linalg.lstsq(L, y, rcond=None)[0]
        r = y - L @ self.bl
        k = ConstantKernel(0.01) * RBF(20.0, (3, 200)) + WhiteKernel(0.005)
        self.g = GaussianProcessRegressor(k, normalize_y=True).fit(d.age.values[:, None], r)
        return self

    def dist(self, d):
        L = np.column_stack([np.ones(len(d)), d.sex.values - self.D.sx, d.licv.values, self.D.site(d)])
        m, s = self.g.predict(d.age.values[:, None], return_std=True)
        return L @ self.bl + m, s

    quantiles = MLP.quantiles
    nll = MLP.nll

    def median_curve(self, d, ages):
        g = ref_grid(d, ages, self.D.sites[0])
        return self.dist(g)[0]


MODELS = [Gauss, Shash, QR, MLP, GBM, GPR]


# ------------------------------------------------------------------ evaluation
def evaluate():
    cross, D, O = P.assemble()
    subj = cross.subject.unique()
    rng = np.random.default_rng(0)
    fold_of = {}
    for s, g in cross.groupby("site"):                              # stratified by site
        u = g.subject.unique()
        rng.shuffle(u)
        fold_of.update({x: i % 5 for i, x in enumerate(u)})
    rows, curves = [], []
    ages = np.array([30.0, 50.0, 69.5, 70.5, 85.0])
    for r in REGIONS:
        d, y = P.region_data(cross, r)
        f = d.subject.map(fold_of).values
        for M in MODELS:
            t0 = time.time()
            Qh = np.zeros((len(d), len(QS)))
            nll = np.full(len(d), np.nan)
            slopes = []
            for k in range(5):
                tr, te = f != k, f == k
                dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
                m = M().fit(dtr, y[tr])
                Qh[te] = m.quantiles(dte)
                if M.nll is not None:
                    nll[te] = m.nll(dte, y[te])
                c = m.median_curve(dtr, ages)
                slopes.append(100 * (c[3] - c[2]))
            pin = np.mean([np.mean(np.maximum(q * (y - Qh[:, j]), (q - 1) * (y - Qh[:, j])))
                           for j, q in enumerate(QS)])
            below = (y[:, None] <= Qh).mean(0)
            cal = np.mean(np.abs(below - QS))
            site_cal = max(np.mean(np.abs((y[d.site.values == s][:, None] <= Qh[d.site.values == s]).mean(0) - QS))
                           for s in d.site.unique())
            band = (d.age.values < 40, (d.age.values >= 40) & (d.age.values < 65), d.age.values >= 65)
            band_cal = max(np.mean(np.abs((y[b][:, None] <= Qh[b]).mean(0) - QS)) for b in band)
            cov95 = np.mean((y >= Qh[:, 1]) & (y <= Qh[:, -2]))
            rows.append({"region": r, "model": M.name, "CRPS_pct": 100 * 2 * pin, "NLL": np.nanmean(nll),
                         "cal_err": cal, "worst_site_cal": site_cal, "worst_ageband_cal": band_cal,
                         "cov95": cov95, "slope70": np.mean(slopes), "slope70_sd": np.std(slopes),
                         "seconds": time.time() - t0})
            print(f"  {r:22s} {M.name:6s} CRPS {100*2*pin:.3f}%  NLL {np.nanmean(nll):+.3f}  cal {cal:.3f}  "
                  f"site {site_cal:.3f}  age {band_cal:.3f}  cov95 {cov95:.3f}  slope70 {np.mean(slopes):+.2f}"
                  f"±{np.std(slopes):.2f}  ({time.time()-t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
    R["rank_CRPS"] = R.groupby("region").CRPS_pct.rank()
    print("\n== mean over regions ==")
    print(R.groupby("model")[["CRPS_pct", "NLL", "cal_err", "worst_site_cal", "worst_ageband_cal", "cov95",
                              "slope70_sd", "rank_CRPS", "seconds"]].mean().sort_values("CRPS_pct").round(4).to_string())


if __name__ == "__main__":
    evaluate()
