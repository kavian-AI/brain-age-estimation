"""Atlas v2 normative model: the published SHASH location-scale model (pilot/shash.py) plus
switchable terms, each motivated by a check on this pool or by another atlas:

  location   sexcurve : sex x natural spline(age, 3 df)   male-female gap changes with age
                         (held-out: men's tissue z 0.2-0.37 SD low after 60 without it)
             ageicv   : (age-60)/10 x log ICV            head-size exponent changes with age
                         (temporal horn, third ventricle: b 0.8 young -> 1.4-1.6 after 40)
  scale      scale_sex: sex in log sigma                 spread differs by sex (CentileBrain fits sexes apart)
             scale_icv: log ICV, log ICV^2 in log sigma  spread at extreme head sizes
  sites      shrink   : empirical-Bayes ridge on scanner offsets (mean and spread), variance taken
                        from sites with n >= 100 - a fixed-variance version of BrainChart's
                        random(Study): small sites are pulled toward the typical scanner.
Everything else (age splines, power-law head size, site-specific spread, SHASH nu/tau, z, centiles)
is unchanged. Imports pilot/ modules; caller must put pilot/ on sys.path first."""
import numpy as np, patsy, torch
import p3_pooled as P
import shash
from shash import _sinh_q, Z1


class AtlasModel(shash.SHASH):
    def __init__(self, d, ly, sexcurve=False, ageicv=False, scale_sex=False, scale_icv=False, shrink=False, **kw):
        self.sexcurve, self.ageicv, self.scale_sex, self.scale_icv, self.shrink = sexcurve, ageicv, scale_sex, scale_icv, shrink
        if sexcurve:
            self._Bsx = patsy.dmatrix(f"cr(age, df=3, lower_bound={kw.get('lo', 18.0)}, upper_bound={kw.get('hi', 98.0)}) - 1",
                                      d, return_type="dataframe").design_info
        P.LS.__init__(self, d, ly, **kw)                       # Gaussian ML start, on the extended design
        self._fit_shash(d, ly)

    # ------------------------------------------------------------ design
    def _Xm(self, d, zero_site=False):
        cols = [P.LS._Xm(self, d, zero_site=zero_site)]
        if self.ageicv:
            cols.append((((d.age.values - 60) / 10) * d.licv.values)[:, None])
        if self.sexcurve:
            B = patsy.build_design_matrices([self._Bsx], d, return_type="dataframe")[0].values
            cols.append(B[:, 1:] * (d.sex.values - self.sx)[:, None])      # cr() spans the constant: drop 1
        return np.column_stack(cols)

    def _Xs(self, d, zero_site=False):
        cols = [P.LS._Xs(self, d, zero_site=zero_site)]
        if self.scale_sex:
            cols.append((d.sex.values - self.sx)[:, None])
        if self.scale_icv:
            cols.append(np.column_stack([d.licv.values, d.licv.values ** 2]))
        return np.column_stack(cols)

    def _site_slices(self):
        K = len(self.sites)
        if K < 2:
            return None, None
        nB, nBs = len(self.Bm.column_names), len(self.Bs.column_names)
        sm = slice(nB + 1 + self.G, nB + 1 + self.G + K - 1)
        ss = slice(nBs, nBs + K - 1) if self.site_sd else None
        return sm, ss

    # ------------------------------------------------------------ fit
    def _fit_shash(self, d, ly):
        Xm = torch.tensor(self._Xm(d), dtype=torch.float64)
        Xs = torch.tensor(self._Xs(d), dtype=torch.float64)
        y = torch.tensor(np.asarray(ly, float), dtype=torch.float64)
        bm = torch.tensor(self.bm, dtype=torch.float64, requires_grad=True)
        bl = torch.tensor(self.bs / 2.0, dtype=torch.float64, requires_grad=True)
        nu = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        lt = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        pen = self._eb_penalty(d) if self.shrink else None

        def nll():
            out = -self._logpdf(Xm, Xs, y, bm, bl, nu, lt).sum()
            if pen is not None:
                sm, ss, im, is_ = pen
                cm = bm[sm]; offm = torch.cat([-cm.sum().reshape(1), cm])
                out = out + (offm ** 2).sum() * im
                if ss is not None:
                    cs = bl[ss]; offs = torch.cat([-cs.sum().reshape(1), cs])
                    out = out + (offs ** 2).sum() * is_
            return out
        opt = torch.optim.LBFGS([bm, bl, nu, lt], max_iter=400, tolerance_grad=1e-9, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); loss = nll(); loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.loglik = float(self._logpdf(Xm, Xs, y, bm, bl, nu, lt).sum())    # unpenalised log-likelihood
        self.bm = bm.detach().numpy().copy()
        self.bs = 2.0 * bl.detach().numpy()
        self.nu = float(nu.detach()); self.tau = float(torch.exp(lt.detach()))
        self.c_sd = (_sinh_q(Z1, self.nu, self.tau) - _sinh_q(-Z1, self.nu, self.tau)) / 2.0
        self.k_params += 2

    def _eb_penalty(self, d):
        """1/(2 tau^2) for location and log-sigma scanner offsets; tau from sites with n >= 100."""
        sm, ss = self._site_slices()
        if sm is None:
            return None
        n = d.site.value_counts().reindex(self.sites).fillna(0).values
        big = n >= 100
        if big.sum() < 5:
            self.eb = None; return None
        offm = np.r_[-self.bm[sm].sum(), self.bm[sm]]
        s2 = float(np.median(np.exp(np.clip(self._Xs(d) @ self.bs, -16, 4))))
        tm2 = max(np.var(offm[big], ddof=1) - np.mean(s2 / n[big]), 1e-6)
        if ss is not None:
            offs = np.r_[-self.bs[ss].sum(), self.bs[ss]] / 2.0
            ts2 = max(np.var(offs[big], ddof=1) - np.mean(1.0 / (2 * n[big])), 1e-6)
        else:
            ts2 = None
        self.eb = {"tau_loc_pct": 100 * np.sqrt(tm2), "tau_logsd": None if ts2 is None else np.sqrt(ts2),
                   "n_big_sites": int(big.sum())}
        return sm, ss, 1.0 / (2 * tm2), (None if ts2 is None else 1.0 / (2 * ts2))

    # ------------------------------------------------------------ scoring helpers
    def logpdf_np(self, d, ly):
        """held-out log density of log volume (proper scoring rule)."""
        loc, scale = self._loc_scale(d)
        z0 = (np.asarray(ly, float) - loc) / scale
        a = self.tau * np.arcsinh(z0) - self.nu
        return (np.log(self.tau) - np.log(scale) + np.logaddexp(a, -a) - np.log(2) - 0.5 * np.log1p(z0 ** 2)
                - 0.5 * np.sinh(a) ** 2 - 0.5 * np.log(2 * np.pi))

    def z_unseen_site(self, d, ly):
        """z0 and z for people on a scanner the model never saw (site effects set to zero)."""
        loc, scale = self._loc_scale(d, zero_site=True)
        z0 = (np.asarray(ly, float) - loc) / scale
        return z0, np.sinh(self.tau * np.arcsinh(z0) - self.nu)
