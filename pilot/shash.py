# SHASH location-scale model: the pooled normative model with skewness and tail weight.
#
# Same structure as p3_pooled.LS (age spline + sex + head-size power law + scanner offset for
# the location; age spline + scanner offset for the log scale), but healthy log volumes are
# sinh-arcsinh distributed (Jones & Pewsey 2009) instead of Gaussian:
#     log y = mu + sigma * sinh( (asinh(Z) + nu) / tau ),   Z ~ N(0, 1)
#     nu  > 0 / < 0 : right / left skew          tau < 1 / > 1 : heavier / lighter tails than Gaussian
# nu and tau are one number per region (age-constant). Fitted by maximum likelihood (L-BFGS),
# starting from the Gaussian fit. Everything the pipeline uses keeps its meaning:
#   z(d, y)        normal deviate of a person  ->  centile = Phi(z)     (exact under SHASH)
#   mu(ages)       MEDIAN log volume of the reference person (so slope = aging rate of the median)
#   sigma(ages)    SD-equivalent spread in %: half the distance between the 15.9th and 84.1th
#                  centiles, so resolution R = spread / |slope| keeps its meaning
#   quantile(...)  any centile
import numpy as np
import torch
from scipy import stats

import p3_pooled as P

torch.set_num_threads(4)
Z1 = stats.norm.ppf(0.8413447460685429)


def _sinh_q(z, nu, tau):
    return np.sinh((np.arcsinh(z) + nu) / tau)


class SHASH(P.LS):
    def __init__(self, d, ly, **kw):
        super().__init__(d, ly, **kw)                         # Gaussian ML fit = starting point
        Xm = torch.tensor(self._Xm(d), dtype=torch.float64)
        Xs = torch.tensor(self._Xs(d), dtype=torch.float64)
        y = torch.tensor(np.asarray(ly, float), dtype=torch.float64)
        bm = torch.tensor(self.bm, dtype=torch.float64, requires_grad=True)
        bl = torch.tensor(self.bs / 2.0, dtype=torch.float64, requires_grad=True)   # log-sigma coefficients
        nu = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        lt = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([bm, bl, nu, lt], max_iter=400, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

        def nll():
            return -self._logpdf(Xm, Xs, y, bm, bl, nu, lt).sum()

        def closure():
            opt.zero_grad()
            loss = nll()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            self.loglik = float(-nll())
        self.bm = bm.detach().numpy().copy()
        self.bs = 2.0 * bl.detach().numpy()                     # keep LS convention: log-variance scale
        self.nu = float(nu.detach())
        self.tau = float(torch.exp(lt.detach()))
        self.c_sd = (_sinh_q(Z1, self.nu, self.tau) - _sinh_q(-Z1, self.nu, self.tau)) / 2.0
        self.k_params += 2

    @staticmethod
    def _logpdf(Xm, Xs, y, bm, bl, nu, lt):
        mu = Xm @ bm
        ls = torch.clamp(Xs @ bl, -8.0, 2.0)
        tau = torch.exp(lt)
        z0 = (y - mu) / torch.exp(ls)
        a = tau * torch.asinh(z0) - nu
        logcosh = torch.logaddexp(a, -a) - np.log(2.0)
        return (lt - ls + logcosh - 0.5 * torch.log1p(z0 ** 2) - 0.5 * torch.sinh(a) ** 2
                - 0.5 * np.log(2 * np.pi))

    # ---- location / scale helpers
    def _loc_scale(self, d, zero_site=False):
        loc = self._Xm(d, zero_site=zero_site) @ self.bm
        scale = np.sqrt(np.exp(np.clip(self._Xs(d, zero_site=zero_site) @ self.bs, -16, 4)))
        return loc, scale

    def mu(self, ages):                                       # median of the reference person
        g = self._grid(ages)
        loc, scale = self._loc_scale(g, zero_site=True)
        return loc + scale * np.sinh(self.nu / self.tau)

    def sigma(self, ages, site=None):                         # SD-equivalent spread, %
        return super().sigma(ages, site) * self.c_sd

    def z(self, d, ly):
        loc, scale = self._loc_scale(d)
        z0 = (np.asarray(ly, float) - loc) / scale
        return np.sinh(self.tau * np.arcsinh(z0) - self.nu)

    def quantile_from(self, loc, scale, p):
        """Centile p for given location / scale (arrays): loc + scale * sinh((asinh(z_p)+nu)/tau)."""
        return loc + scale * _sinh_q(stats.norm.ppf(p), self.nu, self.tau)
