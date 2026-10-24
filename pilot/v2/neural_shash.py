"""Rematch: atlas v2 vs a neural-SHASH network (outputs mu, log sigma, nu, log tau; SHASH likelihood),
on identical held-out folds. Judged on held-out log-likelihood, calibration, sex-by-age gaps, aging rate at 70."""
import os, sys, time, numpy as np, pandas as pd, torch
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "pilot")); sys.path.insert(0, HERE)
import v2_build as B                               # same pool loader, same fit() and spec as the final atlas
import p3_pooled as P
torch.set_num_threads(4); T0 = time.time()
REGIONS = ["hippocampus", "entorhinal", "thalamus", "putamen", "brainstem", "lateral ventricles", "inf. lat. ventricles", "precuneus"]
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
if QUICK: REGIONS = REGIONS[:1]
N_ENS, HID = (1, 64) if QUICK else (3, 64)

def shash_logpdf_t(y, mu, ls, nu, lt):
    tau = torch.exp(lt); z0 = (y - mu) / torch.exp(ls); a = tau * torch.asinh(z0) - nu
    return lt - ls + torch.logaddexp(a, -a) - np.log(2.0) - 0.5 * torch.log1p(z0 ** 2) - 0.5 * torch.sinh(a) ** 2 - 0.5 * np.log(2 * np.pi)
def shash_cdf(y, mu, s, nu, tau): return stats.norm.cdf(np.sinh(tau * np.arcsinh((y - mu) / s) - nu))

class Net:
    """ensemble of MLPs; inputs: standardized age, sex, standardized log ICV, scanner one-hot."""
    def __init__(self, d, y, seed0):
        self.sites = sorted(d.site.unique()); self.am, self.asd = d.age.mean(), d.age.std(); self.lm, self.lsd = d.licv.mean(), d.licv.std()
        self.ym, self.ysd = y.mean(), y.std(); X = self._X(d); Y = torch.tensor((y - self.ym) / self.ysd)
        self.members = []
        for s in range(N_ENS):
            g = torch.Generator().manual_seed(seed0 + s); torch.manual_seed(seed0 + s)
            perm = torch.randperm(len(Y), generator=g); nv = len(Y) // 10; va, tr = perm[:nv], perm[nv:]
            net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], HID), torch.nn.SiLU(), torch.nn.Linear(HID, HID), torch.nn.SiLU(),
                                      torch.nn.Linear(HID, 4)).double()
            opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
            best, bad, state = 1e18, 0, None
            for ep in range(3000):
                opt.zero_grad(); o = net(X[tr]); loss = -shash_logpdf_t(Y[tr], o[:, 0], o[:, 1], o[:, 2], o[:, 3]).mean()
                loss.backward(); opt.step()
                with torch.no_grad():
                    o = net(X[va]); v = -shash_logpdf_t(Y[va], o[:, 0], o[:, 1], o[:, 2], o[:, 3]).mean().item()
                if v < best - 1e-5: best, bad, state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad > 150: break
            net.load_state_dict(state); self.members.append(net)
    def _X(self, d):
        S = np.zeros((len(d), len(self.sites)))
        idx = {s: i for i, s in enumerate(self.sites)}
        for r, s in enumerate(d.site.values):
            if s in idx: S[r, idx[s]] = 1.0
        return torch.tensor(np.column_stack([(d.age.values - self.am) / self.asd, d.sex.values, (d.licv.values - self.lm) / self.lsd, S]))
    def params(self, d):
        with torch.no_grad():
            X = self._X(d); out = []
            for net in self.members:
                o = net(X).numpy(); out.append((self.ym + self.ysd * o[:, 0], self.ysd * np.exp(o[:, 1]), o[:, 2], np.exp(o[:, 3])))
        return out
    def logp_z(self, d, y):
        ps = self.params(d)
        dens = np.mean([np.exp(shash_logpdf_t(torch.tensor(y), torch.tensor(mu), torch.tensor(np.log(s)), torch.tensor(nu), torch.tensor(np.log(tau))).numpy())
                        for mu, s, nu, tau in ps], axis=0)
        cdf = np.mean([shash_cdf(y, mu, s, nu, tau) for mu, s, nu, tau in ps], axis=0)
        return np.log(dens), stats.norm.ppf(np.clip(cdf, 1e-9, 1 - 1e-9))
    def median(self, d):
        ps = self.params(d); return np.mean([mu + s * np.sinh(nu / tau) for mu, s, nu, tau in ps], axis=0)

allp, _ = B.load()
subj = allp.subject.unique(); fold_of = dict(zip(subj, np.random.default_rng(0).permutation(len(subj)) % 5))
rows = []
for r in REGIONS:
    d, ly = P.region_data(allp, r); f = d.subject.map(fold_of).values
    res = {k: (np.full(len(d), np.nan), np.full(len(d), np.nan)) for k in ("v2", "net")}
    for k in (range(1) if QUICK else range(5)):
        tr, te = f != k, f == k
        dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
        keep = dte.site.isin(set(dtr.site)).values; dte = dte[keep]; yte = np.log(dte[r].values.astype(float)); idx = np.where(te)[0][keep]
        m = B.fit(dtr, ly[tr], r); res["v2"][0][idx] = m.logpdf_np(dte, yte); res["v2"][1][idx] = m.z(dte, yte)
        net = Net(dtr, ly[tr], 100 * k); lp, z = net.logp_z(dte, yte); res["net"][0][idx] = lp; res["net"][1][idx] = z
        print(f"[{(time.time()-T0)/60:5.1f} min] {r} fold {k}", flush=True)
    # aging rate at 70 by sex, reference head, average scanner (scanner-weighted for the network)
    mfull = B.fit(d, ly, r); netfull = Net(d, ly, 999); w = d.site.value_counts(normalize=True)
    rate = {}
    for sx in (0, 1):
        g = pd.DataFrame({"age": [69.5, 70.5], "sex": float(sx), "licv": 0.0, "site": mfull.sites[0], "icv_method": "SynthStrip"})
        loc, sc = mfull._loc_scale(g, zero_site=True); med = loc + sc * np.sinh(mfull.nu / mfull.tau)
        rate[("v2", sx)] = 100 * (med[1] - med[0])
        meds = sum(w[s] * netfull.median(g.assign(site=s)) for s in w.index)
        rate[("net", sx)] = 100 * (meds[1] - meds[0])
    band = pd.cut(d.age, [0, 18, 40, 60, 75, 100])
    for k in ("v2", "net"):
        lp, z = res[k]; ok = np.isfinite(z)
        gaps = [abs(np.nanmean(z[(band == b).values & (d.sex.values == 1)]) - np.nanmean(z[(band == b).values & (d.sex.values == 0)]))
                for b in band.cat.categories]
        rows.append({"region": r, "model": k, "heldout logp": np.nanmean(lp), "SD z": np.nanstd(z), "in1.96": np.nanmean(np.abs(z[ok]) < 1.96),
                     "max|sex gap|": np.nanmax(gaps), "rate@70 women": rate[(k, 0)], "rate@70 men": rate[(k, 1)]})
    print(f"[{(time.time()-T0)/60:5.1f} min] done {r}", flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "neural_shash_results.csv"), index=False)
pd.set_option("display.width", 250); print("\n", R.round(4).to_string(index=False))
w = R.pivot(index="region", columns="model", values="heldout logp")
print(f"\nheld-out log-likelihood: v2 better in {(w.v2 > w.net).sum()}/{len(w)} regions; mean difference v2 - net = {(w.v2 - w.net).mean():+.4f}")
rr = R.pivot(index="region", columns="model", values="rate@70 women"); print("aging rate @70 (women), network / v2:", (rr.net / rr.v2).round(2).to_dict())
