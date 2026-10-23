#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing the user's three ideas for FUSING age and disease information,
instead of computing them separately.

All models share: same subjects, same CV folds, same input features
(regional volumes / ICV + sex). Chronological age is available as a
supervision signal but is NOT given as an input feature to the disease
head, so the age pathway has to *earn* its contribution.

Contenders
----------
FAIR   volumes + age -> logistic regression      (the baseline that won before)
DIRECT MLP trained only on the disease label       (no age at all)
MT     MULTI-TASK MLP: shared trunk, 2 heads (age regression + disease)   [idea 1]
TL     TRANSFER: pretrain trunk on age, then fine-tune on disease         [idea 2]
ENS    ENSEMBLE: average of brain-age-gap risk + DIRECT MLP risk          [idea 3]

Metric: AUC, pooled out-of-fold, repeated over seeds.
"""

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.manual_seed(0)
DATA = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\data"
OUT = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation\pilot"
DEV = "cpu"


# ----------------------------------------------------------------- models
class Trunk(nn.Module):
    def __init__(self, d_in, d_h=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(d_h, d_h), nn.ReLU(), nn.Dropout(0.3))

    def forward(self, x):
        return self.net(x)


class MultiTask(nn.Module):
    """Shared trunk, one age head (regression), one disease head (classification)."""
    def __init__(self, d_in, d_h=32):
        super().__init__()
        self.trunk = Trunk(d_in, d_h)
        self.age = nn.Linear(d_h, 1)
        self.dx = nn.Linear(d_h, 1)

    def forward(self, x):
        z = self.trunk(x)
        return self.age(z).squeeze(1), self.dx(z).squeeze(1)


def _t(a):
    return torch.tensor(a, dtype=torch.float32, device=DEV)


def train_direct(Xtr, ytr, epochs=250, lr=1e-3):
    m = nn.Sequential(Trunk(Xtr.shape[1]), nn.Linear(32, 1)).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    X, y = _t(Xtr), _t(ytr)
    for _ in range(epochs):
        opt.zero_grad()
        out = m(X).squeeze(1)
        lossf(out, y).backward()
        opt.step()
    return m


def train_multitask(Xtr, ytr, atr, epochs=300, lr=1e-3, w_age=0.3):
    m = MultiTask(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-3)
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    X, y, a = _t(Xtr), _t(ytr), _t(atr)
    for _ in range(epochs):
        opt.zero_grad()
        pa, pd_ = m(X)
        (bce(pd_, y) + w_age * mse(pa, a)).backward()
        opt.step()
    return m


def train_transfer(Xtr, atr, ytr, ep_pre=300, ep_ft=200, lr=1e-3):
    # 1) pretrain trunk+agehead to predict age on ALL training subjects
    m = MultiTask(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(list(m.trunk.parameters()) + list(m.age.parameters()),
                           lr=lr, weight_decay=1e-3)
    mse = nn.MSELoss()
    X, a, y = _t(Xtr), _t(atr), _t(ytr)
    for _ in range(ep_pre):
        opt.zero_grad()
        pa, _ = m(X)
        mse(pa, a).backward()
        opt.step()
    # 2) fine-tune whole net on the disease label (trunk unfrozen, low lr)
    opt2 = torch.optim.Adam(list(m.trunk.parameters()) + list(m.dx.parameters()),
                            lr=lr * 0.3, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(ep_ft):
        opt2.zero_grad()
        _, pd_ = m(X)
        bce(pd_, y).backward()
        opt2.step()
    return m


@torch.no_grad()
def prob_direct(m, X):
    return torch.sigmoid(m(_t(X)).squeeze(1)).cpu().numpy()


@torch.no_grad()
def prob_dx(m, X):
    return torch.sigmoid(m(_t(X))[1]).cpu().numpy()


def gap_score(vol_tr, age_tr, extra_tr, y_tr, vol_te, age_te, extra_te):
    """Classic brain-age gap risk: age model on healthy -> gap -> logistic."""
    h = y_tr == 0
    Xh = np.column_stack([vol_tr[h], extra_tr[h]])
    sc = StandardScaler().fit(Xh)
    m = LinearRegression().fit(sc.transform(Xh), age_tr[h])
    b = LinearRegression().fit(age_tr[h].reshape(-1, 1),
                               m.predict(sc.transform(Xh)) - age_tr[h])

    def gap(vol, age, extra):
        X = sc.transform(np.column_stack([vol, extra]))
        return (m.predict(X) - age) - b.predict(age.reshape(-1, 1))

    g_tr = gap(vol_tr, age_tr, extra_tr).reshape(-1, 1)
    g_te = gap(vol_te, age_te, extra_te).reshape(-1, 1)
    clf = LogisticRegression(max_iter=2000).fit(g_tr, y_tr)
    return clf.predict_proba(g_te)[:, 1]


def run(vol, age, extra, y, name, n_rep=8):
    keys = ["FAIR", "DIRECT", "MT", "TL", "ENS"]
    res = {k: [] for k in keys}
    for rep in range(n_rep):
        oof = {k: np.zeros(len(y)) for k in keys}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(vol, y):
            voln = vol / vol[:, -1:].clip(min=1) if False else vol
            feat = np.column_stack([vol, extra])          # volumes + ICV + sex
            sc = StandardScaler().fit(feat[tr])
            Ftr, Fte = sc.transform(feat[tr]), sc.transform(feat[te])
            atr = age[tr]

            # FAIR: volumes + age -> logistic
            fair = LogisticRegression(max_iter=2000)
            Xa_tr = np.column_stack([Ftr, (atr - atr.mean()) / (atr.std() + 1e-9)])
            Xa_te = np.column_stack([Fte, (age[te] - atr.mean()) / (atr.std() + 1e-9)])
            oof["FAIR"][te] = fair.fit(Xa_tr, y[tr]).predict_proba(Xa_te)[:, 1]

            # DIRECT MLP (no age)
            md = train_direct(Ftr, y[tr])
            p_direct = prob_direct(md, Fte)
            oof["DIRECT"][te] = p_direct

            # idea 1: multi-task
            mt = train_multitask(Ftr, y[tr], (atr - atr.mean()) / (atr.std() + 1e-9))
            oof["MT"][te] = prob_dx(mt, Fte)

            # idea 2: transfer
            tl = train_transfer(Ftr, (atr - atr.mean()) / (atr.std() + 1e-9), y[tr])
            oof["TL"][te] = prob_dx(tl, Fte)

            # idea 3: ensemble of brain-age-gap risk + direct MLP
            g = gap_score(vol[tr], age[tr], extra[tr], y[tr],
                          vol[te], age[te], extra[te])
            def rank(z):
                r = z.argsort().argsort().astype(float)
                return r / (len(r) - 1 + 1e-9)
            oof["ENS"][te] = 0.5 * rank(g) + 0.5 * rank(p_direct)
        for k in keys:
            res[k].append(roc_auc_score(y, oof[k]))

    print(f"\n{'='*66}\n{name}  (n={len(y)}, patients={int(y.sum())})\n{'='*66}")
    lbl = {"FAIR": "volumes + age  (logistic)      [prior winner]",
           "DIRECT": "direct MLP  (disease only)",
           "MT": "multi-task MLP  (shared trunk)  [idea 1]",
           "TL": "pretrain-age -> fine-tune       [idea 2]",
           "ENS": "ensemble: gap + direct          [idea 3]"}
    for k in ["FAIR", "DIRECT", "MT", "TL", "ENS"]:
        a = np.array(res[k])
        print(f"  {lbl[k]:46s} AUC = {a.mean():.3f}  "
              f"[{np.percentile(a,2.5):.3f}, {np.percentile(a,97.5):.3f}]")
    best = max(("MT", "TL", "ENS"), key=lambda k: np.mean(res[k]))
    d = np.array(res[best]) - np.array(res["FAIR"])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    print(f"\n  best fusion ({best}) - FAIR baseline = {d.mean():+.3f}  p={p:.3f}")
    print("  ->", "FUSION HELPS" if d.mean() > 0 and p < .05
          else "fusion does not beat the simple baseline")
    return res


def main():
    # ADNI: AD vs CN
    d = pd.read_csv(f"{DATA}/ADNIMERGE_05Mar2026.csv", low_memory=False)
    V = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
    a = d.dropna(subset=V + ["ICV", "AGE", "DX", "PTGENDER"]).sort_values("Years_bl")
    a = a.groupby("RID").first().reset_index()
    a = a[a.DX.isin(["CN", "Dementia"])]
    extra = np.column_stack([a.ICV.values, (a.PTGENDER == "Male").astype(int).values])
    r1 = run(a[V].values, a.AGE.values, extra,
             (a.DX == "Dementia").astype(int).values, "ADNI: AD vs CN (diagnosis)")

    # ADNI prognosis: MCI -> dementia
    d2 = d.dropna(subset=["DX"]).sort_values(["RID", "Years_bl"])
    lab = {}
    for rid, g in d2.groupby("RID"):
        dx = g.DX.tolist()
        if dx[0] == "MCI":
            lab[rid] = int("Dementia" in dx)
    b = d.dropna(subset=V + ["ICV", "AGE", "PTGENDER"]).groupby("RID").first()
    b = b[b.index.isin(lab)]
    y = np.array([lab[r] for r in b.index])
    extra_b = np.column_stack([b.ICV.values, (b.PTGENDER == "Male").astype(int).values])
    r2 = run(b[V].values, b.AGE.values, extra_b, y, "ADNI: MCI->dementia (prognosis)")

    np.savez(f"{OUT}/exp4_fusion.npz",
             **{f"diag_{k}": np.array(v) for k, v in r1.items()},
             **{f"prog_{k}": np.array(v) for k, v in r2.items()})
    print(f"\nsaved -> {OUT}/exp4_fusion.npz")


if __name__ == "__main__":
    main()
