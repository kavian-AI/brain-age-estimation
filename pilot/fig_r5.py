#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
All figures for Report 5 (Persian labels). Numbers either recomputed here with the
same functions the experiments used, or copied from FINDINGS_LOG where the
experiment printed a table (marked "from log").
"""

import os
import sys
import warnings

import arabic_reshaper
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bidi.algorithm import get_display

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = r"C:\Users\Sibetalaee\Desktop\PhD\Brain Age Estimation"
OUT = os.path.join(BASE, "pilot")
PD = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
BLUE, RED, GREEN, GREY, ORANGE, PURPLE = "#2c6fbb", "#c0504d", "#4a8c4a", "#9a9a9a", "#e8a33d", "#7b5ea7"
plt.rcParams.update({"font.family": "Tahoma", "font.size": 14, "axes.titlesize": 15,
                     "axes.labelsize": 14, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
                     "legend.fontsize": 12.5})


def fa(t):
    return get_display(arabic_reshaper.reshape(str(t)))


def num(x, d=2):
    return f"{x:.{d}f}".translate(PD).replace(".", "٫").replace("-", "−")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ============================================================ fig 23: concept
def fig23():
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    ax.axis("off")
    boxes = [(0.70, 0.62, fa("MRI افراد سالم")), (0.36, 0.62, fa("مدل یادگیری ماشین")),
             (0.02, 0.62, fa("سن پیش‌بینی‌شده")), (0.70, 0.12, fa("MRI فرد جدید")),
             (0.36, 0.12, fa("همان مدل")), (0.02, 0.12, fa("شکاف سن مغز"))]
    for x, y, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.28, 0.22, fc="#eef3f8", ec=BLUE, lw=1.8))
        ax.text(x + 0.14, y + 0.11, t, ha="center", va="center", fontsize=13)
    for y in (0.73, 0.23):
        for x in (0.30, 0.64):
            ax.annotate("", xy=(x + 0.06, y), xytext=(x, y), arrowprops=dict(arrowstyle="<-", lw=1.8))
    ax.text(0.5, 0.93, fa("مرحله‌ی آموزش: فقط افراد سالم، هدف = سن تقویمی"), ha="center", fontsize=13.5, color=BLUE)
    ax.text(0.5, 0.43, fa("مرحله‌ی آزمون: هر فردی"), ha="center", fontsize=13.5, color=BLUE)
    ax.text(0.16, 0.02, fa("شکاف = سن پیش‌بینی‌شده منهای سن واقعی"), ha="center", fontsize=11.5)
    ax.set_title(fa("الف) سن مغز چگونه ساخته می‌شود"))
    ax = axs[1]
    ax.set_xlim(-0.2, 1.25)
    ax.set_ylim(-0.25, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    th = np.deg2rad(58)
    ax.annotate("", xy=(1.1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=3, color=BLUE))
    ax.annotate("", xy=(np.cos(th), np.sin(th)), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=3, color=RED))
    ax.text(1.12, -0.1, fa("جهت سن مغز"), color=BLUE, ha="right", fontsize=13.5)
    ax.text(np.cos(th) + 0.02, np.sin(th) + 0.05, fa("جهت بیماری"), color=RED, fontsize=13.5)
    a = np.linspace(0, th, 40)
    ax.plot(0.28 * np.cos(a), 0.28 * np.sin(a), color=GREY)
    ax.text(0.31, 0.14, "θ", fontsize=16)
    ax.plot([np.cos(th), np.cos(th)], [np.sin(th), 0], ls="--", color=GREY)
    ax.text(0.5, -0.2, fa("η = cos²θ : سهمی که سن مغز از اطلاعات بیماری برمی‌دارد"), fontsize=12.5, ha="center")
    ax.set_title(fa("ب) قانون هم‌راستایی در یک تصویر"))
    save(fig, "fig23_concept_fa.png")


# ============================================================ fig 24: where brain age looks
def fig24():
    import exp62_rich_atlas as R
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.preprocessing import StandardScaler
    a, m, T, names = R.build("X")
    cn = m.DX.values == "CN"
    Z, _ = R.normative(m, names, cn)
    ag = m.age_visit.values.astype(float)
    Xv = np.column_stack([m[n].values for n in names] + [m.ICV_T.values, m.SEX.values])
    sc = StandardScaler().fit(Xv[cn])
    am = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(sc.transform(Xv[cn]), ag[cn])
    wa = np.abs(am.coef_[:len(names)])
    sel = m.DX.isin(["CN", "Dementia"]).values
    dm = LogisticRegression(C=0.05, max_iter=5000).fit(
        StandardScaler().fit_transform(np.column_stack([Z[sel], ag[sel]])), (m.DX.values[sel] == "Dementia").astype(int))
    wd = np.abs(dm.coef_[0][:len(names)])
    cls = T.cls.values
    labels = {"mtl": "لوب گیجگاهی داخلی", "cortex": "سایر قشر", "subcortical": "زیرقشری",
              "ventricle": "بطن‌ها", "callosum": "جسم پینه‌ای", "brainstem": "ساقه‌ی مغز", "cerebellum": "مخچه"}
    order = ["mtl", "cortex", "subcortical", "ventricle", "callosum", "brainstem", "cerebellum"]
    sa = [100 * wa[cls == c].sum() / wa.sum() for c in order]
    sd = [100 * wd[cls == c].sum() / wd.sum() for c in order]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(order))
    axs[0].bar(x - 0.2, sa, 0.4, color=BLUE, label=fa("مدل سن مغز"))
    axs[0].bar(x + 0.2, sd, 0.4, color=RED, label=fa("مدل تشخیص آلزایمر"))
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([fa(labels[c]) for c in order], rotation=30, ha="right")
    axs[0].set_ylabel(fa("سهم از کل وزن (درصد)"))
    axs[0].legend()
    axs[0].set_title(fa("الف) هر مدل وزنش را کجا خرج می‌کند؟ (ADNI، ۹۸ ناحیه)"))
    ra = pd.Series(wa).rank()
    rd = pd.Series(wd).rank()
    colors = [RED if c == "mtl" else BLUE if c in ("callosum", "brainstem", "ventricle") else GREY for c in cls]
    axs[1].scatter(ra, rd, c=colors, s=45, alpha=0.8)
    axs[1].set_xlabel(fa("رتبه‌ی اهمیت در مدل سن مغز"))
    axs[1].set_ylabel(fa("رتبه‌ی اهمیت در مدل آلزایمر"))
    axs[1].text(0.98, 0.03, fa("همبستگی رتبه‌ای") + chr(10) + "ADNI: " + num(ra.corr(rd), 3)
                + "    OASIS: " + num(-0.029, 3),
                transform=axs[1].transAxes, ha="right", va="bottom", fontsize=13,
                bbox=dict(fc="white", ec=GREY))
    axs[1].set_title(fa("ب) دو رتبه‌بندی بر هم عمودند"))
    save(fig, "fig24_orthogonal_fa.png")


# ============================================================ fig 25: selection (from log, E62b)
def fig25():
    arms = [("سن مغز، ۶ ناحیه", 0.904, BLUE), ("میانگین، ۶ ناحیه", 0.931, ORANGE),
            ("سن مغز، ۹۸ ناحیه", 0.881, BLUE), ("میانگین، ۹۸ ناحیه", 0.857, ORANGE),
            ("میانگین، ۱۶ ناحیه‌ی MTL", 0.940, GREEN), ("میانگین، ۸۲ ناحیه‌ی دیگر", 0.811, GREY),
            ("فقط هیپوکامپ", 0.922, PURPLE)]
    fig, ax = plt.subplots(figsize=(13, 5.8))
    y = np.arange(len(arms))[::-1]
    ax.barh(y, [a[1] for a in arms], color=[a[2] for a in arms])
    for yy, a in zip(y, arms):
        ax.text(a[1] + 0.003, yy, num(a[1], 3), va="center", fontsize=13)
    ax.set_yticks(y)
    ax.set_yticklabels([fa(a[0]) for a in arms])
    ax.set_xlim(0.78, 0.97)
    ax.set_xlabel(fa("AUC تشخیص آلزایمر از سالم، روی همان ۱۳۶۹ نفر"))
    ax.set_title(fa("راز در انتخاب نواحی است، نه در وزن‌دهی"))
    save(fig, "fig25_selection_fa.png")


# ============================================================ fig 26: the law
def fig26():
    import exp62_rich_atlas as R
    import exp66_oasis_external as O
    from scipy.stats import norm
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import roc_auc_score
    D = pd.read_csv(os.path.join(OUT, "exp67_subsets.csv"))
    a, m, T, names = R.build("X")
    Ro, _, _ = O.load_fs()
    common = [n for n in names if n in Ro.columns]
    ia = [names.index(n) for n in common]
    sign = T.sign.values[ia]
    cn, ad = m.DX.values == "CN", m.DX.values == "Dementia"
    Za, _ = R.normative(m, names, cn)
    Za = Za[:, ia]
    S = LedoitWolf().fit(np.vstack([Za[cn] - Za[cn].mean(0), Za[ad] - Za[ad].mean(0)])).covariance_
    mu = Za[ad].mean(0) - Za[cn].mean(0)
    clin = pd.read_csv(os.path.join(BASE, "data", "oasis_longitudinal.csv")).rename(
        columns={"MRI ID": "session", "Subject ID": "subject"})
    M = Ro[Ro.mpr == "mpr-1"].merge(clin, on="session")
    M["SEX"] = (M["M/F"] == "M").astype(int)
    M = M.sort_values(["subject", "Visit"]).reset_index(drop=True)
    B = M.drop_duplicates("subject").reset_index(drop=True)
    ever = M.groupby("subject").CDR.max()
    ref = ((B.CDR == 0) & (B.subject.map(ever) == 0)).values
    y = (B.CDR > 0).astype(int).values
    Zo = O.normative(B[common].values.astype(float), B.Age.values.astype(float), B.SEX.values,
                     B.eTIV.values.astype(float), ref, B.subject.values.astype(str))
    rng = np.random.RandomState(1)
    pr, ob = [], []
    for _ in range(600):
        k = rng.choice([3, 5, 8, 12, 20, 35, 60, len(common)])
        cols = rng.choice(len(common), k, replace=False)
        w = np.zeros(len(common))
        w[cols] = sign[cols]
        pr.append(abs(w @ mu) / np.sqrt(w @ S @ w))
        ob.append(roc_auc_score(y, Zo @ w))
    pr, ob = np.array(pr), np.array(ob)
    dob = np.sqrt(2) * norm.ppf(np.clip(ob, 0.501, 0.999))
    c = (pr @ dob) / (pr @ pr)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.3))
    for s, col in [("uniform", ORANGE), ("gap", BLUE)]:
        q = D[D.summary == s]
        axs[0].scatter(q.pred, q.obs, s=14, alpha=0.6, color=col,
                       label=fa("میانگین روی زیرمجموعه‌ی تصادفی" if s == "uniform" else "سن مغز روی زیرمجموعه‌ی تصادفی"))
    lim = [0.6, 1.0]
    axs[0].plot(lim, lim, ls="--", color=GREY)
    axs[0].set_xlim(lim)
    axs[0].set_ylim(lim)
    axs[0].set_xlabel(fa("AUC پیش‌بینی‌شده با قانون"))
    axs[0].set_ylabel(fa("AUC مشاهده‌شده روی نیمه‌ی کنارگذاشته"))
    axs[0].legend(loc="lower right")
    axs[0].set_title(fa("الف) داخل ADNI، ۵۰۰ شاخص نادیده؛ همبستگی " + num(0.96)))
    axs[1].scatter(norm.cdf(c * pr / np.sqrt(2)), ob, s=14, alpha=0.6, color=GREEN)
    lim = [0.5, 0.85]
    axs[1].plot(lim, lim, ls="--", color=GREY)
    axs[1].set_xlim(lim)
    axs[1].set_ylim(lim)
    axs[1].set_xlabel(fa("AUC پیش‌بینی‌شده از هندسه‌ی ADNI"))
    axs[1].set_ylabel(fa("AUC مشاهده‌شده در OASIS"))
    axs[1].set_title(fa("ب) مطالعه و نرم‌افزار دیگر؛ اسپیرمن " + num(0.855, 3)))
    save(fig, "fig26_law_fa.png")


# ============================================================ fig 27: ENIGMA (from log E68, E72)
def fig27():
    pred = {"اسکیزوفرنی": (0.37, 0.51), "دوقطبی": (0.28, 0.29), "افسردگی": (0.04, 0.10),
            "اوتیسم": (0.10, 0.07), "ADHD": (-0.13, 0.06)}
    eta = [("آلزایمر", 0.29), ("اسکیزوفرنی", 0.19), ("دوقطبی", 0.19), ("پارکینسون مرحله ۳", 0.18),
           ("پارکینسون مرحله ۴-۵", 0.16), ("صرع", 0.14), ("پارکینسون مرحله ۱", 0.04), ("ADHD", 0.04),
           ("OCD", 0.03), ("اوتیسم", 0.02), ("افسردگی", 0.01), ("حذف 22q11.2", 0.004)]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.3))
    for k, (p, o) in pred.items():
        axs[0].scatter(p, o, s=90, color=BLUE)
        axs[0].text(p + 0.012, o + 0.012, fa(k) if k != "ADHD" else k, fontsize=13)
    axs[0].plot([-0.2, 0.6], [-0.2, 0.6], ls="--", color=GREY)
    axs[0].set_xlim(-0.2, 0.6)
    axs[0].set_ylim(-0.05, 0.6)
    axs[0].set_xlabel(fa("اندازه‌ی اثر سن مغز، پیش‌بینی‌شده از نقشه‌های ENIGMA"))
    axs[0].set_ylabel(fa("اندازه‌ی اثر منتشرشده (Kaufmann 2019)"))
    axs[0].set_title(fa("الف) پیش‌بینی بدون داده‌ی فردی بیمار؛ اسپیرمن " + num(0.90)))
    y = np.arange(len(eta))[::-1]
    axs[1].barh(y, [100 * e for _, e in eta], color=[RED if n == "آلزایمر" else BLUE for n, _ in eta])
    axs[1].set_yticks(y)
    axs[1].set_yticklabels([fa(n) if n not in ("ADHD", "OCD") else n for n, _ in eta])
    axs[1].set_xlabel(fa("درصد اطلاعات ساختاری اختلال که سن مغز برمی‌دارد") + " (η)")
    axs[1].set_title(fa("ب) سن مغز برای کدام اختلال ساخته شده است؟"))
    save(fig, "fig27_enigma_fa.png")


# ============================================================ fig 28: decomposition (from log E69)
def fig28():
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
    g = ["سالم", "MCI", "آلزایمر"]
    dis = [0.13, 1.83, 4.10]
    oth = [-0.05, 0.37, 0.64]
    x = np.arange(3)
    axs[0].bar(x, dis, color=RED, label=fa("سال‌های شبیه آلزایمر"))
    axs[0].bar(x, oth, bottom=[max(d, 0) for d in dis], color=GREY, label=fa("سال‌های دیگر"))
    for xi, d, o in zip(x, dis, oth):
        axs[0].text(xi, d + max(o, 0) + 0.12, fa("کل " + num(d + o)), ha="center", fontsize=13)
    axs[0].set_ylim(0, 5.6)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([fa(t) if t != "MCI" else t for t in g])
    axs[0].set_ylabel(fa("شکاف سن مغز (سال)"))
    axs[0].legend(loc="upper left")
    axs[0].set_title(fa("الف) از سال‌های اضافه، چند سال آلزایمر است؟"))
    bio = ["آمیلوئید (PET)", "ژن APOE4", "تاو (CSF)"]
    d_ = [0.302, 0.251, 0.174]
    o_ = [0.011, 0.031, -0.113]
    t_ = [0.212, 0.179, 0.026]
    x = np.arange(3)
    axs[1].bar(x - 0.27, t_, 0.27, color=BLUE, label=fa("کل شکاف"))
    axs[1].bar(x, d_, 0.27, color=RED, label=fa("سال‌های آلزایمری"))
    axs[1].bar(x + 0.27, o_, 0.27, color=GREY, label=fa("سال‌های دیگر"))
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([fa(b) for b in bio])
    axs[1].set_ylabel(fa("همبستگی جزئی در افراد MCI"))
    axs[1].legend()
    axs[1].set_title(fa("ب) زیست‌شناسی آلزایمر فقط در یک بخش زندگی می‌کند"))
    save(fig, "fig28_decomposition_fa.png")


# ============================================================ fig 29: SNAP Kaplan-Meier
def fig29():
    import exp53_units_and_dimension as E
    import exp62_rich_atlas as R
    import exp69_decomposed_brain_age as D
    from lifelines import KaplanMeierFitter
    a, m, T, names = R.build("X")
    cn = m.DX.values == "CN"
    Z, _ = R.normative(m, names, cn)
    gap = R.rich_gap(m, names, cn)
    gd, go, _ = D.decompose(m, Z, gap)
    m = m.assign(gap=gap, gd=gd, go=go)
    M = m[m.DX.values == "MCI"].copy()
    keep, dur, ev = E.survival(a, M.PTID.values, ["Dementia"])
    o = {x: i for i, x in enumerate(keep)}
    M["T"] = [dur[o[x]] if x in o else np.nan for x in M.PTID]
    M["E"] = [ev[o[x]] if x in o else np.nan for x in M.PTID]
    M = M.dropna(subset=["T", "E"])
    H = M[M.gap >= np.percentile(M.gap, 66.7)]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for lab, q, col in [("عمدتاً شبیه آلزایمر", H.go <= H.gd, RED), ("عمدتاً غیرآلزایمری (شبیه SNAP)", H.go > H.gd, BLUE)]:
        k = KaplanMeierFitter().fit(H["T"][q], H["E"][q])
        sf = k.survival_function_
        ax.step(sf.index, 1 - sf.iloc[:, 0], where="post", color=col, lw=2.5,
                label=fa(lab) + f"  (n={int(q.sum())})".translate(PD))
    ax.set_xlim(0, 8)
    ax.set_xlabel(fa("سال از اسکن پایه"))
    ax.set_ylabel(fa("نسبت تبدیل‌شده به دمانس"))
    ax.legend(loc="upper left")
    ax.set_title(fa("دو گروه با شکاف سن مغز یکسان؛ نسبت خطر " + num(0.42)))
    save(fig, "fig29_snap_fa.png")


# ============================================================ fig 30: frontier (from log E70b)
def fig30():
    std = [(31.6, 3.68, 0.846), (100, 3.64, 0.867), (316, 3.66, 0.878), (1000, 3.77, 0.875), (3162, 3.99, 0.853)]
    aw = [(3.65, 0.871), (3.67, 0.882), (3.62, 0.895), (3.66, 0.899), (3.65, 0.906), (3.68, 0.908),
          (3.73, 0.911), (3.73, 0.914), (3.88, 0.913), (3.80, 0.918), (3.78, 0.915)]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    s = np.array([(b, c) for _, b, c in std])
    ax.plot(s[:, 0], s[:, 1], "-o", color=BLUE, lw=2, ms=8, label=fa("سن مغز استاندارد (شدت تنظیم‌های مختلف)"))
    w = np.array(aw)
    ax.scatter(w[:, 0], w[:, 1], s=80, color=RED, marker="D", label=fa("سن مغز آگاه به بیماری"))
    ax.set_xlabel(fa("خطای تخمین سن در افراد سالم (سال) — کمتر بهتر"))
    ax.set_ylabel(fa("AUC تشخیص آلزایمر — بیشتر بهتر"))
    ax.invert_xaxis()
    ax.legend(loc="lower left")
    ax.set_title(fa("مرز استاندارد، مرز واقعی نیست"))
    save(fig, "fig30_frontier_fa.png")


# ============================================================ fig 31: dual reference
def fig31():
    import exp53_units_and_dimension as E
    import exp63_dual_reference as DR
    a, b = E.load()
    gH, gD = DR.dual_gaps(b)
    tH = np.quantile(gH[b.DX.values == "CN"], 0.9)
    tD = np.quantile(gD[b.DX.values == "Dementia"], 0.1)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.3), gridspec_kw={"width_ratios": [1.3, 1]})
    for g, col, lab in [("CN", GREEN, "سالم"), ("MCI", ORANGE, "MCI"), ("Dementia", RED, "آلزایمر")]:
        q = b.DX.values == g
        axs[0].scatter(gH[q], gD[q], s=8, alpha=0.45, color=col, label=fa(lab) if lab != "MCI" else lab)
    axs[0].set_xlim(-12, 25)
    axs[0].axvline(tH, color="k", ls="--")
    axs[0].axhline(tD, color="k", ls="--")
    axs[0].set_xlabel(fa("شکاف با مدل آموزش‌دیده روی سالم‌ها (سال)"))
    axs[0].set_ylabel(fa("شکاف با مدل آموزش‌دیده روی بیماران (سال)"))
    axs[0].legend(markerscale=3)
    axs[0].set_title(fa("الف) مدل دوگانه: دو مرجع، چهار ناحیه"))
    cats = ["سالم (توافق)", "پرچم: شبیه هر دو", "پرچم: شبیه هیچ‌کدام", "بیمار (توافق)"]
    conv = [22.5, 30.3, 42.9, 61.4]
    axs[1].bar(range(4), conv, color=[GREEN, ORANGE, PURPLE, RED])
    for i, c in enumerate(conv):
        axs[1].text(i, c + 1, num(c, 1) + "٪", ha="center", fontsize=13)
    axs[1].set_xticks(range(4))
    axs[1].set_xticklabels([fa(c) for c in cats], rotation=20, ha="right")
    axs[1].set_ylabel(fa("درصد تبدیل به دمانس در افراد MCI"))
    axs[1].set_title(fa("ب) نردبان خطر در افراد MCI"))
    save(fig, "fig31_dual_fa.png")


if __name__ == "__main__":
    for f in [fig23, fig25, fig27, fig28, fig30, fig24, fig26, fig29, fig31]:
        f()
