#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure for Report 6: the map of research branches (Persian labels)."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from fig_r5 import fa, save, BLUE, RED, GREEN, GREY, ORANGE, PURPLE

LAYERS = [
    ("الف) ساختن نشانگر", BLUE, [
        ("۱", "دقت، معماری و تعمیم‌پذیری", "Smith · Westlye · Davatzikos · Booth/Cole · Kann", False),
        ("۲", "سن مغز موضعی و وکسلی", "Irimia · Cole", False),
        ("۳", "مدل‌سازی نورماتیو و چارت‌ها", "Marquand · Bethlehem/Seidlitz · Frangou · Thompson", True),
    ]),
    ("ب) فهمیدن نشانگر", RED, [
        ("۴", "اعتبار و تفسیر شکاف سن مغز", "Smith · Fjell/Vidal-Piñeiro · Schulz/Ritter · Yeo", True),
        ("۵", "ناهمگونی و ابعاد پیری", "Davatzikos · Alexander/Young · Smith", False),
        ("۶", "جداسازی پیری از بیماری", "Davatzikos · Pohl/Adeli · Barkhof · Moreno", True),
        ("۷", "پیوند زیستی و چندوجهی", "Westlye/Andreassen · Wen · Wyss-Coray · Zalesky · Gispert", False),
    ]),
    ("ج) پیش‌بینی و کاربرد", GREEN, [
        ("۸", "سرعت پیری و مدل‌های طولی", "Hariri/Moffitt/Caspi · Buckner · Irimia · Durrleman", False),
        ("۹", "مدل‌های مولد و خلاف‌واقع", "Alexander/Ravì · Tsaftaris · Nachev", False),
        ("۱۰", "ترجمه‌ی بالینی و عدالت", "Ibáñez · Cole · Teipel · Booth", True),
    ]),
]


def fig32():
    fig, ax = plt.subplots(figsize=(17, 10.5))
    ax.set_xlim(0, 30)
    ax.set_ylim(-1.4, 19)
    ax.axis("off")
    top = FancyBboxPatch((7.0, 17.2), 16, 1.4, boxstyle="round,pad=0.15", fc="#1F3A5F", ec="none")
    ax.add_patch(top)
    ax.text(15, 17.9, fa("پژوهش سن مغز در بیماری‌های تخریب‌کننده‌ی عصبی"),
            ha="center", va="center", color="white", fontsize=17, weight="bold")
    # columns from right to left, as Persian is read
    xs = [20.4, 10.4, 0.4]
    w = 9.2
    for (title, col, items), x in zip(LAYERS, xs):
        ax.plot([15, x + w / 2], [17.2, 16.3], color=GREY, lw=1.5)
        hdr = FancyBboxPatch((x, 15.3), w, 1.0, boxstyle="round,pad=0.1", fc=col, ec="none")
        ax.add_patch(hdr)
        ax.text(x + w / 2, 15.8, fa(title), ha="center", va="center", color="white",
                fontsize=15, weight="bold")
        h = 3.25
        for k, (n, name, who, ours) in enumerate(items):
            y = 14.6 - (k + 1) * (h + 0.25)
            box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc="white",
                                 ec=col, lw=3.2 if ours else 1.4)
            ax.add_patch(box)
            label = fa(f"شاخه‌ی {n}: {name}")
            ax.text(x + w - 0.3, y + h - 0.8, label, ha="right", va="center", fontsize=13.5,
                    weight="bold", color=col)
            ax.text(x + w / 2, y + 1.15, who, ha="center", va="center", fontsize=10.5,
                    color="#333333", wrap=True)
            if ours:
                ax.scatter([x + 0.65], [y + 0.55], marker="*", s=520, color=ORANGE, zorder=5)
    ax.scatter([29.4], [-0.9], marker="*", s=420, color=ORANGE)
    ax.text(28.9, -0.9, fa("ستاره و کادر پررنگ: شاخه‌هایی که پژوهش ما (قانون هم‌راستایی و سن مغز تجزیه‌شده) در آن‌ها سهم دارد"),
            ha="right", va="center", fontsize=12.5, color="#333333")
    save(fig, "fig32_branches_fa.png")


if __name__ == "__main__":
    fig32()
