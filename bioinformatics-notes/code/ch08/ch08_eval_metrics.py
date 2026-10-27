# -*- coding: utf-8 -*-
"""第 8 章 §7.3：Burset & Guigó (1996) 的核苷酸水平与外显子水平评价指标，并检查表 8.1。"""
import numpy as np

def nt_metrics(true_iv, pred_iv, L):
    t = np.zeros(L, bool); p = np.zeros(L, bool)
    for a, b in true_iv: t[a:b] = True
    for a, b in pred_iv: p[a:b] = True
    TP, FP = int((t & p).sum()), int((~t & p).sum())
    FN, TN = int((t & ~p).sum()), int((~t & ~p).sum())
    cc = (TP * TN - FP * FN) / np.sqrt(float(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    acp = (TP / (TP + FN) + TP / (TP + FP) + TN / (TN + FP) + TN / (TN + FN)) / 4
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, Sn=TP / (TP + FN), Sp=TP / (TP + FP),
                Sp_classic=TN / (TN + FP), CC=cc, AC=2 * (acp - 0.5))

def exon_metrics(true_ex, pred_ex):
    ov = lambda e, S: any(a < e[1] and e[0] < b for a, b in S)
    T, P = set(true_ex), set(pred_ex)
    return dict(Sn=len(T & P) / len(T), Sp=len(T & P) / len(P),       # 两端边界都对才算对
                ME=sum(not ov(e, P) for e in T) / len(T),               # 与任何预测都不重叠的真实外显子
                WE=sum(not ov(e, T) for e in P) / len(P))               # 与任何真实外显子都不重叠的预测外显子

true_ex = [(1000, 1200), (1500, 1650), (2000, 2300)]
pred_ex = [(1000, 1200), (1490, 1650), (2600, 2700)]
for L in (10_000, 1_000_000):
    m = nt_metrics(true_ex, pred_ex, L)
    print("L=%-9d" % L, " ".join("%s=%s" % (k, ("%.3f" % v) if isinstance(v, float) else v) for k, v in m.items()))
print("外显子水平:", {k: round(v, 3) for k, v in exon_metrics(true_ex, pred_ex).items()})

# 表 8.1（教材 p.194）外显子水平：1-Sn-ME = 真实外显子中“部分重叠但边界不对”的比例
tab = {"FGENESH": (.81, .80, .09, .11), "AUGUSTUS": (.72, .84, .20, .08), "GENSCAN": (.78, .74, .08, .14),
       "GeneParser": (.69, .63, .31, .37), "Grail-I": (.59, .91, .40, .09)}
for k, (sn, sp, me, we) in tab.items():
    print("%-10s 1-Sn-ME = %.2f   1-Sp-WE = %.2f" % (k, round(1 - sn - me, 2) + 0.0, round(1 - sp - we, 2) + 0.0))
