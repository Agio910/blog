# -*- coding: utf-8 -*-
"""第 7 章 §8.3：N50/NG50/L50；Lander-Waterman 公式与模拟对照；
鸡血红蛋白 β 链基因（GenBank J00860）上的表 7.3、7.4、7.6 与式（7.9）-（7.11）。"""
import math
from collections import Counter, defaultdict
import numpy as np

# ---------- 1. 连续性指标 ----------
def nx_stats(lengths, total=None, x=0.5):
    """total=None 时以组装总长为分母（N50/L50），给定基因组大小时为 NG50/LG50。"""
    s = sorted(lengths, reverse=True)
    target = x * (sum(s) if total is None else total)
    cum = 0
    for i, L in enumerate(s, 1):
        cum += L
        if cum >= target:
            return L, i
    return 0, None                     # 组装总长不到基因组的 x，NG50 无定义

ctg = [100, 80, 60, 40, 20]
print("contig:", ctg, " N50, L50 =", nx_stats(ctg), " NG50, LG50 (G=400) =", nx_stats(ctg, 400))
print("把 100 拆成 50+50 后 N50 =", nx_stats([50, 50, 80, 60, 40, 20])[0],
      "；丢掉所有 <50 的 contig 后 N50 =", nx_stats([100, 80, 60])[0])

# ---------- 2. Lander-Waterman：理论值与模拟 ----------
rng = np.random.default_rng(1)
G, L, T = 1_000_000, 100, 20           # 基因组、读长、可检出的最小重叠
sigma = 1 - T / L
print(f"\nLander-Waterman（G={G:,}, L={L}, 最小重叠 T={T}）")
print(" c  | 未覆盖比例 e^-c | 期望contig数 Ne^(-c*sigma) 模拟 | 期望contig长 理论  模拟")
for c in (1, 2, 4, 6, 8, 10):
    N = int(c * G / L)
    exp_n = N * math.exp(-c * sigma)
    exp_len = L * ((math.exp(c * sigma) - 1) / c + 1 - sigma)
    sims = []
    for _ in range(5):
        s = np.sort(rng.integers(0, G - L + 1, N))
        brk = np.flatnonzero(np.diff(s) > L - T)        # 相邻读序重叠不足 T 处断开
        first = np.r_[0, brk + 1]
        last = np.r_[brk, N - 1]
        sims.append((len(first), np.mean(s[last] - s[first] + L)))
    n_sim, len_sim = np.mean(sims, axis=0)
    print(f"{c:>3} | {math.exp(-c):>14.4f} | {exp_n:>12,.0f} {n_sim:>12,.0f} | "
          f"{exp_len:>10,.0f} {len_sim:>8,.0f}")

# ---------- 3. 鸡血红蛋白 β 链基因，438 nt（CDS 去掉起始 ATG 与终止 TAA）----------
SEQ = ("GTGCACTGGACTGCTGAGGAGAAGCAGCTCATCACCGGCCTCTGGGGCAAGGTCAATGTGGCCGAATGTGGGG"
       "CCGAAGCCCTGGCCAGGCTGCTGATCGTCTACCCCTGGACCCAGAGGTTCTTTGCGTCCTTTGGGAACCTCTC"
       "CAGCCCCACTGCCATCCTTGGCAACCCCATGGTCCGCGCCCACGGCAAGAAAGTGCTCACCTCCTTTGGGGAT"
       "GCTGTGAAGAACCTGGACAACATCAAGAACACCTTCTCCCAACTGTCCGAACTGCATTGTGACAAGCTGCATG"
       "TGGACCCCGAGAACTTCAGGCTCCTGGGTGACATCCTCATCATTGTCCTGGCCGCCCACTTCAGCAAGGACTT"
       "CACTCCTGAATGCCAGGCTGCCTGGCAGAAGCTGGTCCGCGTGGTGGCCCATGCCCTGGCTCGCAAGTACCAC")
B = "ACGT"
n = len(SEQ)
base = Counter(SEQ)
print(f"\nJ00860 片段长 {n}，碱基数", {b: base[b] for b in B})

pairs = Counter(SEQ[i:i + 2] for i in range(n - 1))
tab = np.array([[pairs[a + b] for b in B] for a in B])
print("表 7.3 重算（行=第一碱基，列=第二碱基）：")
for a, row in zip(B, tab):
    print("  ", a, row, " 行和", row.sum())
print("   列和", tab.sum(0), " 总计", tab.sum())
expct = np.outer(tab.sum(1), tab.sum(0)) / tab.sum()
print(f"   行列独立性卡方 = {((tab - expct) ** 2 / expct).sum():.2f}（df=9，0.05 临界值 16.92）")
tab_book = tab.copy(); tab_book[3, 2] = 41               # 教材印的 T 行 G 列
e2 = np.outer(tab_book.sum(1), tab_book.sum(0)) / tab_book.sum()
print(f"   若按教材印的 TG=41：卡方 = {((tab_book - e2) ** 2 / e2).sum():.2f}")

codons = Counter(SEQ[i:i + 3] for i in range(0, n, 3))
print("表 7.4 核对：密码子总数", sum(codons.values()), "，CAG =", codons["CAG"],
      "，GCC =", codons["GCC"], "，CTG =", codons["CTG"])

# ---------- 4. Karlin 字码值、最长同向重复及其期望 ----------
val = {b: i for i, b in enumerate(B)}
word_value = lambda w: 1 + sum(val[c] * 4 ** (len(w) - 1 - i) for i, c in enumerate(w))
print("\n字码值 TGACC =", word_value("TGACC"))

def repeats(seq, k):
    pos = defaultdict(list)
    for i in range(len(seq) - k + 1):
        pos[seq[i:i + k]].append(i + 1)
    return {w: p for w, p in pos.items() if len(p) > 1}

k = 8
while repeats(SEQ, k + 1):
    k += 1
print("最长同向重复长度", k, "：", repeats(SEQ, k))
P = sum((base[b] / n) ** 2 for b in B)
mu = (0.6359 + 2 * math.log(n) + math.log(1 - P)) / math.log(1 / P) - 1
var = 1.645 / math.log(P) ** 2
print(f"P = {P:.4f}，式(7.9) mu_L = {mu:.2f}，式(7.10) sigma^2 = {var:.3f}，"
      f"mu + 1.645 sigma = {mu + 1.645 * math.sqrt(var):.2f}")
