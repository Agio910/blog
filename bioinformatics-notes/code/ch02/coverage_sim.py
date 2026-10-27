# -*- coding: utf-8 -*-
"""第 2 章 §6.3/§7.2：鸟枪法测序的覆盖度——泊松近似、Lander–Waterman、过离散与短 contig 的覆盖度噪声。

numpy 即可；随机种子固定，输出可复现。
"""
import math
import numpy as np

rng = np.random.default_rng(2026)
G, L = 1_000_000, 150          # 基因组长度、读长


def simulate_cov(c, rate=None):
    """读长起点在 [0, G-L] 上均匀（或按 rate 加权）抽取，返回每个位置的覆盖深度。"""
    n = int(round(c * G / L))
    if rate is None:
        starts = rng.integers(0, G - L + 1, size=n)
    else:
        w = rate[: G - L + 1] / rate[: G - L + 1].sum()
        starts = rng.choice(G - L + 1, size=n, p=w)
    diff = np.zeros(G + 1, dtype=np.int64)
    np.add.at(diff, starts, 1)
    np.add.at(diff, starts + L, -1)
    return np.cumsum(diff)[:G], starts


def islands(starts, min_overlap=0):
    """把相互重叠（重叠 >= min_overlap）的读长连成"岛"（Lander–Waterman 意义下的 contig）。"""
    s = np.sort(starts)
    gaps = np.diff(s)                      # 相邻起点间距
    breaks = gaps > (L - min_overlap)      # 间距超过 L - T 就接不上
    return int(breaks.sum()) + 1


print("== 1. 均匀抽样下，覆盖深度近似 Poisson(c) ==")
print("  c    读长数   零覆盖比例(全长/去两端)   e^-c       均值   方差/均值   P(深度<5) 实测 / 泊松")
for c in (1, 2, 5, 10, 30):
    cov, starts = simulate_cov(c)
    lam = c
    p_lt5 = sum(math.exp(-lam) * lam ** k / math.factorial(k) for k in range(5))
    inner = cov[L:G - L]                    # 去掉两端各 L bp（那里能覆盖到的读长起点更少）
    print("%3d  %7d   %9.5f / %-9.5f   %9.2e   %6.2f   %7.3f    %.4f / %.4f" % (
        c, len(starts), (cov == 0).mean(), (inner == 0).mean(), math.exp(-c), cov.mean(),
        cov.var() / cov.mean(), (cov < 5).mean(), p_lt5))

print("\n== 2. Lander–Waterman：期望岛数 N·exp(-c(1-θ))，θ = T/L ==")
print("  c   T(最小重叠)    实测岛数    公式      N")
for c in (1, 2, 5, 8):
    cov, starts = simulate_cov(c)
    N = len(starts)
    for T in (0, 30):
        theta = T / L
        print("%3d   %5d       %7d   %9.1f   %d" % (c, T, islands(starts, T), N * math.exp(-c * (1 - theta)), N))

print("\n== 3. 过离散：沿基因组的覆盖率按 1 kb 窗口乘以对数正态因子（模拟 GC/扩增偏好）==")
win = 1000
factor = np.repeat(rng.lognormal(mean=0.0, sigma=0.5, size=G // win), win)
cov_b, _ = simulate_cov(30, rate=factor)
cov_u, _ = simulate_cov(30)
for name, cv in (("均匀", cov_u), ("有偏", cov_b)):
    print("%s: 均值 %.1f，方差/均值 %.2f，深度<10 的比例 %.4f，零覆盖比例 %.5f" % (
        name, cv.mean(), cv.var() / cv.mean(), (cv < 10).mean(), (cv == 0).mean()))

print("\n== 4. 短 contig 的平均覆盖度有多可靠？（读长 150 bp，泊松抽样）==")
print("  深度  contig长度   期望读长数λ   理论CV=1/sqrt(λ)   模拟CV")
for c in (5, 20):
    for clen in (1000, 2500, 10000):
        lam = c * clen / L
        # 每个 contig 上落下的读长数 ~ Poisson(λ)，平均覆盖度估计 = 读长数*L/clen
        est = rng.poisson(lam, size=20000) * L / clen
        print("  %3d   %6d      %8.1f          %.3f            %.3f" % (
            c, clen, lam, 1 / math.sqrt(lam), est.std() / est.mean()))
