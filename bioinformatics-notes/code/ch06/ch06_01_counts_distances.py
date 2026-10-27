# -*- coding: utf-8 -*-
"""第 6 章 代码 1：二叉树的数目；p 距离、JC69、K2P；用矩阵指数核对教材式（6.1）（6.4）；
由表 6.4 的差异数反推序列长度。只依赖 numpy、scipy。"""
from math import log, prod
import numpy as np
from scipy.linalg import expm


def fmt(v):
    return f"{v:,}" if v < 10**12 else f"{float(v):.3e}"


# ---------- 1. 二叉树的数目 ----------
def n_rooted(n):            # (2n-3)!!
    return prod(range(1, 2 * n - 2, 2))


def n_unrooted(n):          # (2n-5)!!
    return prod(range(1, 2 * n - 4, 2)) if n >= 3 else 1


print("== 1. 二叉树数目 ==")
print(f"{'n':>3} {'无根 (2n-5)!!':>16} {'有根 (2n-3)!!':>16} {'2^n':>12}")
for n in [3, 4, 5, 6, 8, 10, 20, 50]:
    print(f"{n:>3} {fmt(n_unrooted(n)):>16} {fmt(n_rooted(n)):>16} {fmt(2**n):>12}")
assert all(n_rooted(n) == n_unrooted(n + 1) for n in range(3, 30))

# ---------- 2. 距离公式与教材例子 ----------
def jc69(p):
    return -0.75 * np.log(1 - 4 * p / 3)


def k2p(P, Q):
    return -0.5 * np.log(1 - 2 * P - Q) - 0.25 * np.log(1 - 2 * Q)


print("\n== 2. 兔/鸡 beta-球蛋白（教材 p.129：438 bp，58 个转换，63 个颠换）==")
L, ts, tv = 438, 58, 63
P, Q = ts / L, tv / L
p = P + Q
q = 1 - p
print(f"P = {P:.4f}, Q = {Q:.4f}, p = {p:.4f}, q = {q:.4f}")
print(f"p 距离       = {p:.4f}")
print(f"JC69         = {jc69(p):.4f}   (教材 0.345)")
print(f"K2P          = {k2p(P, Q):.4f}   (教材 0.351)")
print(f"K2P(P,Q 先舍入成 0.132/0.144) = {k2p(0.132, 0.144):.4f}")
print(f"式(6.1) 印刷版 3/4*ln(4/(4q-1)) = {0.75 * log(4 / (4 * q - 1)):.4f}")
print(f"式(6.1) 正确版 3/4*ln(3/(4q-1)) = {0.75 * log(3 / (4 * q - 1)):.4f}")

# ---------- 3. 用 K2P 速率矩阵的矩阵指数核对式（6.3）（6.4）（6.5）----------
print("\n== 3. K2P：矩阵指数 vs 教材公式（两条序列各自从祖先分化 t）==")
alpha, beta, t = 0.30, 0.10, 0.5           # 每条谱系的转换速率、单个颠换方向的速率
bases = "AGCT"                              # A,G 为嘌呤；C,T 为嘧啶
purine = {"A", "G"}
R = np.zeros((4, 4))
for i, x in enumerate(bases):
    for j, y in enumerate(bases):
        if i != j:
            R[i, j] = alpha if ((x in purine) == (y in purine)) else beta
np.fill_diagonal(R, -R.sum(axis=1))
Pm = expm(R * 2 * t)                        # 序列 1 -> 祖先 -> 序列 2，共 2t
pi = np.full(4, 0.25)
joint = pi[:, None] * Pm                    # 两条序列在一个位点上的联合分布
is_ts = np.array([[i != j and ((x in purine) == (y in purine)) for j, y in enumerate(bases)]
                  for i, x in enumerate(bases)])
is_tv = np.array([[(x in purine) != (y in purine) for y in bases] for x in bases])
PI, PII = joint[is_ts].sum(), joint[is_tv].sum()
print(f"矩阵指数：P_I = {PI:.6f}, P_II = {PII:.6f}")
print(f"式(6.3)：1/4[1-2e^(-4(a+b)t)+e^(-8bt)] = "
      f"{0.25 * (1 - 2 * np.exp(-4 * (alpha + beta) * t) + np.exp(-8 * beta * t)):.6f}")
print(f"式(6.4) 印刷版 1/4(1-e^(-8bt)) = {0.25 * (1 - np.exp(-8 * beta * t)):.6f}")
print(f"式(6.4) 正确版 1/2(1-e^(-8bt)) = {0.5 * (1 - np.exp(-8 * beta * t)):.6f}")
print(f"式(6.5) 由 P_I、P_II 反推 K = {k2p(PI, PII):.6f}；真值 2(a+2b)t = {2 * (alpha + 2 * beta) * t:.6f}")

# ---------- 4. 表 6.4：由差异数反推序列长度 ----------
print("\n== 4. 表 6.4：差异数 -> JC 距离，哪个序列长度能复现教材的三位小数 ==")
pairs = ["hu-ch", "hu-go", "ch-go", "hu-or", "ch-or", "go-or", "hu-gi", "ch-gi", "go-gi", "or-gi"]
diffs = np.array([1, 3, 2, 9, 8, 6, 12, 11, 11, 11])
book = np.array([0.015, 0.045, 0.030, 0.143, 0.126, 0.092, 0.198, 0.179, 0.179, 0.179])
ok = [Ls for Ls in range(40, 201) if np.all(np.round(jc69(diffs / Ls), 3) == book)]
print("能逐一复现 10 个距离的长度：", ok)
Ls = ok[0]
d = jc69(diffs / Ls)
se = np.sqrt((diffs / Ls) * (1 - diffs / Ls) / (Ls * (1 - 4 * diffs / Ls / 3) ** 2))
for name, k, dd, s_ in zip(pairs, diffs, d, se):
    print(f"{name:6s} 差异 {k:2d}  JC = {dd:.4f}  标准误 = {s_:.4f}")
print(f"JC 在 p -> 3/4 时发散：p = 0.70 时 d = {jc69(0.70):.3f}，p = 0.74 时 d = {jc69(0.74):.3f}")
