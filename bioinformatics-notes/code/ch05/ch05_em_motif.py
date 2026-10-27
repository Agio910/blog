# -*- coding: utf-8 -*-
"""第 5 章：用 EM 找基序（OOPS 模型：每条序列恰有一个位点），复现教材 p.114 的设定：
10 条 100 nt 的 DNA，共有一个 20 nt 的结合位点。只依赖 numpy。"""
import numpy as np

B = "ACGT"
N, LSEQ, W = 10, 100, 20
NPOS = LSEQ - W + 1                      # 可能的起点数：81（教材误作 78）


def simulate(rng, p_mut=0.2):
    motif = rng.integers(0, 4, W)
    X = rng.integers(0, 4, (N, LSEQ))
    starts = rng.integers(0, NPOS, N)
    for i, s in enumerate(starts):
        site = motif.copy()
        hit = rng.random(W) < p_mut      # 每个位点约 20% 的位置发生替换
        site[hit] = (site[hit] + rng.integers(1, 4, hit.sum())) % 4
        X[i, s:s + W] = site
    return X, motif, starts


def e_step(X, theta, bg):
    """logLR[i,k] = sum_j log2(theta[j, x] / bg[x])；整条序列的背景乘积对所有 k 相同，已约掉。"""
    logr = np.log2(theta) - np.log2(bg)               # W x 4
    L = np.zeros((N, NPOS))
    for j in range(W):
        L += logr[j][X[:, j:j + NPOS]]
    m = L.max(axis=1, keepdims=True)
    Z = 2.0 ** (L - m)
    tot = Z.sum(axis=1, keepdims=True)
    loglik = float((m[:, 0] + np.log2(tot[:, 0] / NPOS)).sum())   # 相对背景模型的对数似然比（bits）
    return Z / tot, loglik


def m_step(X, Z, bg, beta=1.0):
    C = np.zeros((W, 4))
    for j in range(W):
        win = X[:, j:j + NPOS]
        for b in range(4):
            C[j, b] = (Z * (win == b)).sum()
    raw = C.copy()
    C += beta * bg                                     # 伪计数，按背景分配
    return C / C.sum(axis=1, keepdims=True), raw


def run_em(X, theta, bg, iters=500, tol=1e-8):
    ll_old = -np.inf
    for _ in range(iters):
        Z, ll = e_step(X, theta, bg)
        theta, raw = m_step(X, Z, bg)
        if ll - ll_old < tol:
            break
        ll_old = ll
    Z, ll = e_step(X, theta, bg)
    return theta, Z, ll, raw


rng = np.random.default_rng(7)
X, motif, starts = simulate(rng)
bg = np.bincount(X.ravel(), minlength=4) / X.size
print("真实基序:", "".join(B[b] for b in motif))
planted = np.array([X[i, st:st + W] for i, st in enumerate(starts)])
print("植入位点（含突变）的一致序列:", "".join(B[np.bincount(c, minlength=4).argmax()] for c in planted.T))
print("背景频率 A C G T:", np.round(bg, 3))

# 1) 教材的做法：随机挑每条序列上一个 20 nt 区段，统计频率作为初始 PSSM
print("\n== 随机初始化，20 次")
res = []
for r in range(20):
    pick = rng.integers(0, NPOS, N)
    C = np.ones((W, 4)) * 0.25                          # 0.25 作伪计数，避免 0
    for i, s in enumerate(pick):
        C[np.arange(W), X[i, s:s + W]] += 1
    theta0 = C / C.sum(axis=1, keepdims=True)
    theta, Z, ll, _ = run_em(X, theta0, bg)
    d = Z.argmax(axis=1) - starts
    shift = int(np.bincount(d - d.min()).argmax() + d.min())       # 最常见的偏移量
    res.append((ll, int((d == 0).sum()), shift, int((d == shift).sum()),
                "".join(B[b] for b in theta.argmax(axis=1))))
for ll, acc, shift, nshift, cons in sorted(res, reverse=True)[:3] + sorted(res)[:3]:
    print(f"  对数似然比 {ll:6.1f} bits，位点找对 {acc:2d}/10，"
          f"最常见偏移 {shift:+d}（{nshift} 条），一致序列 {cons}")
print("  20 次里位点全对:", sum(r[1] == N for r in res), "次；整体错位一格:",
      sum(abs(r[2]) == 1 and r[3] == N for r in res), "次")

# 2) MEME 的思路：拿真实出现过的子串当起点，各跑几步，挑最好的再跑到收敛
print("\n== 以序列 1 的每个 20 nt 子串为起点")
best = None
for s in range(NPOS):
    theta0 = np.full((W, 4), 0.5 / 3)
    theta0[np.arange(W), X[0, s:s + W]] = 0.5
    theta, Z, ll, _ = run_em(X, theta0, bg, iters=3)
    if best is None or ll > best[0]:
        best = (ll, theta0)
theta, Z, ll, raw = run_em(X, best[1], bg)
acc = int((Z.argmax(axis=1) == starts).sum())
print(f"  收敛后对数似然比 {ll:.1f} bits，位点找对 {acc}/10")
print("  推断一致序列:", "".join(B[b] for b in theta.argmax(axis=1)))
for j in np.where(theta.argmax(axis=1) != motif)[0]:
    print(f"  第 {j + 1} 列与真实基序不同：植入位点 A/C/G/T 计数 {np.bincount(planted[:, j], minlength=4)}，"
          f"theta = {np.round(theta[j], 3)}")
print("  M 步每列期望计数之和（不含伪计数）:", np.round(raw.sum(axis=1)[:5], 6), "...")
