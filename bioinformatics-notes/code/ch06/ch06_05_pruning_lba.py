# -*- coding: utf-8 -*-
"""第 6 章 代码 5：Felsenstein 剪枝算法（JC69，可选离散 Γ），滑轮原理的数值验证，
以及 4 个物种的长枝吸引模拟：最大简约、NJ（p 距离 / JC 距离）、ML（JC / JC+Γ）各自是否找回真树，
并用自举法说明"高支持率不等于正确"。依赖 numpy、scipy；全程约 1 分钟。"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import gamma as gamma_dist


def p_jc(b):
    """JC69 转移概率矩阵；b = 每位点期望替换数。"""
    e = np.exp(-4.0 * b / 3.0)
    P = np.full((4, 4), 0.25 - 0.25 * e)
    np.fill_diagonal(P, 0.25 + 0.75 * e)
    return P


def partial(node, pats, rate=1.0):
    """剪枝递推：叶子是 pats 的行号；内部节点是 [(子节点, 枝长), ...]。
    返回形状 (位点数, 4) 的条件似然 L_node[x] = prod_子 sum_y P_xy(b) L_子[y]。"""
    m = pats.shape[1]
    if isinstance(node, int):
        L = np.zeros((m, 4))
        L[np.arange(m), pats[node]] = 1.0
        return L
    out = np.ones((m, 4))
    for child, b in node:
        out *= partial(child, pats, rate) @ p_jc(b * rate).T
    return out


def loglik(tree, pats, counts, rates=(1.0,)):
    """对根节点按平稳分布 1/4 求和；rates 为离散 Γ 各类的相对速率（等权）。"""
    site = sum(partial(tree, pats, r).sum(axis=1) * 0.25 for r in rates) / len(rates)
    return float(np.sum(counts * np.log(site)))


def discrete_gamma(alpha, k=4):
    """Yang (1994) 的离散 Γ：每类取该分位区间的条件均值，均值为 1。"""
    cut = gamma_dist.ppf(np.arange(1, k) / k, a=alpha, scale=1 / alpha)
    F = gamma_dist.cdf(np.r_[0, cut, np.inf], a=alpha + 1, scale=1 / alpha)
    return k * np.diff(F)


def evolve(parent, b, r, rng):
    keep = rng.random(parent.size) < 0.25 + 0.75 * np.exp(-4 * b * r / 3)
    return np.where(keep, parent, (parent + rng.integers(1, 4, parent.size)) % 4)


def simulate4(m, br, rng, alpha=None):
    """真树 ((0,1),(2,3))；br = (b0, b1, b2, b3, 内枝)。alpha 给定时位点速率 ~ Γ(alpha)，均值 1。"""
    r = rng.gamma(alpha, 1 / alpha, m) if alpha else np.ones(m)
    u = rng.integers(0, 4, m)
    v = evolve(u, br[4], r, rng)
    return np.stack([evolve(u, br[0], r, rng), evolve(u, br[1], r, rng),
                     evolve(v, br[2], r, rng), evolve(v, br[3], r, rng)])


SPLITS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]   # 第 0 个是真树


def compress(X):
    """把相同的位点模式合并：返回（模式矩阵，每种模式出现的次数）。"""
    return np.unique(X, axis=1, return_counts=True)


def mp4(X):
    """4 个物种时，信息位点只有 xxyy 型，每个位点支持一种分割；支持数最多者为简约树。"""
    s = [np.sum((X[i] == X[j]) & (X[k] == X[l]) & (X[i] != X[k])) for (i, j), (k, l) in SPLITS]
    return int(np.argmax(s)) if s.count(max(s)) == 1 else -1


def nj4(X, correct):
    """4 个物种时 NJ 等价于四点法：选 d_ij + d_kl 最小的分割。"""
    p = np.array([[np.mean(X[i] != X[j]) for j in range(4)] for i in range(4)])
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.where(p < 0.75, -0.75 * np.log(1 - 4 * p / 3), np.inf) if correct else p
    s = [D[i, j] + D[k, l] for (i, j), (k, l) in SPLITS]
    return int(np.argmin(s)) if np.isfinite(min(s)) and s.count(min(s)) == 1 else -1


def ml4(X, use_gamma=False):
    pats, cnt = compress(X)
    best = []
    for (i, j), (k, l) in SPLITS:
        def nll(th):
            b = np.exp(th[:5])
            rates = discrete_gamma(np.exp(th[5])) if use_gamma else (1.0,)
            tree = [(i, b[i]), (j, b[j]), ([(k, b[k]), (l, b[l])], b[4])]   # 根放在 (i,j) 一侧的内部节点
            return -loglik(tree, pats, cnt, rates)
        x0 = np.r_[np.log(np.full(5, 0.2)), [0.0] * use_gamma]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=[(-12, 3)] * 5 + [(-4, 4)] * use_gamma)
        best.append(-res.fun)
    return int(np.argmax(best))


# ---------- 1. 滑轮原理：根在内枝上滑动，似然不变 ----------
rng = np.random.default_rng(0)
X3 = simulate4(500, (0.1, 0.2, 0.3, 0.1, 0.1), rng)[:3]
pats3, cnt3 = compress(X3)
print("== 1. 滑轮原理（三条序列，v1=0.1, v2=0.2, v3+v4=0.4）==")
for v4 in [0.0, 0.1, 0.2, 0.3, 0.4]:
    tree = [([(0, 0.1), (1, 0.2)], v4), (2, 0.4 - v4)]
    print(f"v4 = {v4:.1f}, v3 = {0.4 - v4:.1f}:  lnL = {loglik(tree, pats3, cnt3):.6f}")

# ---------- 2. 两条序列：枝长的最大似然估计就是 JC 距离 ----------
a, b = X3[0], X3[1]
pats2, cnt2 = compress(np.stack([a, b]))
fit = minimize_scalar(lambda t: -loglik([(0, t), (1, 0.0)], pats2, cnt2), bounds=(1e-6, 5), method="bounded")
p = np.mean(a != b)
print("\n== 2. 两条序列 ==")
print(f"p = {p:.4f}；ML 估计的 v1+v2 = {fit.x:.4f}；JC 距离 = {-0.75 * np.log(1 - 4 * p / 3):.4f}；"
      f"教材记号 v = 4K/3 = {4 / 3 * fit.x:.4f}")

# 三条序列：式 (6.25)–(6.30) 用两两 JC 距离解星形树枝长，与真正的 ML 估计比较
K = {(i, j): -0.75 * np.log(1 - 4 * np.mean(X3[i] != X3[j]) / 3) for i in range(3) for j in range(i + 1, 3)}
pw = np.array([K[0, 1] + K[0, 2] - K[1, 2], K[0, 1] + K[1, 2] - K[0, 2], K[0, 2] + K[1, 2] - K[0, 1]]) / 2
star = lambda v: [(0, v[0]), (1, v[1]), (2, v[2])]
fit3 = minimize(lambda th: -loglik(star(np.exp(th)), pats3, cnt3), np.log(pw), method="L-BFGS-B")
print("三条序列（真枝长 0.1, 0.2, 0.4）：两两 JC 解 =", np.round(pw, 4),
      f"lnL = {loglik(star(pw), pats3, cnt3):.4f}；ML =", np.round(np.exp(fit3.x), 4), f"lnL = {-fit3.fun:.4f}")

# ---------- 3. 长枝吸引：真树 ((0,1),(2,3))，0 与 2 是长枝 ----------
br = (0.75, 0.05, 0.75, 0.05, 0.05)
print("\n== 3. 长枝吸引（每格 50 次重复，数值为找回真树的比例）==")
print("数据按 JC69 模拟（所有位点同速率）")
for m in [200, 1000, 5000]:
    hit = {"MP": 0, "NJ(p)": 0, "NJ(JC)": 0, "ML(JC)": 0}
    for _ in range(50):
        X = simulate4(m, br, rng)
        hit["MP"] += mp4(X) == 0
        hit["NJ(p)"] += nj4(X, False) == 0
        hit["NJ(JC)"] += nj4(X, True) == 0
        hit["ML(JC)"] += ml4(X) == 0
    print(f"  L = {m:5d}: " + "  ".join(f"{k} {v / 50:.2f}" for k, v in hit.items()))
print("数据按 JC69+Γ 模拟（alpha = 0.5，位点间速率差异大）")
for m in [1000, 5000]:
    hit = {"MP": 0, "NJ(JC)": 0, "ML(JC)": 0, "ML(JC+Γ)": 0}
    for _ in range(50):
        X = simulate4(m, br, rng, alpha=0.5)
        hit["MP"] += mp4(X) == 0
        hit["NJ(JC)"] += nj4(X, True) == 0
        hit["ML(JC)"] += ml4(X) == 0
        hit["ML(JC+Γ)"] += ml4(X, use_gamma=True) == 0
    print(f"  L = {m:5d}: " + "  ".join(f"{k} {v / 50:.2f}" for k, v in hit.items()))

# ---------- 4. 自举支持率：错的分割也能得到 100% ----------
X = simulate4(5000, br, rng, alpha=0.5)
print("\n== 4. 一份 JC+Γ 数据（L=5000）的 100 次自举，各分割得到的支持率 ==")
names = ["01|23（真）", "02|13", "03|12"]
for label, f in [("MP", mp4), ("ML(JC)", ml4), ("ML(JC+Γ)", lambda Y: ml4(Y, True))]:
    votes = np.zeros(3)
    for _ in range(100):
        idx = rng.integers(0, X.shape[1], X.shape[1])     # 对位点有放回重抽样
        k = f(X[:, idx])
        if k >= 0:
            votes[k] += 1
    print(f"  {label:9s} " + "  ".join(f"{n} {v:3.0f}%" for n, v in zip(names, votes)))
