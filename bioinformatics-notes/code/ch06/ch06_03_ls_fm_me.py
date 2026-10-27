# -*- coding: utf-8 -*-
"""第 6 章 代码 3：拓扑矩阵 A（教材 p.136 的 d = AL + ε），最小二乘枝长，
Fitch–Margoliash 百分标准差（式 6.6），最小进化（ME）准则下比较 5 个物种的全部 15 种无根树。"""
from itertools import combinations
import numpy as np

NAMES = ["hu", "ch", "go", "or", "gi"]
D = np.array([[0.000, 0.015, 0.045, 0.143, 0.198],
              [0.015, 0.000, 0.030, 0.126, 0.179],
              [0.045, 0.030, 0.000, 0.092, 0.179],
              [0.143, 0.126, 0.092, 0.000, 0.179],
              [0.198, 0.179, 0.179, 0.179, 0.000]])
PAIRS = list(combinations(range(5), 2))
d = np.array([D[i, j] for i, j in PAIRS])


def topologies():
    """5 个物种的无根二叉树 ((a,b),c,(d,e))：选中间的 c（5 种）× 其余四个分成两对（3 种）= 15。"""
    out = []
    for c in range(5):
        o = [x for x in range(5) if x != c]
        for p1, p2 in [((o[0], o[1]), (o[2], o[3])), ((o[0], o[2]), (o[1], o[3])), ((o[0], o[3]), (o[1], o[2]))]:
            out.append((p1, c, p2))
    return out


def design(top):
    """行 = 物种对，列 = 7 条枝（5 条末端枝 + 樱桃 p1 的内枝 + 樱桃 p2 的内枝）。"""
    p1, c, p2 = top
    A = np.zeros((len(PAIRS), 7))
    for r, (i, j) in enumerate(PAIRS):
        A[r, i] = A[r, j] = 1
        A[r, 5] = (i in p1) != (j in p1)
        A[r, 6] = (i in p2) != (j in p2)
    return A


def newick(top):
    (a, b), c, (e, f) = top
    n = NAMES
    return f"(({n[a]},{n[b]}),{n[c]},({n[e]},{n[f]}))"


def s_percent(dv, ev, denom):
    return 100 * np.sqrt(np.sum(((dv - ev) / dv) ** 2) / denom)


n = 5
# ---------- 1. 图 6.10 的 FM 树：路径长度、表 6.7、式 (6.6) ----------
top_fm = ((0, 1), 2, (3, 4))
A = design(top_fm)
L_fig = np.array([0.016, 0.000, 0.006, 0.057, 0.122, 0.024, 0.044])  # hu,ch,go,or,gi, C-E, E-G
e_fig = A @ L_fig
e_tab = np.array([0.016, 0.046, 0.141, 0.208, 0.030, 0.125, 0.192, 0.107, 0.174, 0.181])  # 表 6.7 下三角
print("== 图 6.10 的树：树上路径长度 vs 表 6.7 ==")
for (i, j), dv, ef, et in zip(PAIRS, d, e_fig, e_tab):
    flag = "" if abs(ef - et) < 5e-4 else "  <- 不一致"
    print(f"{NAMES[i]}-{NAMES[j]}: 观测 {dv:.3f}  图6.10枝长之和 {ef:.3f}  表6.7 {et:.3f}{flag}")
for lab, ev in [("图 6.10 枝长", e_fig), ("表 6.7 数值", e_tab)]:
    print(f"{lab}: s = {s_percent(d, ev, n * (n - 1)):.2f}（分母 n(n-1)，教材式 6.6）, "
          f"{s_percent(d, ev, n * (n - 1) / 2):.2f}（分母 n(n-1)/2）；教材写 1.94")

# 同一拓扑下让 s 最小的枝长（Fitch–Margoliash 准则 = 以 1/d^2 为权的加权最小二乘）
L_fm, *_ = np.linalg.lstsq(A / d[:, None], np.ones(len(d)), rcond=None)
print("FM 最优枝长（hu,ch,go,or,gi,C-E,E-G）:", " ".join(f"{x:.4f}" for x in L_fm),
      f"  s = {s_percent(d, A @ L_fm, n * (n - 1)):.2f} / {s_percent(d, A @ L_fm, n * (n - 1) / 2):.2f}")

# ---------- 2. 最小进化：15 种拓扑的 OLS 枝长总和 S ----------
print("\n== 15 种无根树：OLS 总枝长 S（ME 准则）与 FM 最优 s（分母 n(n-1)）==")
rows = []
for top in topologies():
    A = design(top)
    L_ols, *_ = np.linalg.lstsq(A, d, rcond=None)              # = (A^T A)^{-1} A^T d
    L_w, *_ = np.linalg.lstsq(A / d[:, None], np.ones(len(d)), rcond=None)
    rows.append((L_ols.sum(), s_percent(d, A @ L_w, n * (n - 1)), newick(top), L_ols))
rows.sort(key=lambda x: x[0])
for S, s, nw, L_ols in rows[:5]:
    print(f"S = {S:.4f}   s = {s:5.2f}   {nw}")
print("...（其余 10 种 S 更大）")
S0, s0, nw0, L0 = rows[0]
print("ME 树的 OLS 枝长（hu,ch,go,or,gi,两条内枝）:", " ".join(f"{x:.4f}" for x in L0))
