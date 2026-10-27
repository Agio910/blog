# -*- coding: utf-8 -*-
"""第 5 章：熵、信息量与 PSSM。
1) 复核表 5.4、表 5.5；2) 从图 5.6 的联配重新计算 Kinase2 基序；
3) 带伪计数的 PSSM 与打分；4) 小样本偏差；5) 图 5.9 的 λ 操纵基因位点；6) 表 5.2 的几处数字。
只依赖 numpy。"""
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
rng = np.random.default_rng(0)


def H(p):
    """香农熵（比特），p 为频率向量。"""
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) + 0.0


def col_freq(col, alphabet):
    return np.array([col.count(a) for a in alphabet], float) / len(col)


# ---------- 1. 表 5.4（核苷酸，Hmax = 2）----------
print("== 表 5.4")
for name, col in zip(["#1", "#2", "#3"], ["GGGG", "AATT", "GACT"]):
    h = H(col_freq(col, "ACGT"))
    print(f"{name} {col}: H = {h:.2f}, IC = {2 - h:.2f}")

# ---------- 2. 表 5.5 按教材印刷的频率 ----------
print("\n== 表 5.5（教材频率）")
tab55 = [{"F": .8, "Y": .2}, {"F": .2, "Y": .8}, {"I": .4, "L": .2, "V": .4},
         {"L": .2, "V": .8}, {"I": .4, "L": .4, "M": .2}, {"D": 1.0}, {"D": 1.0},
         {"I": .2, "V": .8}, {"W": 1.0}]
printed = [3.59, 3.59, 2.80, 3.59, 2.80, 4.32, 4.32, 3.59, 4.32]
hmax = np.log2(20)
ic_book = [hmax - H(list(c.values())) for c in tab55]
print("精确 IC :", " ".join(f"{x:.4f}" for x in ic_book))
print("教材印刷:", " ".join(f"{x:.2f}" for x in printed))
print(f"合计：精确 {sum(ic_book):.4f}；教材 {sum(printed):.2f}")

# ---------- 3. 直接从图 5.6 的联配计算 ----------
aln = {"CZ1": "FFVVMDDVW", "Xa1": "FLIVLDDVW", "Prf": "FLILIDDVW",
       "rx": "YLVVIDDIW", "Cre3": "FILVLDDVW"}
seqs = list(aln.values())
N, L = len(seqs), len(seqs[0])
cols = ["".join(s[j] for s in seqs) for j in range(L)]
print("\n== 由图 5.6 联配重算")
ic_aln = []
for j, c in enumerate(cols, 1):
    f = col_freq(c, AA)
    ic = hmax - H(f)
    ic_aln.append(ic)
    comp = ", ".join(f"{a}:{c.count(a) / N:.1f}" for a in sorted(set(c)))
    print(f"#{j} {c}  {comp:<22s} IC = {ic:.3f}")
print(f"合计 IC = {sum(ic_aln):.2f}")
cons = "".join(max(sorted(set(c)), key=c.count) for c in cols)
regex = "-".join(c[0] if len(set(c)) == 1 else "[" + "".join(sorted(set(c))) + "]" for c in cols)
print("一致序列（并列时取字母序靠前者）:", cons)
print("列出每列全部残基的正则表达式:", regex)

# ---------- 4. 带伪计数的 PSSM（背景取均匀 1/20）----------
print("\n== PSSM：q = (n + beta*p)/(N + beta)，s = log2(q/p)，p = 1/20，beta = 1")
p_bg = np.full(20, 1 / 20)
beta = 1.0
counts = np.array([[c.count(a) for a in AA] for c in cols], float)   # L x 20
q = (counts + beta * p_bg) / (N + beta)
S = np.log2(q / p_bg)                                                 # L x 20
for j in range(L):
    obs = [(AA[b], S[j, b]) for b in range(20) if counts[j, b] > 0]
    unseen = S[j, counts[j] == 0][0] if (counts[j] == 0).any() else None
    print(f"#{j + 1} " + "  ".join(f"{a}:{s:+.2f}" for a, s in obs)
          + (f"  未出现残基:{unseen:+.2f}" if unseen is not None else ""))
idx = {a: i for i, a in enumerate(AA)}


def score(w, M):
    return float(sum(M[j, idx[a]] for j, a in enumerate(w)))


for name, s in aln.items():
    print(f"{name:5s} {s}  得分 {score(s, S):6.2f} bits")
print(f"最高可能得分 {S.max(axis=1).sum():.2f} bits")
# 没有伪计数时，第 7 列出现一次 E 就让得分变成负无穷
with np.errstate(divide="ignore"):
    S0 = np.log2((counts / N) / p_bg)
mut = "FLVVIDEVW"
print(f"{mut}（第 7 列 D→E）：无伪计数 {score(mut, S0)}，beta=1 时 {score(mut, S):.2f}")
# 背景随机 9 肽的得分分布（蒙特卡罗）
R = rng.integers(0, 20, size=(1_000_000, L))
rand_scores = S[np.arange(L), R].sum(axis=1)
thr = min(score(s, S) for s in seqs)
print(f"随机 9 肽：平均 {rand_scores.mean():.2f}，99.9% 分位 {np.quantile(rand_scores, 0.999):.2f}，"
      f"P(得分 >= {thr:.2f}) = {(rand_scores >= thr).mean():.1e}")
# 每列相对熵 D(q||p)：背景均匀时它等于 log2(20) - H(q)，即带伪计数的“信息量”
D = (q * np.log2(q / p_bg)).sum(axis=1)
print("每列相对熵 D(q||p):", " ".join(f"{x:.2f}" for x in D), f"| 合计 {D.sum():.2f}")

# ---------- 5. 小样本偏差 ----------
print("\n== 小样本偏差：完全随机的列，插件估计的 IC 期望是多少")
for K, n in [(20, 5), (4, 12), (20, 50), (4, 100)]:
    X = rng.integers(0, K, size=(200_000, n))
    cnt = np.stack([(X == k).sum(axis=1) for k in range(K)], axis=1) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        hh = -np.nansum(np.where(cnt > 0, cnt * np.log2(cnt), 0.0), axis=1)
    ic = np.log2(K) - hh
    approx = (K - 1) / (2 * n * np.log(2))
    print(f"K={K:2d}, n={n:3d}: 平均 IC {ic.mean():.3f}，最小 {ic.min():.3f}，"
          f"二阶近似 (K-1)/(2n ln2) = {approx:.3f}")

# ---------- 6. 图 5.9：12 条 λ cI/cro 结合位点（从教材图转写，位置 -9..+9）----------
lam = ["GTATCACCGCCAGTGGTAT", "ATACCACTGGCGGTGATAC", "TCAACACCGCCAGAGATAA",
       "TTATCTCTGGCGGTGTTGA", "TTATCACCGCAGATGGTTA", "TAACCATCTGCGGTGATAA",
       "CTATCACCGCAAGGGATAA", "TTATCCCTTGCGGTGATAG", "CTAACACCGTGCGTGTTGA",
       "TCAACACGCACGGTGTTAG", "TTACCTCTGGCGGTGATAA", "TTATCACCGCCAGAGGTAA"]
rc = lambda s: s[::-1].translate(str.maketrans("ACGT", "TGCA"))
pairs = [(i + 1, lam.index(rc(s)) + 1) for i, s in enumerate(lam) if rc(s) in lam]
print("\n== 图 5.9：互为反向互补的序列对", pairs[::2])
pos = list(range(-9, 10))
icl = []
for j, x in enumerate(pos):
    c = "".join(s[j] for s in lam)
    icl.append(2 - H(col_freq(c, "ACGT")))
print("位置:", " ".join(f"{x:+d}" for x in pos))
print("IC  :", " ".join(f"{v:.1f}" for v in icl))
print("只出现一种碱基的位置:", [f"{x:+d}{lam[0][j]}" for j, x in enumerate(pos) if icl[j] == 2.0])

# ---------- 7. 表 5.2 的几处数字 ----------
print("\n== 表 5.2")
t52 = {"背景": [.27, .25, .25, .23], "列1": [.4, .4, .2, .2], "列2": [.1, .1, .1, .7],
       "列20": [.2, .2, .4, .2]}                               # 顺序 G C A T
print("各列之和:", {k: round(sum(v), 2) for k, v in t52.items()})
print("G 背景频率 224/800 =", 224 / 800)
print("长 100、宽 20 的可能起点数 =", 100 - 20 + 1)
lr = (0.2 / 0.25) * (0.7 / 0.23)
print(f"窗口前两位 A、T 的似然比贡献 = (0.2/0.25)*(0.7/0.23) = {lr:.3f}，即 {np.log2(lr):.2f} bits")
