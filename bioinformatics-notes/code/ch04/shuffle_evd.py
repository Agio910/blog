"""第 4 章：用打乱序列估计局部联配得分的零分布，检验极值（Gumbel）分布与 Karlin–Altschul 参数。

1) 向量化的 Smith–Waterman（BLOSUM62，空位 11/1），一次比对一批等长序列；
2) 造一对只有 50 个残基同源的"弱同源"序列，用 1000 次打乱求经验 P 值，并与正态近似、Gumbel 和 E = Kmn·e^{-λS} 对比；
3) 用 2000 对随机序列拟合 Gumbel，看有空位、无空位两种情况下的 λ。
运行：python shuffle_evd.py    依赖：numpy、scipy、Biopython；同目录的 nw_sw.py（只用于自检）
"""
import numpy as np
from scipy.stats import gumbel_r, norm
from Bio.Align import substitution_matrices
from nw_sw import gotoh

AA = "ARNDCQEGHILKMFPSTWYV"
RR = np.array([78.05, 51.29, 44.87, 53.64, 19.25, 42.64, 62.95, 73.77, 21.99, 51.42,
               90.19, 57.44, 22.43, 38.56, 52.03, 71.20, 58.41, 13.30, 32.16, 64.41])
p = RR / RR.sum()                                   # Robinson & Robinson 背景频率
B62 = substitution_matrices.load("BLOSUM62")
S = np.array([[int(B62[a, b]) for b in AA] for a in AA])
rng = np.random.default_rng(42)


def sw_batch(q, D, o=11, e=1, gapped=True):
    """q：查询序列（整数编码，长 m）；D：B 条等长目标序列（B×n）。返回每条的最优局部得分。
    水平空位用"累计最大值"一次算完一整行：Y[j] = max_{k<j}(H0[k] + e·k) − o − e·j。"""
    B, n = D.shape
    Hp = np.zeros((B, n + 1)); Xp = np.full((B, n), -np.inf); best = np.zeros(B)
    ar = e * np.arange(1, n + 1)
    for i in range(len(q)):
        M = Hp[:, :-1] + S[q[i]][D]                         # 对角：q_i 与 d_j 配对
        if not gapped:
            H = np.maximum(M, 0)
        else:
            X = np.maximum(Hp[:, 1:] - o - e, Xp - e)       # 竖直空位（来自上一行）
            H0 = np.maximum(np.maximum(M, X), 0)
            G = np.concatenate([np.zeros((B, 1)), H0 + ar], axis=1)
            Y = np.maximum.accumulate(G, axis=1)[:, :-1] - o - ar
            H = np.maximum(H0, Y); Xp = X
        best = np.maximum(best, H.max(1))
        Hp = np.concatenate([np.zeros((B, 1)), H], axis=1)
    return best


def gumbel_fit(x, m, n):
    """最大似然拟合 Gumbel（scipy.stats.gumbel_r）：λ = 1/尺度，K = e^{λμ}/(mn)。"""
    mu, scale = gumbel_r.fit(x)
    lam = 1 / scale
    return lam, np.exp(lam * mu) / (m * n), mu


# (0) 自检：向量化版本与逐格回溯版本（nw_sw.gotoh）得分一致
q = rng.choice(20, 60, p=p); D = rng.choice(20, (5, 50), p=p)
txt = lambda v: "".join(AA[i] for i in v)
ref = [gotoh(txt(q), txt(d), lambda a, b: S[AA.index(a), AA.index(b)], 11, 1, local=True)[0] for d in D]
print("向量化 SW 与 nw_sw.gotoh 一致:", np.allclose(sw_batch(q, D), ref))

# (1) 一对"弱同源"序列：目标序列长 200，只有中间 50 个残基与查询序列同源。同源段按
#     BLOSUM62 隐含的目标频率 q_ij = p_i p_j e^{λ s_ij} 逐位抽样（期望一致度约 30%），两侧是无关序列
lam_u = 0.3176
target = np.outer(p, p) * np.exp(lam_u * S); target /= target.sum()
cond = target / target.sum(1, keepdims=True)
m = n = 200
query = rng.choice(20, m, p=p)
seg = np.array([rng.choice(20, p=cond[a]) for a in query[75:125]])
subj = np.concatenate([rng.choice(20, 75, p=p), seg, rng.choice(20, 75, p=p)])
s_obs = sw_batch(query, subj[None, :])[0]
null = sw_batch(query, np.array([rng.permutation(subj) for _ in range(1000)]))  # 打乱保持组成
lam, K, mu = gumbel_fit(null, m, n)
z = (s_obs - null.mean()) / null.std()
print("\n同源段一致度 %.2f；SW 得分 S = %d" % (np.mean(query[75:125] == seg), s_obs))
print("1000 次打乱：均值 %.1f，标准差 %.1f，最大 %d，偏度 %.2f（正态 0，Gumbel 1.14）"
      % (null.mean(), null.std(), null.max(), ((null - null.mean()) ** 3).mean() / null.std() ** 3))
print("经验 P = %.3f；按正态（z = %.1f）P = %.1e；按拟合的 Gumbel（λ = %.3f，K = %.3f）P = %.1e"
      % ((null >= s_obs).mean(), z, norm.sf(z), lam, K, gumbel_r.sf(s_obs, mu, 1 / lam)))
print("按表 4.6 的有空位参数（λ = 0.267，K = 0.041）：E = Kmn·e^{-λS} = %.1e" % (0.041 * m * n * np.exp(-0.267 * s_obs)))
tail = [(x, (null >= x).mean(), norm.sf((x - null.mean()) / null.std()), gumbel_r.sf(x, mu, 1 / lam)) for x in (35, 40, 45, 50)]
print("尾部（阈值, 经验, 正态, Gumbel）:", [(x, round(float(a), 3), round(float(b), 4), round(float(c), 3)) for x, a, b, c in tail])

# (2) 2000 对独立随机序列：长度对拟合出的 λ 的影响
for L in (100, 400):
    qq = rng.choice(20, L, p=p); DD = rng.choice(20, (2000, L), p=p)
    for gapped in (False, True):
        lam, K, _ = gumbel_fit(sw_batch(qq, DD, gapped=gapped), L, L)
        print("长度 %3d，%s：λ = %.3f，K = %.3f" % (L, "有空位 11/1" if gapped else "无空位    ", lam, K))
print("理论/表 4.6：无空位 λ = 0.3176，K = 0.134；有空位 λ = 0.267，K = 0.041")
