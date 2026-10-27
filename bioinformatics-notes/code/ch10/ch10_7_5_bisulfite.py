import numpy as np
from collections import Counter
from scipy.stats import fisher_exact, beta

rng = np.random.default_rng(0)

# 1) 三字母比对的代价：C→T 转换后字母表变小，k-mer 唯一性下降，多重比对增多
genome = ''.join(rng.choice(list('ACGT'), size=1_000_000, p=[0.3, 0.2, 0.2, 0.3]))
def unique_frac(s, k):
    c = Counter(s[i:i + k] for i in range(len(s) - k + 1))
    return sum(1 for i in range(len(s) - k + 1) if c[s[i:i + k]] == 1) / (len(s) - k + 1)
for k in (12, 14, 16):
    print(f'k={k}: 原始基因组唯一 k-mer 比例 {unique_frac(genome, k):.3f}，'
          f'C→T 转换后 {unique_frac(genome.replace("C", "T"), k):.3f}')

# 2) 甲基化水平 β = M/(M+U)，低覆盖时置信区间很宽（Clopper–Pearson）
for M, U in [(3, 1), (75, 25)]:
    lo, hi = beta.ppf(0.025, M, U + 1), beta.ppf(0.975, M + 1, U)
    print(f'M={M}, U={U}: β={M / (M + U):.2f}，95% 区间 [{lo:.2f}, {hi:.2f}]')

# 3) 逐 CpG 的 Fisher 检验找 DMC，再把相邻 DMC 合并成 DMR
n_cpg = 300
pos = np.cumsum(rng.integers(5, 60, n_cpg))                 # CpG 坐标
b1 = rng.beta(8, 2, n_cpg)                                  # 样本 1：大多高甲基化
b2 = b1.copy(); b2[100:130] = rng.beta(2, 8, 30)            # 样本 2：第 100–129 个 CpG 去甲基化（真 DMR）
cov1, cov2 = rng.poisson(15, n_cpg) + 1, rng.poisson(15, n_cpg) + 1
M1, M2 = rng.binomial(cov1, b1), rng.binomial(cov2, b2)
p = np.array([fisher_exact([[M1[i], cov1[i] - M1[i]], [M2[i], cov2[i] - M2[i]]])[1] for i in range(n_cpg)])
order = np.argsort(p); adj = np.empty(n_cpg)
adj[order] = np.minimum(1, np.minimum.accumulate((p[order] * n_cpg / np.arange(1, n_cpg + 1))[::-1])[::-1])
dmc = np.where(adj < 0.05)[0]
dmrs, cur = [], [dmc[0]] if len(dmc) else []
for i in dmc[1:]:                                           # 相邻 DMC 距离 < 100bp 就并入同一个区域
    if pos[i] - pos[cur[-1]] < 100: cur.append(i)
    else: dmrs.append(cur); cur = [i]
if cur: dmrs.append(cur)
print('DMC 个数:', len(dmc), '| DMR（≥3 个 CpG）:',
      [(int(pos[d[0]]), int(pos[d[-1]]), len(d)) for d in dmrs if len(d) >= 3],
      '| 真 DMR 范围:', (int(pos[100]), int(pos[129])))
