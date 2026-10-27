import numpy as np
from scipy import stats
from scipy.stats import rankdata

rng = np.random.default_rng(0)
# 30 个样本；3 个共表达模块各 20 个基因 + 3000 个只有噪声的背景基因
n_s, sizes, n_bg = 30, [20, 20, 20], 3000
X = rng.normal(size=(sum(sizes) + n_bg, n_s))
start = 0
for s in sizes:                                        # 同一模块的基因共享一个潜在驱动信号
    X[start:start + s] += 1.5 * rng.normal(size=n_s)
    start += s
module = np.repeat([0, 1, 2, -1], sizes + [n_bg])

r = np.corrcoef(X)
np.fill_diagonal(r, -np.inf)
iu = np.triu_indices(len(X), 1)
same = (module[iu[0]] == module[iu[1]]) & (module[iu[0]] >= 0)
noise = (module[iu[0]] < 0) & (module[iu[1]] < 0)
K = int(same.sum())

# 1) 只有 30 个样本时，纯噪声基因对里有多少能“碰巧”达到 r ≥ 0.5
t = 0.5 * np.sqrt(n_s - 2) / np.sqrt(1 - 0.5 ** 2)
print(f'噪声基因对 {noise.sum():,} 个；理论上 r≥0.5 的有 {noise.sum() * stats.t.sf(t, n_s - 2):,.0f} 个，'
      f'实际 {int((r[iu][noise] >= 0.5).sum()):,} 个')
t_bonf = stats.t.isf(0.05 / len(iu[0]), n_s - 2)           # 对全部基因对做 Bonferroni 时需要的 t
print(f'对 {len(iu[0]):,} 个基因对做 Bonferroni，r 至少要 {t_bonf / np.sqrt(t_bonf ** 2 + n_s - 2):.2f}')

# 2) 互排位 MR = √(rank(a→b)·rank(b→a))，以及 Wisecaver 的指数衰减边权阈值
rank = rankdata(-r, axis=1)
mr = np.sqrt(rank * rank.T)
print('MR 阈值 1+τ·ln100:', {tau: round(float(1 + tau * np.log(100)), 1) for tau in (5, 10, 25, 50, 100)})
for name, score in [('PCC', r[iu]), ('MR', -mr[iu])]:
    top = np.argsort(-score)[:K]
    print(f'按 {name} 取最强的 {K} 条边，其中模块内的比例 {same[top].mean():.0%}')

# 3) WGCNA 软阈值：a_ij = |r_ij|^β；看度分布的无标度拟合 R² 和平均连接度
r0 = r.copy(); np.fill_diagonal(r0, 0)
for b in (1, 4, 8, 12):
    k = (np.abs(r0) ** b).sum(1)
    hist, e = np.histogram(k, bins=12); mid = (e[:-1] + e[1:]) / 2; ok = hist > 0
    x, y = np.log10(mid[ok]), np.log10(hist[ok] / hist.sum())
    print(f'β={b:2d}: 平均连接度 {k.mean():8.2f}，无标度拟合 R²={np.corrcoef(x, y)[0, 1] ** 2:.2f}，'
          f'斜率 {np.polyfit(x, y, 1)[0]:.2f}')
