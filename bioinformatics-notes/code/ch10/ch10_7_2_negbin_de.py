import sys
import numpy as np
from scipy import stats

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)

# 1) Gamma-Poisson 混合 = 负二项：方差 = μ + αμ²
mu, alpha = 100, 0.2
lam = rng.gamma(shape=1 / alpha, scale=mu * alpha, size=200_000)   # 生物学变异
y = rng.poisson(lam)                                                # 测序抽样
print(f'均值 {y.mean():.1f}  方差 {y.var():.0f}  理论 μ+αμ² = {mu + alpha * mu**2:.0f}  泊松会给出 {mu}')

# 2) 模拟 3 vs 3 的差异表达实验：2000 个基因，其中 200 个真差异（倍数 2 或 1/2）
G = 2000
base = rng.lognormal(mean=4, sigma=1.5, size=G)                     # 基因的基线表达
disp = 0.02 + 1.0 / base                                            # 真实离散度随均值下降
de = np.zeros(G, bool); de[:200] = True
fold = np.where(de, rng.choice([0.5, 2.0], G), 1.0)
sf = np.array([0.8, 1.0, 1.2, 0.9, 1.1, 1.0])                       # 6 个样本的大小因子
group = np.array([0, 0, 0, 1, 1, 1])
mean = base[:, None] * np.where(group == 1, fold[:, None], 1.0) * sf
cnt = rng.poisson(rng.gamma(1 / disp[:, None], mean * disp[:, None]))

norm = cnt / sf
ma, mb = norm[:, group == 0].mean(1), norm[:, group == 1].mean(1)
keep = (ma + mb) > 1                                                # 去掉几乎不表达的基因

def bh(p):                                                          # Benjamini–Hochberg 校正后的 p 值
    m = len(p); order = np.argsort(p)
    adj = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[order] = np.minimum(adj, 1); return out

def wald_nb(a):                                                     # 两组设计的 NB Wald 检验（教学简化版）
    def info(m, cols):                                              # log 均值的 Fisher 信息 Σ μ/(1+αμ)
        mu_j = m[:, None] * sf[cols]
        return (mu_j / (1 + a[:, None] * mu_j)).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        lfc = np.log(mb + 1e-8) - np.log(ma + 1e-8)
        se = np.sqrt(1 / info(ma, group == 0) + 1 / info(mb, group == 1))
        return 2 * stats.norm.sf(np.abs(lfc / se))

# 逐基因矩估计 α = (var − mean)/mean²（组内合并方差）
v = (norm[:, group == 0].var(1, ddof=1) + norm[:, group == 1].var(1, ddof=1)) / 2
m_all = (ma + mb) / 2
a_gene = np.clip((v - m_all) / np.maximum(m_all, 1e-8) ** 2, 1e-4, 10)
# 跨基因借力：拟合“离散度-均值”趋势 α(μ) = a0 + a1/μ（只用没被截断的估计）
X = np.c_[np.ones(G), 1 / np.maximum(m_all, 1e-8)]
ok = keep & (a_gene > 1e-4)
a_trend = np.clip(X @ np.linalg.lstsq(X[ok], a_gene[ok], rcond=None)[0], 1e-4, None)

def shrink(w):                                                      # w = 保留逐基因估计的权重（对数尺度）
    return np.exp(w * np.log(a_gene) + (1 - w) * np.log(a_trend))

logcpm = np.log2(cnt / cnt.sum(0) * 1e6 + 1)
p_t = stats.ttest_ind(logcpm[:, group == 1], logcpm[:, group == 0], axis=1, equal_var=False).pvalue
tests = {'泊松 Wald (α=0)': wald_nb(np.zeros(G)),
         'NB, 逐基因 α': wald_nb(a_gene),
         'NB, 部分收缩 (w=0.2)': wald_nb(shrink(0.2)),
         'NB, 全收缩到趋势': wald_nb(a_trend),
         'NB, 真实 α (参照)': wald_nb(disp),
         'Welch t, log2 CPM': p_t}
for name, p in tests.items():
    p = np.where(np.isfinite(p), p, 1.0)[keep]
    hit = bh(p) < 0.05
    truth = de[keep]
    fdr = (hit & ~truth).sum() / max(hit.sum(), 1)
    print(f'{name:<20} 判为差异 {hit.sum():4d}  单次 FDP {fdr:5.1%}  检出率 {(hit & truth).sum() / truth.sum():5.1%}')
