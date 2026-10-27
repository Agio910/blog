import numpy as np
from scipy.special import gammaln

rng = np.random.default_rng(0)

def simpson(n):                       # 式 11.2：Gini-Simpson，1 − Σp²
    p = n / n.sum()
    return 1 - (p ** 2).sum()

def shannon(n, base=2.0):             # 式 11.3（教材用 log2；QIIME/vegan 默认自然对数）
    p = n[n > 0] / n.sum()
    return max(0.0, -(p * np.log(p)).sum() / np.log(base))

def chao1(n):                         # 式 11.4：偏差校正版
    f1, f2 = (n == 1).sum(), (n == 2).sum()
    return (n > 0).sum() + f1 * (f1 - 1) / (2 * (f2 + 1))

def hill(p, q):                       # Hill 数：有效物种数，q=0/1/2 分别对应丰富度、exp(Shannon)、1/Σp²
    p = p[p > 0] / p.sum()
    if q == 0:
        return len(p)
    if q == 1:
        return np.exp(-(p * np.log(p)).sum())
    return ((p ** q).sum()) ** (1 / (1 - q))

# 1) 教材例题
print('Simpson  甲(99,1) = %.4f   乙(50,50) = %.4f' % (simpson(np.array([99, 1])), simpson(np.array([50, 50]))))
print('Shannon  A(100,0) = %.3f   B(50,50) = %.3f   C(99,1) = %.3f'
      % tuple(shannon(np.array(x)) for x in ([100, 0], [50, 50], [99, 1])))

# 2) 一个“真实”群落：1000 个物种，丰度呈对数正态（少数优势、大量稀有）
S = 1000
p_true = rng.lognormal(0, 2, S); p_true /= p_true.sum()
print('真值: 丰富度 %d   Hill q1 = %.1f   Hill q2 = %.1f   Simpson = %.4f'
      % (S, hill(p_true, 1), hill(p_true, 2), 1 - (p_true ** 2).sum()))
for depth in (1_000, 5_000, 20_000, 100_000):
    n = rng.multinomial(depth, p_true)
    print(f'测序 {depth:>7,} 条: S_obs {int((n > 0).sum()):4d}  Chao1 {chao1(n):6.0f}  '
          f'Hill q1 {hill(n.astype(float), 1):6.1f}  Hill q2 {hill(n.astype(float), 2):5.1f}  Simpson {simpson(n):.4f}')

# 3) 式 11.5 的稀释期望（超几何）与随机抽样平均对比
n = rng.multinomial(20_000, p_true)
N = n.sum()
def rarefy_expect(n, m):              # E(S_m) = Σ [1 − C(N−N_i, m)/C(N, m)]
    lc = lambda a, b: gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)
    keep = n[n > 0]
    absent = np.where(N - keep >= m, np.exp(lc(N - keep, m) - lc(N, m)), 0.0)
    return (1 - absent).sum()
reads = np.repeat(np.arange(S), n)
for m in (1_000, 5_000):
    sim = np.mean([len(np.unique(rng.choice(reads, m, replace=False))) for _ in range(200)])
    print(f'稀释到 {m:,} 条: 式11.5 期望 {rarefy_expect(n, m):.1f}   200 次随机抽样平均 {sim:.1f}')

# 4) 演示错误单例与人为删除单例对Chao1的影响
n = rng.multinomial(20_000, p_true)
noisy = np.concatenate([n, np.ones(300, int)])              # 300 条测序错误各自成为“新物种”
denoised = np.where(n == 1, 0, n)                            # 人为删除所有单例，仅演示过滤的影响，并非完整DADA2流程
print(f'原始: S_obs {int((n > 0).sum())}, Chao1 {chao1(n):.0f} | 混入 300 个错误单例: S_obs {int((noisy > 0).sum())}, '
      f'Chao1 {chao1(noisy):.0f} | 去掉单例: S_obs {int((denoised > 0).sum())}, Chao1 {chao1(denoised):.0f}')
