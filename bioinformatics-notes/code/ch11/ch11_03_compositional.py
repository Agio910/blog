import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# 1) 闭合导致的伪相关：各物种的绝对丰度相互独立，换算成比例后却出现负相关
D, n = 5, 2000
A = rng.lognormal(0, 0.5, (n, D)) * np.array([10, 5, 2, 1, 1])     # 5 个物种的绝对丰度，彼此独立
P = A / A.sum(1, keepdims=True)
iu = np.triu_indices(D, 1)
print('绝对丰度两两相关（应约为 0）:', np.round(np.corrcoef(A.T)[iu], 2).tolist())
print('相对丰度两两相关            :', np.round(np.corrcoef(P.T)[iu], 2).tolist())

# 2) 差异丰度：B 组里 taxon 0–9 的绝对丰度升高 8 倍，taxon 10–49 不变
D, n_per, changed = 50, 15, np.arange(10)
base = rng.lognormal(0, 1.5, D)
def group(bloom):
    load = rng.lognormal(0, 0.3, n_per)                              # 样本间总菌量也有波动
    absl = base * rng.lognormal(0, 0.5, (n_per, D)) * load[:, None]
    if bloom:
        absl[:, changed] *= 8
    reads = np.array([rng.multinomial(20_000, a / a.sum()) for a in absl])
    return absl, reads
absA, rA = group(False)
absB, rB = group(True)

def bh(p):
    m = len(p); o = np.argsort(p)
    a = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(a, 1); return out

def report(name, xa, xb, taxa):
    p = np.array([stats.mannwhitneyu(xa[:, j], xb[:, j]).pvalue for j in range(xa.shape[1])])
    sig = bh(p) < 0.05
    true = np.isin(taxa, changed)
    print(f'{name:<24} 真变化的 10 个检出 {(sig & true).sum():2d} 个；未变化的 {(~true).sum()} 个里误判 {(sig & ~true).sum():2d} 个')

prop = lambda r: r / r.sum(1, keepdims=True)
clr = lambda r: (lambda L: L - L.mean(1, keepdims=True))(np.log(r + 0.5))
taxa = np.arange(D)
report('绝对丰度（已知绝对丰度参照）', absA, absB, taxa)
report('相对丰度（比例）', prop(rA), prop(rB), taxa)
report('CLR 变换', clr(rA), clr(rB), taxa)
ref = int(next(j for j in np.argsort(base)[::-1] if j not in changed))   # 最常见的未变化物种
keep = taxa != ref                                                   # ALR：以一个已知不变、较常见的物种为参照
alr = lambda r: (np.log(r + 0.5) - np.log(r[:, [ref]] + 0.5))[:, keep]
report(f'ALR（参照 taxon {ref}）', alr(rA), alr(rB), taxa[keep])
fold = np.median(prop(rB), 0) / np.median(prop(rA), 0)
print('比例数据里，未变化物种的中位数“变化倍数” B/A = %.2f；CLR 里的平均偏移 = %.2f（自然对数）'
      % (np.median(fold[10:]), np.median(clr(rB)[:, 10:].mean(0) - clr(rA)[:, 10:].mean(0))))
