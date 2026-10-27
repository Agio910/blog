import numpy as np
from scipy.stats import fisher_exact

rng = np.random.default_rng(0)
# 一条 5 kb 转录本切成 200 个 25bp 窗口（MeRIP-PF 的做法）
W = 200
expr = rng.gamma(20, 1, W)                       # 转录本表达沿位置的起伏（input 和 IP 共享）
enrich = np.ones(W); enrich[150:162] = 8.0       # 终止密码子附近一个 m6A 峰（真实富集 8 倍）
inp = rng.poisson(expr)                          # input（常规 RNA-Seq）读序
ip = rng.poisson(expr * enrich * 0.5)            # IP 读序（文库更小，所以整体乘 0.5）
N_inp, N_ip = inp.sum(), ip.sum()

# 每个窗口一个 2×2 表：窗口内/外 × IP/input，单侧 Fisher 检验 IP 是否富集
p = np.array([fisher_exact([[ip[i], N_ip - ip[i]], [inp[i], N_inp - inp[i]]], alternative='greater')[1]
              for i in range(W)])
order = np.argsort(p); adj = np.empty(W)
adj[order] = np.minimum(1, np.minimum.accumulate((p[order] * W / np.arange(1, W + 1))[::-1])[::-1])
sig = adj < 0.05
peaks, i = [], 0
while i < W:                                     # 相邻显著窗口合并成峰
    if sig[i]:
        j = i
        while j + 1 < W and sig[j + 1]: j += 1
        peaks.append((i * 25, (j + 1) * 25)); i = j + 1
    else:
        i += 1
print('显著窗口数:', int(sig.sum()), '| 合并后的峰 (bp):', peaks, '| 真实峰: (3750, 4050)')
