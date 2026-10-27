import numpy as np
from scipy.stats import rankdata

rng = np.random.default_rng(0)
NG, GLEN, READ = 40, 200_000, 150

# 1) 用 2 阶马尔可夫链生成 40 个“基因组”：GC 35%–65%，三核苷酸偏好只在背景附近小幅波动
def make_genome():
    gc = rng.uniform(0.35, 0.65)
    bg = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    cum = np.cumsum(rng.dirichlet(300 * bg, size=16), axis=1)       # 16 个二核苷酸上下文 × 下一个碱基
    u = rng.random(GLEN)
    s = np.empty(GLEN, np.int8); s[:2] = rng.integers(0, 4, 2)
    for i in range(2, GLEN):
        s[i] = np.searchsorted(cum[s[i - 2] * 4 + s[i - 1]], u[i])
    return s
genomes = [make_genome() for _ in range(NG)]

# 2) 规范四核苷酸频率（136 维；正反链合并），CLR 变换
idx = np.arange(256)
rc = sum((3 - (idx >> (2 * k)) & 3) << (2 * (3 - k)) for k in range(4))
canon = np.unique(np.minimum(idx, rc), return_inverse=True)[1]       # 256 → 136
def tnf(seq):
    k = seq[:-3].astype(int) * 64 + seq[1:-2] * 16 + seq[2:-1] * 4 + seq[3:]
    lf = np.log((np.bincount(canon[k], minlength=136) + 0.5))
    return lf - lf.mean()

def contigs(L):                                                      # 每个基因组切成不重叠的长 L 片段
    X, y = [], []
    for g, s in enumerate(genomes):
        for st in range(0, GLEN - L + 1, L):
            X.append(tnf(s[st:st + L])); y.append(g)
    return np.array(X), np.array(y)

def coverage(y, L, S):                                               # S 个样本里的覆盖度（负二项噪声）
    ab = rng.lognormal(0, 1.0, (NG, S)) * 10                         # 每个基因组在每个样本里的平均深度
    mu = ab[y] * L / READ
    reads = rng.poisson(rng.gamma(8, mu / 8))                        # gamma 形状 8：模拟 GC 偏好等额外波动
    return np.log(reads * READ / L + 0.1)

def auc(d, same):                                                    # P(同基因组距离 < 不同基因组距离)
    r = rankdata(d)
    n1, n0 = same.sum(), (~same).sum()
    return 1 - (r[same].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def knn_purity(F, y, k=5):                                           # 每个 contig 的 k 个最近邻里同源的比例
    sq = (F ** 2).sum(1)
    hit = 0
    for st in range(0, len(F), 1000):
        d = sq[st:st + 1000, None] - 2 * F[st:st + 1000] @ F.T + sq[None]
        d[np.arange(len(d)), np.arange(st, st + len(d))] = np.inf
        nn = np.argpartition(d, k, axis=1)[:, :k]
        hit += (y[nn] == y[st:st + 1000, None]).sum()
    return hit / (len(F) * k)

print(f'{"contig 长":>8} | AUC：{"TNF":>5} {"覆盖3":>6} {"覆盖20":>6} {"合并3":>6} {"合并20":>6} | 5-NN 纯度：TNF  合并3  合并20 | TNF 组内/组间 SD')
for L in (1000, 2500, 10000):
    X, y = contigs(L)
    pairs = rng.integers(0, len(y), (300_000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    same = y[pairs[:, 0]] == y[pairs[:, 1]]
    dist = lambda F: np.linalg.norm(F[pairs[:, 0]] - F[pairs[:, 1]], axis=1)
    Xz = X / np.median(dist(X))                                      # 把两类特征缩放到可比的尺度
    covs = {S: coverage(y, L, S) for S in (3, 20)}
    Cz = {S: c / np.median(dist(c)) for S, c in covs.items()}
    aucs = [auc(dist(Xz), same)] + [auc(dist(Cz[S]), same) for S in (3, 20)]
    aucs += [auc(dist(Xz) + dist(Cz[S]), same) for S in (3, 20)]
    pur = [knn_purity(Xz, y)] + [knn_purity(np.hstack([Xz, Cz[S]]), y) for S in (3, 20)]
    within = np.mean([X[y == g].std(0).mean() for g in range(NG)])
    between = np.array([X[y == g].mean(0) for g in range(NG)]).std(0).mean()
    print(f'{L:>7}bp |      ' + ' '.join(f'{v:6.3f}' for v in aucs) + ' |           ' + '  '.join(f'{v:5.2f}' for v in pur)
          + f' | {within / between:5.2f}')
