import numpy as np
from collections import Counter
from scipy.stats import poisson

rng = np.random.default_rng(0)
L, err = 250, 0.005                                       # 扩增子长度；每碱基替换错误率（真实流程中由 DADA2 从数据学出）

def mutate(seq, k):                                       # 在 k 个随机位置做替换
    s = seq.copy()
    pos = rng.choice(L, k, replace=False)
    s[pos] = (s[pos] + rng.integers(1, 4, k)) % 4
    return s

# 真实变体：V2、V3 与 V1 只差 2 个、1 个碱基；V5 与 V4 差 3 个；V1 与 V4 差 20 个
V1 = rng.integers(0, 4, L)
V = {'V1': V1, 'V2': mutate(V1, 2), 'V3': mutate(V1, 1)}
V['V4'] = mutate(V1, 20); V['V5'] = mutate(V['V4'], 3)
abund = {'V1': 5000, 'V2': 1000, 'V3': 300, 'V4': 2000, 'V5': 50}

reads = []
for name, n in abund.items():
    for _ in range(n):
        r = V[name].copy()
        hit = rng.random(L) < err                         # 测序错误
        r[hit] = (r[hit] + rng.integers(1, 4, hit.sum())) % 4
        reads.append(r)
for _ in range(250):                                      # PCR 嵌合体：V1 的前半段 + V4 的后半段（断点 100–150）
    bp = rng.integers(100, 151)
    reads.append(np.concatenate([V['V1'][:bp], V['V4'][bp:]]))

uniq = Counter(r.tobytes() for r in reads)                # 去冗余（dereplicate），保留计数
seqs = np.array([np.frombuffer(k, dtype=V1.dtype) for k in uniq])
counts = np.array(list(uniq.values()))
order = np.argsort(-counts); seqs, counts = seqs[order], counts[order]
print(f'读序 {counts.sum()} 条，去冗余后有 {len(seqs)} 种不同序列（100% 聚类就会得到这么多“OTU”）')

ham = lambda a, B: (B != a).sum(-1)                       # 汉明距离（只模拟替换错误，所以序列等长）

# 1) 97% OTU：按丰度贪心聚类，距离中心 ≤ 7 个错配（3% × 250）就并入
centroids = []
for s in seqs:
    if not centroids or ham(s, np.array(centroids)).min() > int(0.03 * L):
        centroids.append(s)
truth = {k: v for k, v in V.items()}
def label(s):
    d = {k: int((s != v).sum()) for k, v in truth.items()}
    k = min(d, key=d.get)
    return k if d[k] == 0 else f'{k}+{d[k]}错配'
print('97% OTU 个数:', len(centroids), '→ 中心序列:', [label(c) for c in centroids])

# 2) 简化版 DADA2：丰度 p 值。序列 s 若完全由某个 ASV 的测序错误产生，期望条数
#    λ = N_partition × (err/3)^d × (1−err)^(L−d)；若观测条数远超 λ（p < 1e-40）就认定为新的真实序列
log_e, log_ok = np.log(err / 3), np.log(1 - err)
asv = [0]
while True:
    D = np.stack([ham(seqs[a], seqs) for a in asv])                  # 每个 ASV 到所有序列的距离
    loglam_unit = D * log_e + (L - D) * log_ok                       # 单条读序变成该序列的对数概率
    part = loglam_unit.argmax(0)                                     # 分配给最可能产生它的 ASV
    N_part = np.array([counts[part == i].sum() for i in range(len(asv))])
    lam = N_part[part] * np.exp(loglam_unit[part, np.arange(len(seqs))])
    p = poisson.sf(counts - 1, lam) / poisson.sf(0, lam)            # P(X ≥ n | X ≥ 1)
    p[asv] = 1.0
    best = int(np.argmin(p))
    if p[best] >= 1e-40:
        break
    asv.append(best)
print('去噪得到的 ASV 个数:', len(asv), '→', [label(seqs[a]) + f'({counts[part == i].sum()} 条)' for i, a in enumerate(asv)])

# 3) 去嵌合体（bimera）：若一个 ASV 能写成两个更丰富 ASV 的“左段 + 右段”，判为嵌合体
def is_bimera(i):
    s = seqs[asv[i]]
    parents = [seqs[a] for j, a in enumerate(asv) if counts[part == j].sum() > 2 * counts[part == i].sum()]
    for a in parents:
        for b in parents:
            left = np.cumsum(s == a) == np.arange(1, L + 1)            # 前缀与 a 完全一致
            right = (np.cumsum((s == b)[::-1]) == np.arange(1, L + 1))[::-1]
            if (left[:-1] & right[1:]).any():
                return True
    return False
kept = [label(seqs[a]) for i, a in enumerate(asv) if not is_bimera(i)]
print('去掉嵌合体后的 ASV:', kept)
