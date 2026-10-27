import numpy as np
from collections import Counter

rng = np.random.default_rng(0)
G, K, R = 30_000, 31, 150                                  # 基因组长度、k、读长
B = np.array(list('ACGT'))

def diverge(seq, rate):                                    # 独立地以 rate 的比例做替换
    s = seq.copy()
    hit = rng.random(len(s)) < rate
    s[hit] = (s[hit] + rng.integers(1, 4, hit.sum())) % 4
    return s

# 分类树：root(0) → 属 A(1)、属 B(2)；A1(3)、A2(4) ∈ A；B1(5)、B2(6) ∈ B。A3 是属 A 的新种，不在数据库里
parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 0: 0}
name = {0: 'root', 1: '属A', 2: '属B', 3: 'A1', 4: 'A2', 5: 'B1', 6: 'B2'}
ancA, ancB = rng.integers(0, 4, G), rng.integers(0, 4, G)
genome = {3: diverge(ancA, 0.04), 4: diverge(ancA, 0.04), 5: diverge(ancB, 0.04), 6: diverge(ancB, 0.04)}
A3 = diverge(ancA, 0.04)

def lineage(t):
    out = [t]
    while t != 0:
        t = parent[t]; out.append(t)
    return out

def lca(a, b):
    la = lineage(a)
    return next(t for t in lineage(b) if t in la)

db = {}                                                    # k-mer → 含有它的所有类群的 LCA
for t, g in genome.items():
    s = ''.join(B[g])
    for i in range(G - K + 1):
        k = s[i:i + K]
        db[k] = lca(db[k], t) if k in db else t
print('数据库 k-mer 数:', len(db), '| 各 LCA 层级的 k-mer 数:', {name[t]: c for t, c in sorted(Counter(db.values()).items())})

def classify(read, conf=0.0):                              # Kraken 1/2 的根到叶路径打分 + Kraken 2 的置信度
    hits = Counter(db.get(read[i:i + K]) for i in range(len(read) - K + 1))
    q = sum(hits.values())
    hits.pop(None, None)
    if not hits:
        return 'unclassified'
    score = {t: sum(c for u, c in hits.items() if u in lineage(t)) for t in hits}   # 各命中节点的根到叶得分
    best = max(score.values())
    tops = [t for t, s in score.items() if s == best]
    call = tops[0]
    for t in tops[1:]:
        call = lca(call, t)                                # 并列时取 LCA
    while True:                                            # 置信度：本类群子树内的 k-mer 占比不够就上移
        clade = sum(c for u, c in hits.items() if call in lineage(u))
        if clade / q >= conf:
            return name[call]
        if call == 0:
            return 'unclassified'
        call = parent[call]

def reads_from(g, n=400):
    starts = rng.integers(0, G - R + 1, n)
    return [''.join(B[diverge(g[s:s + R], 0.005)]) for s in starts]   # 读序含 0.5% 测序错误

for label, g in [('A1（库内物种）', genome[3]), ('A3（库外新种）', A3), ('B2（库内物种）', genome[6])]:
    rs = reads_from(g)
    for conf in (0.0, 0.1):
        res = Counter(classify(r, conf) for r in rs)
        print(f'{label} 置信度 {conf:.2f}：', dict(res.most_common()))
