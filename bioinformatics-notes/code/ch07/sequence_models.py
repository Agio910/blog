"""第7章：Z曲线可逆性、字码表、质量加权的二倍体分型。仅标准库。"""
from collections import defaultdict
import math

step = {'A': (1, 1, 1), 'C': (-1, 1, -1),
        'G': (1, -1, -1), 'T': (-1, -1, 1)}
seq = 'TGACC'
xyz = [(0, 0, 0)]
for b in seq:
    xyz.append(tuple(a + d for a, d in zip(xyz[-1], step[b])))
inv = {v: k for k, v in step.items()}
decoded = ''.join(inv[tuple(b-a for a, b in zip(xyz[i], xyz[i+1]))]
                  for i in range(len(seq)))
assert decoded == seq
print('Z曲线:', xyz)
print('恢复序列:', decoded)

toy = 'TGGAAATAAAACGTAAGTAG'
for k in (2, 3):
    pos = defaultdict(list)
    for i in range(len(toy)-k+1):
        w = toy[i:i+k]
        v = 1 + sum('ACGT'.index(b)*4**(k-j-1) for j,b in enumerate(w))
        pos[(v,w)].append(i+1)
    print(f'{k}-mer字码/序列/位置:', [(v,w,p) for (v,w),p in sorted(pos.items())])

def posterior(reads, q, prior=(0.98, 0.019, 0.001)):
    """reads 是该位点观测碱基；三基因型 AA/AG/GG，Q为每条观测的Phred值。
    假设读段独立且正确定位，各类替换错误等概率。仅作模型演示。
    """
    logs = []
    for genotype, pr in zip(('AA', 'AG', 'GG'), prior):
        logp = math.log(pr)
        for b, quality in zip(reads, q):
            e = 10**(-quality/10)
            prob = sum(1-e if b == a else e/3 for a in genotype)/2
            logp += math.log(prob)
        logs.append(logp)
    m = max(logs)
    weights = [math.exp(v-m) for v in logs]
    return [w/sum(weights) for w in weights]

for reads, qs in [('AAAAAG', [30]*6), ('AAAAAG', [30]*5+[5]),
                  ('AAAGGG', [30]*6)]:
    p = posterior(reads, qs)
    print(f'{reads} Q={qs}: AA/AG/GG=' + '/'.join(f'{x:.6f}' for x in p))
    assert abs(sum(p)-1) < 1e-12
print('模型归一化与Z曲线逆变换检查：通过')
