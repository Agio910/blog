import numpy as np
from math import comb

rng = np.random.default_rng(0)

# 真值：60 个病毒基因组（20–150 kb），每个被组装成 1 + Poisson(3) 个 contig
genome_of, length = [], []
for g in range(60):
    G = int(np.clip(rng.lognormal(np.log(50_000), 0.6), 20_000, 150_000))
    k = 1 + rng.poisson(3)
    cuts = np.sort(rng.choice(np.arange(1, G), k - 1, replace=False)) if k > 1 else np.array([], int)
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, G]):
        genome_of.append(g); length.append(b - a)
genome_of, length = np.array(genome_of), np.array(length)
glen = np.bincount(genome_of, weights=length)
n = len(length)
print(f'{n} 个 contig，其中 {np.sum(np.bincount(genome_of) == 1)} 个基因组只有 1 个 contig')

def conservative():                                        # 每个 contig 自成一个 bin
    return np.arange(n)

def aggressive(p_false=0.15):                              # 同源 contig 全部合并，再有 15% 的 bin 两两误合并
    b = genome_of.copy()
    ids = rng.permutation(60)
    for a, c in zip(ids[: int(60 * p_false)], ids[int(60 * p_false): 2 * int(60 * p_false)]):
        b[b == c] = a
    return b

def realistic(recall=0.8, p_absorb=0.08):                  # 每个 contig 以 80% 概率进入本基因组的 bin，偶尔被别的 bin 吸走
    b = np.where(rng.random(n) < recall, genome_of, 1000 + np.arange(n))
    wrong = rng.random(n) < p_absorb
    b[wrong] = rng.integers(0, 60, wrong.sum())
    return b

def evaluate(bins):
    hq = false_multi = multi = 0
    purities, weights = [], []
    best_comp = np.zeros(60)
    for b in np.unique(bins):
        m = bins == b
        bp = np.bincount(genome_of[m], weights=length[m], minlength=60)
        top = bp.argmax()
        purity, comp = bp[top] / bp.sum(), bp[top] / glen[top]
        purities.append(purity); weights.append(bp.sum())
        best_comp[top] = max(best_comp[top], comp)
        hq += comp >= 0.9 and purity >= 0.95
        if m.sum() > 1:
            multi += 1
            false_multi += (bp > 0).sum() > 1
    ct = np.zeros((60, len(np.unique(bins))), int)
    np.add.at(ct, (genome_of, np.unique(bins, return_inverse=True)[1]), 1)
    sc = sum(comb(int(v), 2) for v in ct.ravel())
    sa, sb = sum(comb(int(v), 2) for v in ct.sum(1)), sum(comb(int(v), 2) for v in ct.sum(0))
    e = sa * sb / comb(n, 2)
    return dict(达阈值bin数=hq, 多contig_bin数=multi, 误合并率=false_multi / multi if multi else float('nan'),
                bp加权精确率=np.average(purities, weights=weights), 平均召回=best_comp.mean(),
                ARI=(sc - e) / ((sa + sb) / 2 - e))

for name, f in [('保守型（全单例）', conservative), ('激进型（合并 + 15% 误合并）', aggressive), ('折中型', realistic),
                ('完美分箱', lambda: genome_of.copy())]:
    r = evaluate(f())
    print(f'{name:<16} ' + '  '.join(f'{k} {v:.2f}' if isinstance(v, float) else f'{k} {v}' for k, v in r.items()))
