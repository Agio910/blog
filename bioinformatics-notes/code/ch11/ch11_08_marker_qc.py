import numpy as np

rng = np.random.default_rng(0)
M = 120                                                    # 单拷贝标记基因个数（类似 bac120）

def genome_contigs(n_contigs=80):                          # 每个基因组被组装成 80 个 contig，标记基因随机分布其上
    return rng.integers(0, n_contigs, M)                   # marker_on[j] = 第 j 个标记所在的 contig

def quality(parts):                                        # parts: [(marker_on, 选入 bin 的 contig 集合), ...]
    copies = np.zeros(M, int)
    for marker_on, chosen in parts:
        copies += np.isin(marker_on, list(chosen))
    completeness = (copies > 0).mean()
    contamination = np.clip(copies - 1, 0, None).sum() / M  # 多出来的拷贝数 / 标记总数（CheckM 的定义）
    return completeness, contamination

def tier(c, k):                                            # MIMAG：高质量还要求 rRNA 与 ≥18 种 tRNA，这里只看两个数
    if k >= 0.1:
        return '污染超标'
    if c > 0.9 and k < 0.05:
        return '近完整阈值（RNA未查）'
    return '中等质量' if c >= 0.5 else '低质量'

A, B = genome_contigs(), genome_contigs()
allc = set(range(80))
cases = {
    '纯 A，60% 的 contig': [(A, set(rng.choice(80, 48, replace=False)))],
    '纯 A，95% 的 contig': [(A, set(rng.choice(80, 76, replace=False)))],
    'A 的 60% + B 的 30%（随机混入）': [(A, set(rng.choice(80, 48, replace=False))), (B, set(rng.choice(80, 24, replace=False)))],
}
for name, parts in cases.items():
    c, k = quality(parts)
    print(f'{name:<30} 完整度 {c:6.1%}  污染度 {k:5.1%}  → {tier(c, k)}')

# “互补型”嵌合：基因组碎成 300 个 contig；bin 含 A 的 75%，再混入 B 中恰好携带 A 所缺标记、且不带重复拷贝的 contig
A3, B3 = genome_contigs(300), genome_contigs(300)
a_part = set(rng.choice(300, 225, replace=False))
present = np.isin(A3, list(a_part))
b_part = {c for c in set(B3[~present]) if not np.any((B3 == c) & present)}
c, k = quality([(A3, a_part), (B3, b_part)])
share = len(b_part) / (len(a_part) + len(b_part))
print(f'{"A 的 75% + B 中恰好补缺的 contig":<30} 完整度 {c:6.1%}  污染度 {k:5.1%}  → {tier(c, k)}'
      f'（实际上 bin 里 {share:.0%} 的 contig 来自 B）')

# 标记基因估计的抽样误差：一个 bin 只含基因组的 30% 时，完整度估计的波动
est = [quality([(genome_contigs(), set(rng.choice(80, 24, replace=False)))])[0] for _ in range(2000)]
print(f'真实完整度 30% 的 bin：标记基因估计 = {np.mean(est):.1%} ± {np.std(est):.1%}（2000 次模拟）')
