import numpy as np
from math import comb
from scipy.stats import hypergeom, fisher_exact

# 1) 教材的 BH 步骤：找最大的 i 使 P(i) ≤ i·q/m，取前 i 个
p = np.array([0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.3240])
q, m = 0.05, len(p)
ps = np.sort(p)
i_max = max(i for i in range(1, m + 1) if ps[i - 1] <= i * q / m)
adj = np.minimum.accumulate((ps * m / np.arange(1, m + 1))[::-1])[::-1]   # BH 校正后的 p 值
print('教材步骤选出', i_max, '个；按校正后 p ≤ 0.05 选出', int((adj <= q).sum()), '个')
print('校正后 p:', np.round(adj, 4).tolist())
print('Bonferroni 选出', int((p <= q / m).sum()), '个')

# 2) 式 10.4：K 个差异基因里至少 x 个落在基因集 S（大小 M）中的概率
def p_formula(N, M, K, x):
    return 1 - sum(comb(M, i) * comb(N - M, K - i) for i in range(x)) / comb(N, K)

N, M, K, x = 20000, 200, 400, 12          # 全基因组 2 万个基因；S 有 200 个；差异基因 400 个，其中 12 个在 S 里
print(f'期望重叠 {K * M / N:.1f}；式10.4 p = {p_formula(N, M, K, x):.2e}；'
      f'超几何上尾 p = {hypergeom.sf(x - 1, N, M, K):.2e}；'
      f'单侧 Fisher p = {fisher_exact([[x, M - x], [K - x, N - M - K + x]], alternative="greater")[1]:.2e}')
# 背景换成“本实验检测到表达的 12 000 个基因”（S 中只有 180 个表达）
print(f'换背景后：期望重叠 {K * 180 / 12000:.1f}，p = {hypergeom.sf(x - 1, 12000, 180, K):.2e}')
