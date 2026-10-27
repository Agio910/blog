import numpy as np

rng = np.random.default_rng(0)
S = 500                                                    # 群落里 500 个物种
cells = rng.lognormal(0, 2.0, S)                           # 细胞丰度：对数正态，跨好几个数量级
gsize = rng.uniform(2e6, 6e6, S)                           # 基因组大小 2–6 Mb
dna_frac = cells * gsize / (cells * gsize).sum()           # 读序比例 ∝ 细胞数 × 基因组大小（不是细胞比例！）
cell_frac = cells / cells.sum()
big = np.argmax(gsize * (cell_frac > np.median(cell_frac)))
print(f'例：物种 {big} 的细胞比例 {cell_frac[big]:.4%}，读序比例 {dna_frac[big]:.4%}（基因组大，读序被放大）')

R, overlap = 150, 31                                       # 读长；组装时需要的最小重叠（k-mer 长度）
theta = overlap / R
print(f'{"测序量":>8} {"≥1× 物种":>8} {"≥10× 物种":>9} {"≥20× 物种":>9}  中位覆盖  最稀有 10% 物种的覆盖范围')
for gbp in (5, 20, 100, 500):
    bases = gbp * 1e9
    cov = bases * dna_frac / gsize                         # 每个物种的平均覆盖深度 c_i
    rare = np.sort(cov)[: S // 10]
    print(f'{gbp:>6} Gb {np.sum(cov >= 1):>8} {np.sum(cov >= 10):>9} {np.sum(cov >= 20):>9}  '
          f'{np.median(cov):8.1f}  {rare.min():.3f}–{rare.max():.3f}')

# Lander–Waterman：覆盖深度 c 下，基因组被覆盖的比例 1−e^{−c}；contig 数 ≈ N e^{−c(1−θ)}
print('\n单个 4 Mb 基因组（理想模型：无重复、无菌株差异）：')
for c in (1, 3, 5, 10, 15, 20):
    N = c * 4e6 / R                                        # 读序条数
    print(f'  覆盖 {c:>2}×：未被覆盖比例 e^-c = {np.exp(-c):.1e}，期望 contig 数 N·e^(-c(1-θ)) = {N * np.exp(-c * (1 - theta)):8.1f}')
