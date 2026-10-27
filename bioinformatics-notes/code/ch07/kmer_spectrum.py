# -*- coding: utf-8 -*-
"""第 7 章 §8.1：模拟二倍体小基因组的短读序，统计 canonical k-mer，
打印 k-mer 频数谱，并按式（7.1）估计基因组大小。只依赖 numpy。"""
import numpy as np

rng = np.random.default_rng(7)
K = 21
G = 100_000                  # 单倍体基因组长度（bp）
HET = 0.01                   # 两个单倍型之间的 SNP 比例
REP_LEN, REP_COPIES = 1_000, 5
READ_LEN, COV_PER_HAP, ERR = 100, 25, 0.005   # 每个单倍型 25x，合计 50x

# 1. 单倍型 A：随机序列，再写入 5 份完全相同的 1 kb 重复元件；单倍型 B 在 1% 的位点上换碱基
hapA = rng.integers(0, 4, G).astype(np.int8)
rep = rng.integers(0, 4, REP_LEN).astype(np.int8)
for s in np.arange(REP_COPIES) * (G // REP_COPIES) + 5_000:
    hapA[s:s + REP_LEN] = rep
hapB = hapA.copy()
snp = rng.choice(G, int(HET * G), replace=False)
hapB[snp] = (hapA[snp] + rng.integers(1, 4, snp.size)) % 4

# 2. 读序：起点均匀分布，一半来自反向互补链，碱基以 ERR 的概率被替换
def sample_reads(hap, n):
    starts = rng.integers(0, G - READ_LEN + 1, n)
    return hap[starts[:, None] + np.arange(READ_LEN)]

n_per_hap = COV_PER_HAP * G // READ_LEN
R = np.vstack([sample_reads(hapA, n_per_hap), sample_reads(hapB, n_per_hap)])
flip = rng.random(len(R)) < 0.5
R[flip] = 3 - R[flip][:, ::-1]                     # A=0,C=1,G=2,T=3，互补即 3-x
err = rng.random(R.shape) < ERR
R[err] = (R[err] + rng.integers(1, 4, err.sum())) % 4

# 3. 2-bit 编码 k-mer；canonical k-mer = min(正向编码, 反向互补编码)
W = READ_LEN - K + 1
fwd = np.zeros((len(R), W), dtype=np.int64)
rc = np.zeros((len(R), W), dtype=np.int64)
for j in range(K):
    col = R[:, j:j + W].astype(np.int64)
    fwd = (fwd << 2) | col
    rc |= (3 - col) << (2 * j)
canon = np.minimum(fwd, rc).ravel()

def spectrum(codes):
    _, cnt = np.unique(codes, return_counts=True)
    return np.bincount(cnt)            # hist[d] = 出现 d 次的不同 k-mer 个数

def estimate(hist, label):
    d = np.arange(len(hist))
    valley = next(i for i in range(2, len(hist) - 1) if hist[i] <= hist[i + 1])
    peak = valley + int(np.argmax(hist[valley:valley + 60]))   # 只在低深度区找峰
    hom = peak                                                   # 默认最高峰是纯合峰
    knum = int((d[valley:] * hist[valley:]).sum())               # 去掉错误 k-mer 后的总数
    print(f"[{label}] 谷底深度={valley}  最高峰深度={peak}  K_num={knum:,}  "
          f"K_num/峰={knum / hom:,.0f} bp")
    return valley, peak, knum

def show(hist, edges):
    ns = [int(hist[lo:hi + 1].sum()) for lo, hi in edges]
    top = max(n for (lo, _), n in zip(edges, ns) if lo >= 3)
    for (lo, hi), n in zip(edges, ns):
        bar = '#' * int(np.ceil(40 * n / top)) if lo >= 3 else '(测序错误)'
        print(f"  {lo:>4}-{hi:<4} {n:>8,}  {bar}")

hist = spectrum(canon)
print(f"读序 {len(R):,} 条 × {READ_LEN} bp，碱基深度 {2 * COV_PER_HAP}x，K={K}")
print("canonical k-mer 频数谱（深度区间, 不同 k-mer 个数）：")
show(hist, [(1, 1), (2, 2), (3, 6), (7, 11), (12, 16), (17, 21), (22, 26), (27, 31),
            (32, 36), (37, 41), (42, 46), (47, 51), (52, 100), (101, 150), (151, 220), (221, 400)])
valley, peak, knum = estimate(hist, "canonical")

c_base = 2 * COV_PER_HAP
print(f"理论纯合峰 = C*(L-K+1)/L*(1-e)^K = {c_base}*{W}/{READ_LEN}*{(1 - ERR) ** K:.3f}"
      f" = {c_base * W / READ_LEN * (1 - ERR) ** K:.1f}")
het_lo, het_hi = valley, int(0.75 * peak)
n_het = int(hist[het_lo:het_hi + 1].sum())
n_hom = int(hist[het_hi + 1:2 * peak].sum())
p = (n_het / 2) / (n_het / 2 + n_hom)
print(f"杂合峰区间 [{het_lo},{het_hi}] 内 {n_het:,} 个 k-mer，纯合区 {n_hom:,} 个；"
      f"杂合 k-mer 比例 p={p:.3f}，h = 1-(1-p)^(1/K) = {1 - (1 - p) ** (1 / K):.4f}（真值 {HET}）")
print(f"若误把杂合峰（约 {peak // 2}）当作 K_depth：{knum / (peak / 2):,.0f} bp，约为真值两倍")
lo, hi = int(0.75 * peak), int(1.5 * peak)
mean_c = (np.arange(lo, hi + 1) * hist[lo:hi + 1]).sum() / hist[lo:hi + 1].sum()
print(f"纯合峰 [{lo},{hi}] 内的平均深度 {mean_c:.2f}；K_num/平均深度 = {knum / mean_c:,.0f} bp（真值 {G:,}）")
d = np.arange(len(hist))
rep_frac = (d[2 * peak + 1:] * hist[2 * peak + 1:]).sum() / knum
print(f"深度 >2c 的 k-mer 占 K_num 的 {rep_frac:.1%}（重复元件真值 {REP_LEN * REP_COPIES / G:.1%}）")

print("\n不合并正反链（只数正向编码）：")
estimate(spectrum(fwd.ravel()), "forward-only")
