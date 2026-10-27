"""教材式（9.1）–（9.8）的数值核对，以及在模拟位点上做相位扫描与多重检验校正。"""
import math
import numpy as np
from scipy.stats import hypergeom

# ---------- 公式核对 ----------
mfei = lambda mfe, L, gc: (abs(mfe) / L * 100) / gc          # 式 9.1，gc 用百分数数值
print(f"式9.1  MFEI(MFE=-45, L=110, GC=42%) = {mfei(-45, 110, 42):.3f}")

def howell(k):                                                # 式 9.2，k 为 8 个相位位置的读段数
    n = sum(x > 0 for x in k)
    return (n - 2) * math.log(1 + sum(k)) if n >= 3 else 0.0
print(f"式9.2  Howell P(n=5, sum k=50) = {howell([10] * 5 + [0] * 3):.2f}")

chen = lambda n, k, N=461, K=21: hypergeom.sf(k - 1, N, K, n)   # 式 9.3-9.4：P(X>=k)
print(f"式9.4  n=10,k=6: 461/21 -> P={chen(10, 6):.2e}（期望 k={10 * 21 / 461:.2f}）；"
      f"误用 462/22 -> {chen(10, 6, 462, 22):.2e}")
for s in (0, 1, 2):                                           # 式 9.5-9.6（pssRNAMiner）
    p_pss = hypergeom.sf(5, 461 - 42 * s, 21, 10)
    p_alt = hypergeom.sf(5, 461, 21 * (2 * s + 1), 10)        # 把 +-s 窗口内的位置都算作"相位"
    print(f"式9.6  s={s}: 单个位置落入相位的概率 {21 / (461 - 42 * s):.3f} vs {21 * (2 * s + 1) / 461:.3f}；"
          f"P={p_pss:.2e} vs {p_alt:.2e}")
l, m, n, k = 21, 9, 10, 5                                     # 式 9.7：印刷版 vs 按 X 求和
pmf = lambda x: math.comb((l - 1) * m, n - x) * math.comb(m, x) / math.comb(l * m, n)
print(f"式9.7  l=21,m=9,n=10,k=5：按 X 求和 {sum(pmf(x) for x in range(k, m + 1)):.3e}；"
      f"照印刷版（被加项含 k）{(m - k + 1) * pmf(k):.3e}")
score = lambda n, P, U, c: (n - 2) * math.log(1 + c * P / (1 + U)) if n > 3 else 0.0
print(f"式9.8  n=4,sumP=40,sumU=9：De Paoli（系数10）{score(4, 40, 9, 10):.2f}；教材（系数9）{score(4, 40, 9, 9):.2f}")

# ---------- 模拟位点：背景 sRNA + 一个 21 nt 相位位点 ----------
rng = np.random.default_rng(7)
L, start, cycles = 3000, 1200, 10
reads = {}                                     # (链, 比对最左坐标) -> 读段数
for c in range(cycles):                        # 相位位点：正链 5' 端在 start+21c，反义链最左坐标偏 -2
    reads[("+", start + 21 * c)] = int(rng.poisson(15)) + 1
    if rng.random() < 0.7:
        reads[("-", start + 21 * c - 2)] = int(rng.poisson(8)) + 1
for _ in range(400):                           # 背景：随机位置、低丰度
    key = (str(rng.choice(["+", "-"])), int(rng.integers(0, L)))
    reads[key] = reads.get(key, 0) + int(rng.geometric(0.4))
merged = {}                                    # 两链合并：负链坐标 +2
for (strand, x), cnt in reads.items():
    key = (strand, x + 2 if strand == "-" else x)
    merged[key] = merged.get(key, 0) + cnt

tot = np.zeros(L + 3)
for (_, x), cnt in merged.items():
    tot[x] += cnt
P_howell = np.array([howell([tot[w + 21 * i] for i in range(8)]) if w + 147 < len(tot) else 0
                     for w in range(L)])
w_best = int(P_howell.argmax())
inside = lambda x: start - 21 <= x <= start + 21 * cycles
print(f"Howell 最大 P={P_howell[w_best]:.1f}，窗口起点 {w_best}；相位位点以外最大 P="
      f"{max(p for w, p in enumerate(P_howell) if not inside(w + 63)):.1f}")

anchors, pvals = sorted(merged), []
pos = set(merged)
for (s0, a) in anchors:                        # Chen 2007：以每个 sRNA 为起点的 231 nt 窗口
    others = [(s, x) for (s, x) in pos if a <= x <= a + 230 and (s, x) != (s0, a)]
    k = sum((x - a) % 21 == 0 for (_, x) in others)
    pvals.append(chen(len(others), k) if others else 1.0)
pvals = np.array(pvals)
m_t = len(pvals)                               # Benjamini-Hochberg
order = np.argsort(pvals)
q = np.minimum.accumulate((pvals[order] * m_t / np.arange(1, m_t + 1))[::-1])[::-1]
sig = np.zeros(m_t, bool)
sig[order] = q <= 0.05
in_locus = np.array([start - 232 <= a <= start + 21 * cycles for (_, a) in anchors])  # 窗口与位点重叠
print(f"检验窗口 {m_t} 个：名义 P<0.05 的背景窗口 {int(((pvals < 0.05) & ~in_locus).sum())} 个；"
      f"BH(FDR 0.05) 后显著 {int(sig.sum())} 个，其中窗口与相位位点重叠的 {int((sig & in_locus).sum())} 个；"
      f"最小 P={pvals.min():.1e}")
