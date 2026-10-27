"""第 4 章：由计分矩阵和背景频率计算无空位联配的 Karlin–Altschul 参数 λ、K、H，
并与教材表 4.5、表 4.6 印出的数值对比；再演示 bit score、E 值与 P 值的换算。

运行：python karlin_altschul.py    依赖：numpy、scipy、Biopython（读取 BLOSUM62）
"""
import numpy as np
from scipy.optimize import brentq
from Bio.Align import substitution_matrices


def karlin_altschul(scores, probs, tol=1e-15, kmax=2000):
    """scores：随机配对可能得到的整数分值；probs：对应概率（要求期望分值 < 0，且存在正分）。
    λ：sum p·e^{λs} = 1 的唯一正根；H = λ·E[s·e^{λs}]（单位：nat）；
    K：Karlin & Altschul (1990) 的级数公式（分值的最大公约数为 1 时），某一项小于 tol 时停止求和。"""
    scores, probs = np.asarray(scores), np.asarray(probs, float)
    assert probs @ scores < 0 and scores.max() > 0
    g = lambda lam: probs @ np.exp(lam * scores) - 1
    hi = 1.0
    while g(hi) < 0:
        hi *= 2
    lam = brentq(g, 1e-9, hi, xtol=1e-14)
    H = lam * probs @ (scores * np.exp(lam * scores))
    lo = scores.min()
    step = np.zeros(scores.max() - lo + 1)
    np.add.at(step, scores - lo, probs)
    dist, off, total = np.array([1.0]), 0, 0.0
    for k in range(1, kmax + 1):            # S_k = 随机游走走 k 步后的累计分值
        dist = np.convolve(dist, step)
        off += lo
        sup = np.arange(off, off + len(dist))
        term = dist[sup < 0] @ np.exp(lam * sup[sup < 0]) + dist[sup >= 0].sum()
        total += term / k
        if term / k < tol:
            break
    K = np.exp(-2 * total) / ((H / lam) * (1 - np.exp(-lam)))
    return lam, K, H


# (1) 表 4.5 的 BLASTN：匹配 +1、错配 -2，四种碱基等频
lam, K, H = karlin_altschul([1, -2], [0.25, 0.75])
print("DNA +1/-2   λ=%.5f  K=%.6f  H=%.5f   （表 4.5 无空位：1.33271、0.620991、1.12409）" % (lam, K, H))
print("            级数在某项 < 1e-4 时截断：K=%.6f；隐含的一致比例 = %.3f"
      % (karlin_altschul([1, -2], [0.25, 0.75], tol=1e-4)[1], 0.25 * np.exp(lam)))

# (2) 表 4.6 的 BLASTX：BLOSUM62 + BLAST 使用的 Robinson & Robinson (1991) 背景频率
RR = dict(A=78.05, R=51.29, N=44.87, D=53.64, C=19.25, Q=42.64, E=62.95, G=73.77, H=21.99,
          I=51.42, L=90.19, K=57.44, M=22.43, F=38.56, P=52.03, S=71.20, T=58.41, W=13.30,
          Y=32.16, V=64.41)
AA = "ARNDCQEGHILKMFPSTWYV"
p = np.array([RR[a] for a in AA]); p /= p.sum()
B62 = substitution_matrices.load("BLOSUM62")
S = np.array([[int(B62[a, b]) for b in AA] for a in AA])
P = np.outer(p, p)
lam, K, H = karlin_altschul(S.ravel(), P.ravel())
print("BLOSUM62    λ=%.6f  K=%.6f  H=%.6f （表 4.6 无空位：0.317606、0.133956、0.401215）" % (lam, K, H))
print("            级数在某项 < 1e-4 时截断：K=%.6f" % karlin_altschul(S.ravel(), P.ravel(), tol=1e-4)[1])
print("            期望分值 = %.3f；H = %.3f bit/配对；λ 的理论值（半比特单位）ln2/2 = %.4f"
      % (P.ravel() @ S.ravel(), H / np.log(2), np.log(2) / 2))
q = P * np.exp(lam * S)                       # 计分矩阵"隐含"的目标频率
print("            隐含目标频率之和 = %.6f；隐含的一致比例 = %.3f" % (q.sum(), np.trace(q)))

# (3) bit score 与 E 值（表 4.5 的有空位参数 λ=1.28, K=0.46；图 4.9 的 HSP 原始分 522）
lam_g, K_g, raw = 1.28, 0.46, 522
bits = (lam_g * raw - np.log(K_g)) / np.log(2)
space = 238_058_802_518_966                    # 表 4.5 的有效搜索空间 m'n'
print("\n原始分 522 → bit score = %.1f（图 4.9 印 965 bits）；E = m'n'·2^-S' = %.1e" % (bits, space * 2.0 ** -bits))
print("同一搜索空间下 E = 0.001 所需的 bit score = %.1f" % np.log2(space / 1e-3))

# (4) E 与 P：P = 1 - exp(-E)
for E in (10, 5, 3, 1, 0.1, 0.01, 0.001):
    print("E = %-6g P = %.5f" % (E, 1 - np.exp(-E)))
