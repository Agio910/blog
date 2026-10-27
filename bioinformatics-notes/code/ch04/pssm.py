"""第 4 章：教材 p.86–87 的 PSSM 例子，逐步重算，并与更规范的伪计数做法对比。

运行：python pssm.py    依赖：numpy
"""
import numpy as np

aln = ["AAGCTA", "TAACTA", "AAACTT", "AGGCTT", "AGACTT"]   # 教材 p.86 的 5 条序列
B = "AGCT"
L, N = len(aln[0]), len(aln)
counts = np.array([[sum(s[k] == b for s in aln) for k in range(L)] for b in B])
freq = counts / N                                   # 步骤 2：各列频率
bg = counts.sum(1) / counts.sum()                   # 背景频率 = 全部 30 个碱基中的比例
ratio = freq / bg[:, None]                          # 步骤 3：除以背景
pssm_book = np.log2(np.where(ratio > 0, ratio, 0.1))   # 步骤 4：教材把为 0 的比值换成 0.1

np.set_printoptions(precision=2, suppress=True)
print("背景频率 A,G,C,T =", bg.round(3))
print("步骤 3（比值）:\n", ratio.round(2))
print("步骤 4（教材做法，log2）:\n", pssm_book.round(1))
print("T 在 #1：比值 = %.3f，log2 = %.2f（教材印作 0.60 和 0.6）" % (ratio[3, 0], pssm_book[3, 0]))

# 更常见的做法：给每个碱基加与背景成比例的伪计数（总伪计数 beta=1），再取对数比
beta = 1.0
freq_pc = (counts + beta * bg[:, None]) / (N + beta)
pssm_pc = np.log2(freq_pc / bg[:, None])
print("伪计数版 PSSM（beta=1）:\n", pssm_pc.round(1) + 0.0)   # +0.0 去掉 -0.0 的显示


def scan(seq, W):
    idx = {b: i for i, b in enumerate(B)}
    return [sum(W[idx[c], k] for k, c in enumerate(seq[p:p + L])) for p in range(len(seq) - L + 1)]


s = scan("AGACTA", pssm_book)[0]
print("AGACTA 得分（教材 PSSM，未四舍五入）= %.2f，2^S = %.0f" % (s, 2 ** s))
print("AGACTA 得分（表中一位小数相加）= %.1f，2^7.5 = %.0f" % (1.0 + 1.6 + 0.6 + 2.6 + 1.7 + 0, 2 ** 7.5))
print("AGACTA 得分（伪计数版）= %.2f" % scan("AGACTA", pssm_pc)[0])

# 分数是"基序模型 / 背景模型"的对数似然比。它的零分布可以精确算出：
# 按背景频率枚举全部 4^6 = 4096 个 6 碱基词
import itertools
words = ["".join(w) for w in itertools.product(B, repeat=L)]
pw = np.array([np.prod([bg[B.index(c)] for c in w]) for w in words])
sw = np.array([scan(w, pssm_book)[0] for w in words])
p_val = pw[sw >= s - 1e-9].sum()
print("背景模型下单个窗口得分 >= %.2f 的概率 = %.2e；扫描 1 kb 序列（约 1000 个窗口）时期望的随机命中数 = %.2f"
      % (s, p_val, 1000 * p_val))

rng = np.random.default_rng(0)
seq = "".join(rng.choice(list(B), size=60, p=bg)) + "AGACTA" + "".join(rng.choice(list(B), size=60, p=bg))
sc = np.array(scan(seq, pssm_book))
top = np.argsort(sc)[::-1][:3]
print("126 bp 随机序列（第 60 位植入 AGACTA）得分最高的三个窗口:",
      [(int(p), seq[p:p + L], round(float(sc[p]), 2)) for p in top])
