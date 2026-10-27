# -*- coding: utf-8 -*-
"""第 2 章 §6.4/§7.4：UMI 去扩增重复、UMI 碰撞（生日问题）与 UMI 测序错误。

numpy 即可；随机种子固定。
"""
import math
import numpy as np

rng = np.random.default_rng(7)

print("== 1. UMI 碰撞：k 个分子各自随机带一个长度为 l 的 UMI，能看到几种不同的 UMI？==")
print("E[不同 UMI 数] = M(1-(1-1/M)^k)，M = 4^l")
print("  l        M          k=100     k=1000    k=10000   （括号内为低估比例）")
for l in (6, 8, 10, 12):
    M = 4 ** l
    cells = []
    for k in (100, 1000, 10000):
        e = M * (1 - (1 - 1 / M) ** k)
        cells.append("%8.1f(%4.1f%%)" % (e, 100 * (1 - e / k)))
    print("%3d  %10d   %s" % (l, M, "  ".join(cells)))
print("同一细胞、同一基因内才需要区分分子；10x v3 用 12 nt UMI，单基因单细胞的分子数通常远小于 4^12。")

print("\n== 2. 扩增偏好：两个基因各 300 个分子，B 的扩增效率更低 ==")
cycles, total_reads = 12, 8000
mol = {"A": 300, "B": 300}
eff = {"A": (0.95, 0.03), "B": (0.75, 0.10)}   # 每轮扩增效率的均值、分子间标准差
w = {}
for g in ("A", "B"):
    e = np.clip(rng.normal(eff[g][0], eff[g][1], size=mol[g]), 0.05, 1.0)
    w[g] = (1 + e) ** cycles                    # 每个原始分子扩增后的拷贝数
lib_A, lib_B = w["A"].sum(), w["B"].sum()
print("真实分子数 A:B = %d:%d" % (mol["A"], mol["B"]))
print("扩增后文库里 A:B 拷贝数之比 = %.2f" % (lib_A / lib_B))
# 测序 = 从文库里按拷贝数比例抽 8000 条读长；同一原始分子的读长带相同 UMI
nA = rng.binomial(total_reads, lib_A / (lib_A + lib_B))
reads, umis = {}, {}
for g, n in (("A", nA), ("B", total_reads - nA)):
    picks = rng.choice(mol[g], size=n, p=w[g] / w[g].sum())
    reads[g], umis[g] = n, len(np.unique(picks))   # 按 UMI 去重 = 数有几个不同的原始分子
print("读长计数 A:B = %d:%d  ->  比值 %.2f" % (reads["A"], reads["B"], reads["A"] / reads["B"]))
print("UMI 计数 A:B = %d:%d  ->  比值 %.2f" % (umis["A"], umis["B"], umis["A"] / umis["B"]))
print("（UMI 计数受测序深度限制：B 的读长少，部分分子一次也没被测到，所以 B 仍略低）")

print("\n== 3. UMI 本身的测序错误会凭空多出分子 ==")
L_umi, n_mol, reads_per_mol, err = 10, 50, 20, 0.01
alphabet = np.array(list("ACGT"))
true_umis = ["".join(rng.choice(alphabet, L_umi)) for _ in range(n_mol)]
observed = []
for u in true_umis:
    for _ in range(reads_per_mol):
        s = list(u)
        for i in range(L_umi):
            if rng.random() < err:
                s[i] = rng.choice([b for b in "ACGT" if b != s[i]])
        observed.append("".join(s))
uniq, counts = np.unique(observed, return_counts=True)
print("真实分子 %d 个；每个分子测 %d 条读长；UMI 每个碱基错误率 %.0f%%" % (n_mol, reads_per_mol, 100 * err))
print("直接数不同 UMI：%d 个" % len(uniq))


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


# 方向性合并的简化版：计数少的 UMI 若与计数 >= 2n-1 的 UMI 只差 1 个碱基，就并入后者（UMI-tools 的 directional 思路）
order = np.argsort(-counts)
parent = {}
for idx in order:
    u, c = uniq[idx], counts[idx]
    for jdx in order:
        if counts[jdx] < 2 * c - 1:
            break
        v = uniq[jdx]
        if v != u and v not in parent and hamming(u, v) == 1:
            parent[u] = v
            break
print("按汉明距离 1 合并后：%d 个" % (len(uniq) - len(parent)))
