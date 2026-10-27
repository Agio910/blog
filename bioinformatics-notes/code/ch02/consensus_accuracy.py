# -*- coding: utf-8 -*-
"""第 2 章 §6.2/§7.3："错误随机 → 加深度可纠正"在什么条件下成立。

单个位点、n 条独立读长（或同一分子的 n 次读取，即 CCS 的 pass），按多数投票取一致碱基。
三种错误模型：
  A 随机替换：读错时等概率读成另外 3 种碱基之一；
  B 最坏情形：读错时总读成同一种错误碱基；
  C 系统误差：在"难读"位点，每条读长以概率 s 读成同一种错误碱基（如同聚物长度系统性读短），
    其余错误仍按 A 随机分布。
精确枚举多项分布，平票时随机选一个。只用标准库。
"""
import math
from itertools import product


def compositions(n, k):
    """n 拆成 k 个非负整数之和的所有方式。"""
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in compositions(n - i, k - 1):
            yield (i,) + rest


def multinom_pmf(counts, probs):
    n = sum(counts)
    lp = math.lgamma(n + 1)
    for c, p in zip(counts, probs):
        if c:
            if p == 0:
                return 0.0
            lp += c * math.log(p) - math.lgamma(c + 1)
    return math.exp(lp)


def p_consensus_correct(n, probs):
    """probs[0] 是读出正确碱基的概率，其余 3 个是三种错误碱基。"""
    tot = 0.0
    for cnt in compositions(n, 4):
        m = max(cnt)
        if cnt[0] != m:
            continue
        ties = sum(1 for c in cnt if c == m)
        tot += multinom_pmf(cnt, probs) / ties
    return tot


def phred(err):
    return float("inf") if err <= 0 else -10 * math.log10(err)


eps = 0.125  # 教材 p.25：SMRT 单碱基准确率 87.5%
A = (1 - eps, eps / 3, eps / 3, eps / 3)
B = (1 - eps, eps, 0.0, 0.0)
print("单读长准确率 %.3f（Q%.1f）" % (1 - eps, phred(eps)))
print("\n深度   模型A(随机替换)          模型B(错误都指向同一碱基)")
for n in (1, 3, 5, 10, 15, 20, 30):
    a = p_consensus_correct(n, A)
    b = p_consensus_correct(n, B)
    print("%3d    %.6f (Q%5.1f)       %.6f (Q%5.1f)" % (n, a, phred(1 - a), b, phred(1 - b)))

print("\n模型 C：难读位点上每条读长以概率 s 读成同一错误碱基（另加 2% 随机错误）")
print("深度   s=0.20    s=0.35    s=0.45    s=0.55")
for n in (1, 5, 10, 20, 30):
    row = []
    for s in (0.20, 0.35, 0.45, 0.55):
        r = 0.02
        probs = (1 - s - r, s + r / 3, r / 3, r / 3)
        row.append(p_consensus_correct(n, probs))
    print("%3d    " % n + "   ".join("%.4f" % x for x in row))
print("s < 0.5 时一致准确率随深度上升，但 s 越接近 0.5 上升越慢；s > 0.5 时深度越大越确定地错。")
