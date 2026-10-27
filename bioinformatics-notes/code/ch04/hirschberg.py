"""第 4 章：Hirschberg 线性空间全局联配（线性空位），与完整 DP 对照。

思路：只保留一行就能算出 NW 的最后一行得分。把 a 从中间切开，正向算 a 前半与 b 各前缀的得分，
反向算 a 后半与 b 各后缀的得分，两者相加取最大处就是最优路径穿过中间行的位置，然后递归两半。
时间仍是 O(mn)（约为完整 DP 的 2 倍），工作空间降为 O(n)。
运行：python hirschberg.py    依赖：无（标准库）；同目录 nw_sw.py 用于对照
"""
import random
from nw_sw import nw_all


def last_row(a, b, s, g):
    prev = [-g * j for j in range(len(b) + 1)]
    for i in range(1, len(a) + 1):
        cur = [-g * i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = max(prev[j - 1] + s(a[i - 1], b[j - 1]), prev[j] - g, cur[j - 1] - g)
        prev = cur
    return prev


def hirschberg(a, b, s, g):
    if not a:
        return "-" * len(b), b
    if not b:
        return a, "-" * len(a)
    if len(a) == 1:                                   # 一个残基：要么对上 b 的某一位，要么全对空位
        best = max(range(len(b)), key=lambda j: s(a, b[j]))
        if s(a, b[best]) - g * (len(b) - 1) >= -g * (len(b) + 1):
            return "-" * best + a + "-" * (len(b) - best - 1), b
        return a + "-" * len(b), "-" + b
    mid = len(a) // 2
    L = last_row(a[:mid], b, s, g)
    R = last_row(a[mid:][::-1], b[::-1], s, g)
    k = max(range(len(b) + 1), key=lambda j: L[j] + R[len(b) - j])
    x1, y1 = hirschberg(a[:mid], b[:k], s, g)
    x2, y2 = hirschberg(a[mid:], b[k:], s, g)
    return x1 + x2, y1 + y2


def score(x, y, s, g):
    return sum(-g if "-" in (p, q) else s(p, q) for p, q in zip(x, y))


if __name__ == "__main__":
    s = lambda p, q: 1 if p == q else -1
    rng = random.Random(0)
    ok = 0
    for _ in range(200):
        a = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 12)))
        b = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 12)))
        x, y = hirschberg(a, b, s, 2)
        F, _ = nw_all(a, b, s, 2)
        ok += (x.replace("-", "") == a and y.replace("-", "") == b and score(x, y, s, 2) == F[-1][-1])
    print("200 对随机短序列：Hirschberg 得分 = 完整 DP 得分，且联配合法：%d/200" % ok)
    a = "".join(rng.choice("ACGT") for _ in range(1500))
    b = "".join(rng.choice("ACGT") for _ in range(1200))
    x, y = hirschberg(a, b, s, 2)
    print("1500 × 1200：得分 %d（= 最后一行算法 %d）；完整 DP 需存 %.1f 万个格子，Hirschberg 每层只存 2 行"
          % (score(x, y, s, 2), last_row(a, b, s, 2)[-1], (len(a) + 1) * (len(b) + 1) / 1e4))
