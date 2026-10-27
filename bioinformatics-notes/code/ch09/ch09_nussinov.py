"""Nussinov 最大配对动态规划（含最小发夹环长 theta 与回溯），输出点括号结构。

递推（0 起编号，N[i][j] 为子串 i..j 的最大配对数）：
  N[i][j] = max( N[i][j-1],                                   # j 不配对
                 max_{i<=k<j-theta, (k,j) 可配} N[i][k-1] + 1 + N[k+1][j-1] )
j - i <= theta 的子串不可能含配对，N = 0。时间 O(n^3)，空间 O(n^2)。
"""
import random

PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}

def fill(seq, theta=3):
    n = len(seq)
    N = [[0] * n for _ in range(n)]
    ops = 0
    for span in range(theta + 1, n):
        for i in range(n - span):
            j = i + span
            best = N[i][j - 1]
            for k in range(i, j - theta):
                ops += 1
                if (seq[k], seq[j]) in PAIRS:
                    left = N[i][k - 1] if k > i else 0
                    best = max(best, left + 1 + N[k + 1][j - 1])
            N[i][j] = best
    return N, ops

def traceback(seq, N, theta=3):
    pairs, stack = [], [(0, len(seq) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i <= theta:
            continue
        if N[i][j] == N[i][j - 1]:            # 优先取"j 不配对"，结果之一
            stack.append((i, j - 1))
            continue
        for k in range(i, j - theta):
            if (seq[k], seq[j]) in PAIRS:
                left = N[i][k - 1] if k > i else 0
                if N[i][j] == left + 1 + N[k + 1][j - 1]:
                    pairs.append((k, j))
                    stack += [(i, k - 1), (k + 1, j - 1)]
                    break
    db = ["."] * len(seq)
    for k, j in pairs:
        db[k], db[j] = "(", ")"
    return "".join(db)

def enumerate_structs(seq, theta=3):
    """穷举所有不交叉、发夹环 >= theta 的结构（每个结构恰好生成一次），返回配对列表的列表。"""
    memo = {}

    def S(i, j):
        if j - i <= theta:
            return [()]
        if (i, j) in memo:
            return memo[(i, j)]
        out = list(S(i, j - 1))
        for k in range(i, j - theta):
            if (seq[k], seq[j]) in PAIRS:
                for a in S(i, k - 1):
                    for b in S(k + 1, j - 1):
                        out.append(a + ((k, j),) + b)
        memo[(i, j)] = out
        return out
    return S(0, len(seq) - 1)

if __name__ == "__main__":
    for name, seq in [("hairpin", "GGGAAAUCCC"),
                      ("two hairpins", "GCGCUUCGGCGC" + "AA" + "GGACUUCGGUCC")]:
        N, _ = fill(seq)
        db = traceback(seq, N)
        allS = enumerate_structs(seq)
        best = max(len(s) for s in allS)
        n_opt = sum(len(s) == best for s in allS)
        print(f"{name}: {seq}")
        print(f"  DP 最大配对数 {N[0][-1]}，穷举 {best}；结构总数 {len(allS)}，其中最优结构 {n_opt} 个")
        print(f"  回溯得到 {db}")

    random.seed(1)
    agree = 0
    for _ in range(200):
        s = "".join(random.choice("ACGU") for _ in range(random.randint(8, 14)))
        agree += fill(s)[0][0][-1] == max(len(t) for t in enumerate_structs(s))
    print(f"随机序列 200 条（长 8-14）：DP 与穷举一致 {agree} 条")
    for n in (50, 100, 200):
        s = "".join(random.choice("ACGU") for _ in range(n))
        print(f"n={n:3d} 内层循环次数 {fill(s)[1]:>9,d}")
