"""第 4 章：Needleman–Wunsch、Smith–Waterman 与 Gotoh 仿射空位（教学版，含回溯）。

运行：python nw_sw.py
依赖：numpy；Biopython（只用来读取 BLOSUM50）。
"""
import random
import numpy as np
from Bio.Align import substitution_matrices

NEG = float("-inf")


# ---------- 1. 线性空位的全局联配，列出全部共最优联配 ----------
def nw_all(a, b, s, g):
    """F[i][j] = a[:i] 与 b[:j] 的最优全局得分；长度为 k 的空位得分为 -g*k。"""
    m, n = len(a), len(b)
    F = [[0] * (n + 1) for _ in range(m + 1)]
    back = [[()] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        F[i][0], back[i][0] = -g * i, ("U",)
    for j in range(1, n + 1):
        F[0][j], back[0][j] = -g * j, ("L",)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c = {"D": F[i - 1][j - 1] + s(a[i - 1], b[j - 1]),  # a_i 与 b_j 配对
                 "U": F[i - 1][j] - g,                          # a_i 对空位
                 "L": F[i][j - 1] - g}                          # b_j 对空位
            F[i][j] = max(c.values())
            back[i][j] = tuple(k for k in "DUL" if c[k] == F[i][j])
    alns = []

    def walk(i, j, x, y):
        if i == j == 0:
            alns.append(("".join(reversed(x)), "".join(reversed(y))))
            return
        for d in back[i][j]:
            if d == "D":
                walk(i - 1, j - 1, x + [a[i - 1]], y + [b[j - 1]])
            elif d == "U":
                walk(i - 1, j, x + [a[i - 1]], y + ["-"])
            else:
                walk(i, j - 1, x + ["-"], y + [b[j - 1]])

    walk(m, n, [], [])
    return F, alns


# ---------- 2. Gotoh 三矩阵：仿射空位，全局或局部，单条回溯 ----------
def gotoh(a, b, s, o, e, local=False):
    """长度为 k 的空位罚 o + e*k（即式 4.1 的 a + bk）。
    M：以 a_i~b_j 结尾；X：以 a_i~空位 结尾（竖直）；Y：以 空位~b_j 结尾（水平）。"""
    m, n = len(a), len(b)
    M = np.full((m + 1, n + 1), NEG)
    X, Y, H = M.copy(), M.copy(), M.copy()
    H[0, 0] = 0
    for i in range(1, m + 1):
        H[i, 0] = 0 if local else -(o + e * i)
        if not local:
            X[i, 0] = H[i, 0]
    for j in range(1, n + 1):
        H[0, j] = 0 if local else -(o + e * j)
        if not local:
            Y[0, j] = H[0, j]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            M[i, j] = H[i - 1, j - 1] + s(a[i - 1], b[j - 1])
            X[i, j] = max(H[i - 1, j] - o - e, X[i - 1, j] - e)
            Y[i, j] = max(H[i, j - 1] - o - e, Y[i, j - 1] - e)
            H[i, j] = max(M[i, j], X[i, j], Y[i, j], 0 if local else NEG)
    i, j = np.unravel_index(np.argmax(H), H.shape) if local else (m, n)
    score, st, x, y = H[i, j], "H", [], []
    while True:                     # 回溯时要记住当前在哪个矩阵（状态）
        if st == "H":
            if (i == 0 and j == 0) or (local and H[i, j] == 0):
                break
            st = "M" if H[i, j] == M[i, j] else ("X" if H[i, j] == X[i, j] else "Y")
        if st == "M":
            x.append(a[i - 1]); y.append(b[j - 1]); i, j, st = i - 1, j - 1, "H"
        elif st == "X":
            x.append(a[i - 1]); y.append("-")
            st = "H" if X[i, j] == H[i - 1, j] - o - e else "X"; i -= 1
        else:
            x.append("-"); y.append(b[j - 1])
            st = "H" if Y[i, j] == H[i, j - 1] - o - e else "Y"; j -= 1
    return score, "".join(reversed(x)), "".join(reversed(y)), (i, j)


def aln_score(x, y, s, o, e):
    """按仿射空位直接给一条联配打分（用来核对回溯结果）。"""
    sc, prev = 0, None
    for p, q in zip(x, y):
        if p == "-":
            sc -= e + (o if prev != "Y" else 0); prev = "Y"
        elif q == "-":
            sc -= e + (o if prev != "X" else 0); prev = "X"
        else:
            sc += s(p, q); prev = "M"
    return sc


def all_alignments(a, b):
    """穷举两条序列的全部联配（只用于很短的序列，验证 DP 的正确性）。"""
    if not a and not b:
        yield "", ""
        return
    if a and b:
        for x, y in all_alignments(a[1:], b[1:]):
            yield a[0] + x, b[0] + y
    if a:
        for x, y in all_alignments(a[1:], b):
            yield a[0] + x, "-" + y
    if b:
        for x, y in all_alignments(a, b[1:]):
            yield "-" + x, b[0] + y


# ---------- 3. 按教材式（4.14）–（4.16）重算表 4.4 ----------
def sw_table44(a, b, q_uses="Q"):
    """教材的 Smith–Waterman 例子。为避免浮点误差，所有分值乘以 3：
    匹配 1→3，错配 -1/3→-1，w_1 = 1+1/3 → 4，u = 1/3 → 1。
    初始条件照教材：P[0,j] = Q[i,0] = 0（不是 -inf）。
    q_uses='Q'：Q_ij = max(H_{i,j-1} - w1, Q_{i,j-1} - u)；'P'：照印刷版写成 P_{i,j-1}。"""
    m, n = len(a), len(b)
    H, P, Q = (np.zeros((m + 1, n + 1), int) for _ in range(3))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            P[i, j] = max(H[i - 1, j] - 4, P[i - 1, j] - 1)
            prev = Q[i, j - 1] if q_uses == "Q" else P[i, j - 1]
            Q[i, j] = max(H[i, j - 1] - 4, prev - 1)
            H[i, j] = max(0, H[i - 1, j - 1] + (3 if a[i - 1] == b[j - 1] else -1),
                          P[i, j], Q[i, j])
    return H, P, Q


# 印刷版表 4.4（p.92），行 i = 1..10（CTATAATCCC），列 j = 1..7（CTGTATC）
T44 = {
    "H": """1.00 0 0 0 0 0 1.00 | 0 2.00 0.67 1.00 0 1.00 0 | 0 0.67 1.67 0.33 2.00 0.67 0.67 |
            0 1.00 0.33 2.67 1.33 3.00 1.67 | 0 0 0.67 1.33 3.67 2.33 2.67 | 0 0 0 1.00 2.33 3.33 2.00 |
            0 1.00 0 1.00 2.00 3.33 3.00 | 1.00 0 0.67 0.33 1.67 2.00 4.33 | 1.00 0.67 0 0.33 1.33 1.67 3.00 |
            1.00 0.67 0.33 0 1.00 1.33 2.67""",
    "P": """-0.33 -0.33 -0.33 -0.33 -0.33 -0.33 -0.33 | -0.33 -0.67 -0.67 -0.67 -0.67 -0.67 -0.33 |
            -0.67 0.67 -0.67 -0.33 -1.00 -0.33 -0.67 | -1.00 0.33 0.33 -0.67 0.67 -0.67 -0.67 |
            -1.33 0 0 1.33 0 1.67 0.33 | -1.33 -0.33 -0.33 1.00 2.33 1.33 1.33 |
            -1.33 -0.67 -0.67 0.67 2.00 2.00 1.00 | 1.33 -0.33 -1.00 0.33 1.67 2.00 1.67 |
            -0.33 -0.67 -0.67 0 1.33 1.67 3.00 | -0.33 -0.67 -1.00 -0.33 1.00 1.33 2.67""",
    "Q": """-0.33 -0.33 -0.67 -1.00 -1.33 -1.33 -1.33 | -0.33 -0.67 0.67 0.33 0 -0.33 -0.33 |
            -0.33 -0.67 -0.67 0.33 0 0.67 0.33 | -0.33 -0.67 -0.33 -0.67 1.33 1.00 1.67 |
            -0.33 -0.67 -1.00 -0.67 0.00 2.33 2.00 | -0.33 -0.67 -1.00 -1.33 -0.33 1.00 2.00 |
            -0.33 -0.67 -0.33 -0.67 -0.33 0.67 2.00 | -0.33 -0.33 -0.67 -0.67 1.00 0.33 0.67 |
            -0.33 -0.33 -0.67 -1.00 -1.00 0 0.33 | -0.33 -0.33 -0.67 -1.00 -1.33 -0.33 0""",
}


def printed(name):
    rows = [r.split() for r in T44[name].split("|")]
    return np.array([[round(float(v) * 3) for v in r] for r in rows])


if __name__ == "__main__":
    # (1) 教材第一节的小例子：匹配 +1、错配 -1、空位 -3
    s1 = lambda x, y: 1 if x == y else -1
    for loc in (False, True):
        sc, x, y, _ = gotoh("TGCGGAGC", "TCGGAGC", s1, 0, 3, local=loc)
        print("TGCGGAGC/TCGGAGC %s：得分 %d  %s / %s" % ("局部" if loc else "全局", sc, x, y))

    # (2) 教材 NW 例子（图 4.2）：匹配 0、错配 -1、空位 -3
    a, b = "CTGTATC", "CTATAATCCC"
    F, alns = nw_all(a, b, lambda x, y: 0 if x == y else -1, 3)
    print("\n图 4.2：最优得分 =", F[-1][-1], "，共最优联配条数 =", len(alns))
    for x, y in alns:
        print("   ", y, "/", x)
    print("H 矩阵（行 = CTGTATC，列 = CTATAATCCC；印刷版只印绝对值）：")
    for row in F:
        print("   ", " ".join(f"{v:4d}" for v in row))

    # (3) 表 4.4：式（4.15）用 Q 还是用 P
    A, B = "CTATAATCCC", "CTGTATC"
    for variant in ("Q", "P"):
        mats = dict(zip("HPQ", sw_table44(A, B, variant)))
        bad = {k: int((M[1:, 1:] != printed(k)).sum()) for k, M in mats.items()}
        print(f"\n式（4.15）写成 {variant}_(i,j-1) 时，与印刷版不一致的格子数：{bad}")
    mats = dict(zip("HPQ", sw_table44(A, B, "Q")))
    for k, M in mats.items():
        for i, j in zip(*np.nonzero(M[1:, 1:] != printed(k))):
            print(f"   {k}({i + 1},{j + 1})：递推 {M[i + 1, j + 1] / 3:.2f}，印刷 {printed(k)[i, j] / 3:.2f}")
    H = mats["H"]
    i, j = np.unravel_index(np.argmax(H), H.shape)
    print(f"max H = {H[i, j] / 3:.2f}，位于 (i, j) = ({i}, {j})")
    sc, x, y, _ = gotoh(A, B, lambda p, q: 3 if p == q else -1, 3, 1, local=True)
    print(f"Gotoh 局部联配：得分 {sc / 3:.2f}  {y} / {x}")

    # (4) 习题 2：BLOSUM50，线性空位 -8
    B50 = substitution_matrices.load("BLOSUM50")
    book = {("A", "A"): 5, ("A", "E"): -1, ("A", "G"): 0, ("A", "H"): -2, ("A", "P"): -1,
            ("A", "W"): -3, ("E", "E"): 6, ("E", "G"): -3, ("E", "H"): 0, ("E", "P"): -1,
            ("E", "W"): -3, ("G", "G"): 8, ("G", "H"): -2, ("G", "P"): -2, ("G", "W"): -3,
            ("H", "H"): 10, ("H", "P"): -2, ("H", "W"): -3, ("P", "P"): 10, ("P", "W"): -4,
            ("W", "W"): 15}
    print("\n习题 2 的 BLOSUM50 子表与标准矩阵一致:", all(B50[p, q] == v for (p, q), v in book.items()))
    s50 = lambda x, y: int(B50[x, y])
    F, alns = nw_all("AGWGAHEA", "PAWHEAEAG", s50, 8)
    print("F 矩阵（行 = AGWGAHEA，列 = PAWHEAEAG）：")
    for row in F:
        print("   ", " ".join(f"{v:4d}" for v in row))
    print("全局最优得分 =", F[-1][-1], "，共最优联配条数 =", len(alns))
    for x, y in alns:
        cols = ["gap" if "-" in (p, q) else s50(p, q) for p, q in zip(x, y)]
        print("   ", x, "/", y, " 逐列:", cols)
    sc, x, y, _ = gotoh("AGWGAHEA", "PAWHEAEAG", s50, 0, 8, local=True)
    print("同一计分下的局部最优：%d  %s / %s" % (sc, x, y))

    # (5) 穷举验证：短随机序列上 Gotoh（全局、局部）= 暴力枚举的最优值
    rng = random.Random(1)
    s2 = lambda x, y: 2 if x == y else -1
    ok = 0
    for t in range(300):
        p = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 5)))
        q = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 5)))
        g = max(aln_score(x, y, s2, 3, 1) for x, y in all_alignments(p, q))
        loc = max([0] + [aln_score(x, y, s2, 3, 1)
                         for i1 in range(len(p)) for i2 in range(i1 + 1, len(p) + 1)
                         for j1 in range(len(q)) for j2 in range(j1 + 1, len(q) + 1)
                         for x, y in all_alignments(p[i1:i2], q[j1:j2])])
        rg, xg, yg, _ = gotoh(p, q, s2, 3, 1)
        rl, xl, yl, _ = gotoh(p, q, s2, 3, 1, local=True)
        ok += (rg == g and rl == loc and aln_score(xg, yg, s2, 3, 1) == g
               and (rl == 0 or aln_score(xl, yl, s2, 3, 1) == loc))
    print(f"\n穷举核对 300 对短序列（o=3, e=1）：全局、局部得分及回溯联配的得分全部一致 {ok}/300")
