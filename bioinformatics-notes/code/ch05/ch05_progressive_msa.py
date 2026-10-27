# -*- coding: utf-8 -*-
"""第 5 章：多序列联配的打分、精确解与渐进式启发。
1) 表 5.1 的“哈希”位差；2) 精确多维 DP 的规模；3) 图 5.2 相似度矩阵的聚类次序；
4) 三条序列：精确 SP 最优（三维 DP）对比渐进式联配。只依赖 numpy。"""
import itertools
import numpy as np

# ---------- 1. 表 5.1：同一残基在两条序列中的位差 ----------
print("== 表 5.1")
seq1 = {i + 1: a for i, a in enumerate("NCSPTA")}        # 教材统一地址：seq1 占 1–6
seq2 = {i + 6: a for i, a in enumerate("ACSPRK")}        # seq2 占 6–11
diag = {}
for (i, a), (j, b) in itertools.product(seq1.items(), seq2.items()):
    if a == b:
        diag.setdefault(i - j, []).append(a)
print("位差 -> 命中残基:", diag)

# ---------- 2. 精确多维 DP 的规模 ----------
print("\n== 精确 DP：N 条长 300 的序列")
for n in range(2, 9):
    print(f"N={n}: 格点 {(301) ** n:.2e}，每格前驱 {2 ** n - 1}，配对打分 {n * (n - 1) // 2} 对")

# ---------- 3. 图 5.2 相似度矩阵的平均连锁聚类 ----------
names = ["HA1", "HB1", "HA2", "HB2", "MY", "PIL", "LG"]
low = {("HB1", "HA1"): 2.11, ("HA2", "HA1"): 32.9, ("HA2", "HB1"): 19.7,
       ("HB2", "HA1"): 20.7, ("HB2", "HB1"): 39.0, ("HB2", "HA2"): 20.4,
       ("MY", "HA1"): 11.0, ("MY", "HB1"): 9.8, ("MY", "HA2"): 10.3, ("MY", "HB2"): 9.7,
       ("PIL", "HA1"): 9.3, ("PIL", "HB1"): 8.6, ("PIL", "HA2"): 9.6, ("PIL", "HB2"): 8.4,
       ("PIL", "MY"): 7.0, ("LG", "HA1"): 7.1, ("LG", "HB1"): 7.3, ("LG", "HA2"): 7.5,
       ("LG", "HB2"): 7.4, ("LG", "MY"): 7.3, ("LG", "PIL"): 4.3}


def average_linkage(sim):
    """在相似度上做 UPGMA：每次合并平均相似度最高的两簇。"""
    clusters = {n: [n] for n in names}
    order = []
    while len(clusters) > 1:
        best = None
        for a, b in itertools.combinations(clusters, 2):
            v = np.mean([sim[frozenset((x, y))] for x in clusters[a] for y in clusters[b]])
            if best is None or v > best[0]:
                best = (v, a, b)
        v, a, b = best
        new = f"({a},{b})"
        clusters[new] = clusters.pop(a) + clusters.pop(b)
        order.append(f"{new}@{v:.1f}")
    return order


print("\n== 图 5.2：按相似度平均连锁的合并次序")
for x in (2.11, 21.1):
    sim = {frozenset(k): v for k, v in low.items()}
    sim[frozenset(("HB1", "HA1"))] = x
    print(f"HA1-HB1 = {x}:", " -> ".join(average_linkage(sim)))

# ---------- 4. 精确 SP 最优 vs 渐进式（DNA，线性空位）----------
MATCH, MISMATCH, GAP = 2, -1, -2


def s(a, b):
    if a == "-" and b == "-":
        return 0
    if a == "-" or b == "-":
        return GAP
    return MATCH if a == b else MISMATCH


def sp(rows):
    return sum(s(r1[k], r2[k]) for r1, r2 in itertools.combinations(rows, 2)
               for k in range(len(rows[0])))


def align_profiles(A, B):
    """profile–profile 全局 DP，列得分 = 两组之间所有残基对的和（即 SP 的跨组部分）。
    已有的列整体保留，新空位整列插入：once a gap, always a gap。"""
    la, lb = len(A[0]), len(B[0])
    colA = [[r[i] for r in A] for i in range(la)]
    colB = [[r[j] for r in B] for j in range(lb)]
    gA, gB = ["-"] * len(A), ["-"] * len(B)
    cs = lambda X, Y: sum(s(x, y) for x in X for y in Y)
    F = np.full((la + 1, lb + 1), -1e18)
    T = np.zeros((la + 1, lb + 1), int)
    F[0, 0] = 0
    for i in range(la + 1):
        for j in range(lb + 1):
            if i and j and F[i - 1, j - 1] + cs(colA[i - 1], colB[j - 1]) > F[i, j]:
                F[i, j], T[i, j] = F[i - 1, j - 1] + cs(colA[i - 1], colB[j - 1]), 0
            if i and F[i - 1, j] + cs(colA[i - 1], gB) > F[i, j]:
                F[i, j], T[i, j] = F[i - 1, j] + cs(colA[i - 1], gB), 1
            if j and F[i, j - 1] + cs(gA, colB[j - 1]) > F[i, j]:
                F[i, j], T[i, j] = F[i, j - 1] + cs(gA, colB[j - 1]), 2
    i, j, out = la, lb, []
    while i or j:
        t = T[i, j]
        if t == 0:
            out.append(colA[i - 1] + colB[j - 1]); i -= 1; j -= 1
        elif t == 1:
            out.append(colA[i - 1] + gB); i -= 1
        else:
            out.append(gA + colB[j - 1]); j -= 1
    out.reverse()
    return ["".join(c[r] for c in out) for r in range(len(A) + len(B))]


def identity_distance(a, b):
    x, y = align_profiles([a], [b])
    pairs = [(u, v) for u, v in zip(x, y) if u != "-" and v != "-"]
    return 1 - sum(u == v for u, v in pairs) / max(len(pairs), 1)


def progressive3(seqs):
    """三条序列：先联配距离最近的一对，再把第三条对齐到这个 profile 上。"""
    d = {p: identity_distance(seqs[p[0]], seqs[p[1]]) for p in itertools.combinations(range(3), 2)}
    i, j = min(d, key=d.get)
    k = 3 - i - j
    pair = align_profiles([seqs[i]], [seqs[j]])
    rows = align_profiles(pair, [seqs[k]])
    back = {i: rows[0], j: rows[1], k: rows[2]}
    return [back[t] for t in range(3)], (i, j), pair


def exact3(a, b, c):
    """三维 DP 求 SP 最优，7 种移动（非空子集前进一格）。"""
    moves = [m for m in itertools.product((0, 1), repeat=3) if any(m)]
    F = np.full((len(a) + 1, len(b) + 1, len(c) + 1), -1e18)
    F[0, 0, 0] = 0
    T = {}
    for i in range(len(a) + 1):
        for j in range(len(b) + 1):
            for k in range(len(c) + 1):
                for m in moves:
                    pi, pj, pk = i - m[0], j - m[1], k - m[2]
                    if min(pi, pj, pk) < 0:
                        continue
                    x = a[pi] if m[0] else "-"
                    y = b[pj] if m[1] else "-"
                    z = c[pk] if m[2] else "-"
                    v = F[pi, pj, pk] + s(x, y) + s(x, z) + s(y, z)
                    if v > F[i, j, k]:
                        F[i, j, k], T[i, j, k] = v, (m, x, y, z)
    i, j, k = len(a), len(b), len(c)
    cols = []
    while i or j or k:
        m, x, y, z = T[i, j, k]
        cols.append((x, y, z))
        i, j, k = i - m[0], j - m[1], k - m[2]
    cols.reverse()
    return ["".join(col[r] for col in cols) for r in range(3)], F[-1, -1, -1]


def mutate(seq, rng, p_sub=0.25, p_indel=0.1):
    out = []
    for ch in seq:
        r = rng.random()
        if r < p_indel / 2:
            continue                                   # 删除
        out.append(rng.choice(list("ACGT")) if rng.random() < p_sub else ch)
        if r > 1 - p_indel / 2:
            out.append(rng.choice(list("ACGT")))       # 插入
    return "".join(out) or "A"


print("\n== 三条序列：精确 SP 最优 vs 渐进式（match +2, mismatch -1, gap -2）")
rng = np.random.default_rng(1)
worse, gaps, example = 0, [], None
TRIALS = 300
for t in range(TRIALS):
    anc = "".join(rng.choice(list("ACGT"), size=8))
    tri = [mutate(anc, rng) for _ in range(3)]
    opt, best = exact3(*tri)
    prog, first, pair = progressive3(tri)
    assert sp(opt) == best
    if sp(prog) < best:
        worse += 1
        gaps.append(best - sp(prog))
        if example is None:
            example = (tri, opt, best, prog, sp(prog), first, pair)
print(f"{TRIALS} 组随机三序列中，渐进式 SP 分低于最优的有 {worse} 组；"
      f"差距平均 {np.mean(gaps):.2f}，最大 {max(gaps):.0f}")
tri, opt, best, prog, ps, first, pair = example
print("例：输入 s1, s2, s3 =", tri)
print(f"  最优 SP = {best:.0f}")
for r in opt:
    print("    " + r)
print(f"  渐进式：先联配最近的一对 s{first[0] + 1}、s{first[1] + 1}（{pair[0]} / {pair[1]}），"
      f"再加入第三条，SP = {ps}")
for r in prog:
    print("    " + r)
