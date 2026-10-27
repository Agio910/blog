# -*- coding: utf-8 -*-
"""第 5 章：由图 5.8A 的小联配估计一个 profile HMM，再用 Viterbi 和前向算法给序列打分。
状态结构按图 5.8B（Durbin 等的标准结构，含 D→I、I→D 转移）。伪计数：转移与匹配态发射都加 1。
插入态发射取背景（均匀 1/20），所以插入残基的对数几率为 0。只依赖 numpy。"""
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
q = 1 / 20                                  # 背景（零模型）频率
aln = ["N.FLS", "N.FLS", "NKYLT", "Q.W-T"]   # 第 2 列是插入列（'.' 表示该序列此处没有残基）
match_cols = [0, 2, 3, 4]                     # 半数以上序列有残基的列 -> M1..M4
K = len(match_cols)


def path_of(row):
    """把一行联配翻译成状态路径（不含 B、E）。"""
    states, k = [], 0
    for c, ch in enumerate(row):
        if c in match_cols:
            k += 1
            states.append(f"D{k}" if ch == "-" else f"M{k}")
        elif ch not in ".-":
            states.append(f"I{k}")
    return states


def nexts(s):
    """图 5.8B 中每个状态允许的去向。"""
    kind, k = s[0], int(s[1:]) if s not in ("B", "E") else 0
    if s == "B":
        kind, k = "M", 0
    tgt = []
    tgt.append(f"M{k + 1}" if k < K else "E")
    tgt.append(f"I{k}")
    if k < K:
        tgt.append(f"D{k + 1}")
    return tgt


states = ["B"] + [f"{t}{k}" for k in range(K + 1) for t in "MID" if not (t in "MD" and k == 0)] + ["E"]
# ---- 计数（加 1 伪计数）----
T = {s: {t: 1.0 for t in nexts(s)} for s in states if s != "E"}
Em = {k: np.ones(20) for k in range(1, K + 1)}
for row in aln:
    p = ["B"] + path_of(row) + ["E"]
    for a, b in zip(p, p[1:]):
        T[a][b] += 1
    res = [ch for ch in row if ch not in ".-"]
    for st, ch in zip([x for x in p[1:-1] if x[0] in "MI"], res):
        if st[0] == "M":
            Em[int(st[1:])][AA.index(ch)] += 1
logT = {s: {t: np.log2(v / sum(d.values())) for t, v in d.items()} for s, d in T.items()}
logE = {k: np.log2(v / v.sum() / q) for k, v in Em.items()}   # 匹配态发射的对数几率（bits）
print("训练序列的状态路径：")
for row in aln:
    print(f"  {row}: B " + " ".join(path_of(row)) + " E")
print("M1→M2、M1→I1、M1→D2 的转移概率:",
      {t: round(float(2 ** v), 3) for t, v in logT["M1"].items()})


def emit(st, ch):
    return logE[int(st[1:])][AA.index(ch)] if st[0] == "M" else 0.0


def decode(x, mode="viterbi"):
    """返回相对零模型的对数几率（bits）；mode='forward' 时对所有路径求和。"""
    n = len(x)
    order = ["B"] + [f"{t}{k}" for k in range(K + 1) for t in "DMI" if not (t in "MD" and k == 0)]
    V = {s: np.full(n + 1, -np.inf) for s in order + ["E"]}
    ptr = {s: [None] * (n + 1) for s in order + ["E"]}
    V["B"][0] = 0.0
    comb = (lambda a: max(a)) if mode == "viterbi" else (lambda a: np.log2(np.sum(np.exp2(a))))
    for i in range(n + 1):
        for s in order[1:] + ["E"]:
            if s == "E" and i != n:
                continue
            emitting = s[0] in "MI"
            if emitting and i == 0:
                continue
            j = i - 1 if emitting else i
            cand = [(V[p][j] + logT[p][s], p) for p in order if s in logT[p] and V[p][j] > -np.inf]
            if not cand:
                continue
            e = emit(s, x[i - 1]) if emitting else 0.0
            vals = [c for c, _ in cand]
            V[s][i] = comb(vals) + e
            ptr[s][i] = max(cand)[1]
    # 回溯 Viterbi 路径
    path, s, i = [], "E", n
    while s != "B":
        p = ptr[s][i]
        if s != "E":
            path.append(s)
        if s[0] in "MI":
            i -= 1
        s = p
    return V["E"][n], path[::-1]


print("\n序列        Viterbi 对数几率   前向对数几率   Viterbi 路径")
for x in ["NFLS", "NKYLT", "QWT", "NKFLS", "QWLT", "AAAA"]:
    v, path = decode(x)
    f, _ = decode(x, "forward")
    print(f"  {x:8s} {v:10.2f} {f:14.2f}      " + " ".join(path))
