"""第 4 章：BLAST 的"邻域词"种子与 X-drop 无空位延伸（玩具版）。

运行：python blast_seed.py    依赖：numpy、Biopython（读取 BLOSUM62）
"""
import itertools
import numpy as np
from Bio.Align import substitution_matrices

AA = "ARNDCQEGHILKMFPSTWYV"
B62 = substitution_matrices.load("BLOSUM62")
S = np.array([[int(B62[a, b]) for b in AA] for a in AA])
WORDS = np.array(list(itertools.product(range(20), repeat=3)))      # 全部 8000 个 3 肽


def neighborhood(word, T):
    w = [AA.index(c) for c in word]
    sc = S[w[0], WORDS[:, 0]] + S[w[1], WORDS[:, 1]] + S[w[2], WORDS[:, 2]]
    keep = np.argsort(-sc)[: (sc >= T).sum()]
    return ["".join(AA[i] for i in WORDS[k]) + f"({sc[k]})" for k in keep]


for T in (13, 11):
    nb = neighborhood("PQG", T)
    print(f"PQG 的邻域词（T={T}）共 {len(nb)} 个：", " ".join(nb))

rng = np.random.default_rng(0)
for T in (11, 13):
    sizes = [len(neighborhood("".join(rng.choice(list(AA), 3)), T)) for _ in range(300)]
    print(f"随机 3 肽的邻域大小（T={T}）：平均 {np.mean(sizes):.1f}，中位数 {np.median(sizes):.0f}")


def xdrop_extend(q, d, qi, dj, w=3, X=20):
    """从种子 (qi, dj) 起向右做无空位延伸：累计分比历史最高分低 X 就停。
    返回（最高分，最高分所在位置，停止位置）。"""
    best = cur = sum(S[AA.index(q[qi + k]), AA.index(d[dj + k])] for k in range(w))
    end = k = w
    while qi + k < len(q) and dj + k < len(d):
        cur += S[AA.index(q[qi + k]), AA.index(d[dj + k])]
        k += 1
        if cur > best:
            best, end = cur, k
        elif best - cur > X:
            break
    return best, end, k


q = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"      # 53 个残基
d = q[:30] + "GSGSGS" + q[30:]       # 在第 30 位后插入 6 个残基，后半段因此错开了对角线
for X in (5, 20, 60):
    b, e, stop = xdrop_extend(q, d, 0, 0, X=X)
    print(f"X = {X:2d}：无空位延伸得分 {b}，最高分在第 {e} 位，走到第 {stop} 位停止")

from nw_sw import gotoh
sc, x, y, _ = gotoh(q, d, lambda a, b: int(B62[a, b]), 11, 1, local=True)
print("有空位的局部联配（11/1）得分", int(sc))
print("  ", x)
print("  ", y)
