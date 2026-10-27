# -*- coding: utf-8 -*-
"""第 6 章 代码 4：Fitch 小简约算法（给定树，求最少变更数），信息位点判定，
复现教材图 6.17、表 6.9、图 6.18，并对 5 个物种的全部 15 种无根树打分。纯 Python。"""
from collections import Counter


def fitch(tree, states, log=None):
    """tree：嵌套二元组，叶子是名字；states：叶子 -> 碱基。返回（状态集合，最少变更数）。
    交集非空取交集；为空取并集并计一次变更。无根树任选一条枝放根，得分不变。"""
    if isinstance(tree, str):
        return {states[tree]}, 0
    (s1, c1), (s2, c2) = fitch(tree[0], states, log), fitch(tree[1], states, log)
    s, c = (s1 & s2, c1 + c2) if s1 & s2 else (s1 | s2, c1 + c2 + 1)
    if log is not None:
        log.append("".join(sorted(s)))
    return s, c


def informative(column):
    """至少两种碱基各自出现至少两次。"""
    return sum(v >= 2 for v in Counter(column).values()) >= 2


# ---------- 1. 图 6.17：6 个物种一个位点 ----------
tree617 = ((("A", "B"), (("C", "D"), "E")), "F")      # 节点 1=(A,B) 2=(C,D) 3=(2,E) 4=(1,3) 5=(4,F)
site = dict(zip("ABCDEF", "CTGTAA"))
log = []
_, cost = fitch(tree617, site, log)
print("== 图 6.17 ==")
print("后序访问的内部节点集合（2,3,1,4,5 的顺序）:", log, " 最少变更数 =", cost)

# ---------- 2. 表 6.9：4 条序列 9 个位点 ----------
seqs = {"1": "AAGAGTGCA", "2": "AGCCGTGCA", "3": "AGATATCCG", "4": "AGAGATCCG"}
tops4 = {"((1,2),(3,4))": (("1", "2"), ("3", "4")),
         "((1,3),(2,4))": (("1", "3"), ("2", "4")),
         "((1,4),(2,3))": (("1", "4"), ("2", "3"))}
print("\n== 表 6.9：每个位点在三种拓扑上的最少变更数 ==")
print("位点  列    信息位点  " + "  ".join(tops4))
tot = Counter()
for k in range(9):
    col = {n: s[k] for n, s in seqs.items()}
    scores = [fitch(t, col)[1] for t in tops4.values()]
    for name, sc in zip(tops4, scores):
        tot[name] += sc
    print(f"{k + 1:>3}   {''.join(col.values())}  {str(informative(col.values())):>6}    "
          + "  ".join(f"{sc:^13}" for sc in scores))
print("合计:", dict(tot))
print("5 条序列的列 AAGGT 是否信息位点:", informative("AAGGT"),
      "；按教材'每个不同碱基至少出现两次'则不是")

# ---------- 3. 图 6.18：5 种灵长类线粒体序列的 5 个信息位点，枚举 15 种无根树 ----------
mt = {"hu": "CGCCG", "ch": "CGCCG", "go": "CGTCA", "or": "TATTA", "gi": "TACTA"}
names = list(mt)
print("\n== 图 6.18：15 种无根树的简约得分 ==")
res = []
for c in names:
    o = [x for x in names if x != c]
    for (a, b), (d, e) in [((o[0], o[1]), (o[2], o[3])), ((o[0], o[2]), (o[1], o[3])),
                           ((o[0], o[3]), (o[1], o[2]))]:
        tree = (((a, b), c), (d, e))                    # 在一条内枝上放根
        per_site = [fitch(tree, {n: mt[n][k] for n in names})[1] for k in range(5)]
        res.append((sum(per_site), per_site, f"(({a},{b}),{c},({d},{e}))"))
res.sort()
for total, per_site, nw in res[:4]:
    print(f"{total}  {per_site}  {nw}")
print("...；得分分布:", sorted(Counter(r[0] for r in res).items()))
