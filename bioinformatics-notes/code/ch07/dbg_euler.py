# -*- coding: utf-8 -*-
"""第 7 章 §8.2：边中心德布鲁因图（节点 = (k-1)-mer，边 = k-mer）、
Hierholzer 欧拉路径、欧拉路径枚举与 unitig 压缩。只用标准库。"""
from collections import Counter, defaultdict

def kmer_counts(seqs, k):
    return Counter(s[i:i + k] for s in seqs for i in range(len(s) - k + 1))

def build_dbg(counts, multiplicity=True):
    """每个 k-mer 连一条 前缀->后缀 的边；multiplicity=False 时每种 k-mer 只连一条。"""
    g = defaultdict(list)
    for km, c in sorted(counts.items()):
        g[km[:-1]].extend([km[1:]] * (c if multiplicity else 1))
    return g

def degree_check(g):
    out_d = {u: len(vs) for u, vs in g.items()}
    in_d = Counter(v for vs in g.values() for v in vs)
    nodes = set(out_d) | set(in_d)
    diff = {n: out_d.get(n, 0) - in_d.get(n, 0) for n in nodes}
    starts = [n for n, d in diff.items() if d == 1]
    ends = [n for n, d in diff.items() if d == -1]
    ok = all(abs(d) <= 1 for d in diff.values()) and len(starts) <= 1 and len(ends) <= 1
    return ok, starts, ends, {n: d for n, d in sorted(diff.items()) if d}

def euler_path(g):
    """Hierholzer 算法，O(|E|)。度条件不满足时返回 None（原笔记的代码缺这一步）。"""
    if not any(g.values()):
        return ""
    ok, starts, _, _ = degree_check(g)
    if not ok:
        return None
    adj = {u: list(vs) for u, vs in g.items()}
    stack, path = [starts[0] if starts else next(iter(adj))], []
    while stack:
        u = stack[-1]
        if adj.get(u):
            stack.append(adj[u].pop())
        else:
            path.append(stack.pop())
    path.reverse()
    if len(path) - 1 != sum(len(vs) for vs in g.values()):
        return None                                   # 图不连通
    return path[0] + ''.join(p[-1] for p in path[1:])

def all_euler_strings(g):
    """小图上用回溯枚举所有欧拉路径拼出的不同序列。"""
    ok, starts, _, _ = degree_check(g)
    if not ok:
        return set()
    m = sum(len(vs) for vs in g.values())
    edges = Counter((u, v) for u, vs in g.items() for v in vs)
    out = set()

    def dfs(u, seq, used):
        if used == m:
            out.add(seq)
            return
        for (a, b), c in list(edges.items()):
            if a == u and c > 0:
                edges[(a, b)] -= 1
                dfs(b, seq + b[-1], used + 1)
                edges[(a, b)] += 1
    for s in (starts or list(g)):
        dfs(s, s, 0)
    return out

def unitigs(g):
    """返回最长不分叉路径及孤立环；unitig 是组装中间对象，不等同于最终 contig。"""
    in_d = Counter(v for vs in g.values() for v in vs)
    out_d = {u: len(vs) for u, vs in g.items()}
    simple = lambda n: in_d.get(n, 0) == 1 and out_d.get(n, 0) == 1
    res, used = [], set()
    for u in sorted(g):
        if simple(u):
            continue
        for i, v in enumerate(g[u]):
            used.add((u, i))
            s = u + v[-1]
            while simple(v):
                used.add((v, 0))
                v = g[v][0]
                s += v[-1]
            res.append(s)
    for u in sorted(g):
        for i, v in enumerate(g[u]):
            if (u, i) in used:
                continue
            used.add((u, i))
            s = u + v[-1]
            while v != u:
                used.add((v, 0))
                v = g[v][0]
                s += v[-1]
            res.append(s)
    return res

def report(name, seqs, k, multiplicity=True):
    g = build_dbg(kmer_counts(seqs, k), multiplicity)
    ok, starts, ends, unbalanced = degree_check(g)
    print(f"== {name}（k={k}，{'保留' if multiplicity else '丢弃'} k-mer 拷贝数）")
    print("   边数", sum(len(v) for v in g.values()), " 度不平衡节点(出-入)", unbalanced)
    sols = all_euler_strings(g)
    print("   欧拉路径:", euler_path(g), " 不同解的个数:", len(sols))
    if 1 < len(sols) <= 3:
        for s in sorted(sols):
            print("     ", s)
    print("   unitig:", ', '.join(f"{u}x{c}" if c > 1 else u
                                   for u, c in sorted(Counter(unitigs(g)).items())))

# 习题 3：两条序列本身（已知每个 k-mer 的拷贝数）
report("seq1 ATGGCTCAGTAGGC", ["ATGGCTCAGTAGGC"], 4)
report("seq2 ATGGCTTTCAGTAGAGAGC", ["ATGGCTTTCAGTAGAGAGC"], 4)
report("seq2，只知道 k-mer 集合", ["ATGGCTTTCAGTAGAGAGC"], 4, multiplicity=False)

# 原笔记的例子：两条互相重叠的读序。直接数 k-mer 会把重叠部分记两次
reads = ["ATGGCTCAGT", "TCAGTAGGC"]
report("两条读序，原样计数", reads, 4)
report("两条读序，k-mer 去重", reads, 4, multiplicity=False)

# 三份相同的重复 R（长度 6 > k）把基因组切成可以互换的片段
R = "CTTGAC"
genome = "AAT" + R + "GGA" + R + "TTC" + R + "CCG"
report("含三拷贝重复的基因组 " + genome, [genome], 4)
report("同一基因组，k-1 = 重复长度", [genome], 7)
report("同一基因组，k-1 > 重复长度", [genome], 8)

# 空图、孤立环、断开分量：防止教学代码默默漏掉结果。
assert euler_path({}) == ""
cycle = build_dbg(Counter({"ATG": 1, "TGA": 1, "GAT": 1}))
assert unitigs(cycle) == ["ATGAT"]
assert euler_path({"AA": ["AA"], "CC": ["CC"]}) is None
print("空图、孤立环、不连通图检查：通过")
