# -*- coding: utf-8 -*-
"""第 8 章 §7.4：BLAST 表格（outfmt 6 + qlen slen）取带覆盖度过滤的 best hit；GO 超几何富集 + BH 校正。
数据是手写的玩具例子，只用来演示计算。"""
import csv, io
from math import comb
from scipy.stats import hypergeom

hits_tsv = """\
g1\tsp|P1|KINASE\t45.0\t310\t0\t0\t20\t330\t5\t315\t1e-60\t250\t800\t320
g1\tsp|P2|ADAPTER\t62.0\t240\t0\t0\t400\t640\t1\t240\t1e-70\t300\t800\t900
g2\tsp|P3|TRANSPORTER\t38.0\t420\t0\t0\t3\t420\t10\t430\t1e-40\t180\t430\t450
g3\tsp|P4|HYPOTHETICAL\t28.0\t60\t0\t0\t1\t60\t1\t60\t1e-3\t35\t200\t210
"""
cols = "qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen".split()
best = {}
for r in csv.DictReader(io.StringIO(hits_tsv), fieldnames=cols, delimiter="\t"):
    qcov = (int(r["qend"]) - int(r["qstart"]) + 1) / int(r["qlen"])
    scov = (int(r["send"]) - int(r["sstart"]) + 1) / int(r["slen"])
    ok = float(r["evalue"]) <= 1e-5 and qcov >= 0.5 and scov >= 0.5
    print("%s %-18s bits=%-5s qcov=%.2f scov=%.2f %s" % (r["qseqid"], r["sseqid"], r["bitscore"], qcov, scov, "保留" if ok else "过滤"))
    if ok and float(r["bitscore"]) > best.get(r["qseqid"], (None, -1))[1]:
        best[r["qseqid"]] = (r["sseqid"], float(r["bitscore"]))
print("best hit:", best)

def bh(pvals):
    """Benjamini-Hochberg：q_(i) = min_{j>=i} p_(j) m / j"""
    m = len(pvals); order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m; run = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        run = min(run, pvals[i] * m / rank); q[i] = run
    return q

# 背景 N=10000 个有 GO 注释的基因，目标基因 n=200；每个术语给出 (M=背景中注释到该术语的基因数, k=目标中注释到的)
N, n = 10000, 200
terms = {"GO:A": (150, 12), "GO:B": (400, 15), "GO:C": (60, 1), "GO:D": (1000, 25), "GO:E": (30, 5)}
p = {t: hypergeom.sf(k - 1, N, M, n) for t, (M, k) in terms.items()}      # P(X >= k)
q = dict(zip(p, bh(list(p.values()))))
for t, (M, k) in terms.items():
    print("%s M=%4d k=%2d 期望=%.1f  p=%.2e  BH q=%.2e" % (t, M, k, n * M / N, p[t], q[t]))
M, k = terms["GO:A"]
direct = sum(comb(M, i) * comb(N - M, n - i) for i in range(k, min(n, M) + 1)) / comb(N, n)
print("GO:A 按式子直接求和: %.6e，scipy: %.6e" % (direct, p["GO:A"]))
