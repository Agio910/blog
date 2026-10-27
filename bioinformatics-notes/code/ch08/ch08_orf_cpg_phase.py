# -*- coding: utf-8 -*-
"""第 8 章 §7.1：六框 ORF（含边缘截断）、CpG 岛窗口扫描、内含子相位与 GFF3 phase 的换算。"""
import numpy as np

STOP = {"TAA", "TAG", "TGA"}
COMP = str.maketrans("ACGTN", "TGCAN")
rng = np.random.default_rng(1)

def revcomp(s):
    return s.translate(COMP)[::-1]

def find_orfs(seq, min_aa=80, partial=True):
    """按“终止到终止”的片段扫描六个读框。返回 (链, 起, 止, 氨基酸数, 类型)，
    坐标为正链 0-based 半开区间。片段碰到序列边缘且 partial=True 时，
    认为基因可能延伸到序列之外（5' 端缺起始或 3' 端缺终止），这里只标记边缘截断可能性，不复现 Prodigal 的判定算法。"""
    L, out = len(seq), []
    for strand, s in (("+", seq), ("-", revcomp(seq))):
        for f in range(3):
            cod = [s[i:i + 3] for i in range(f, len(s) - 2, 3)]
            stops = [k for k, c in enumerate(cod) if c in STOP]
            bounds = [-1] + stops + [len(cod)]
            for left, right in zip(bounds[:-1], bounds[1:]):
                l_edge, r_edge = left == -1, right == len(cod)
                if l_edge and partial:
                    k0 = 0                       # 5' 端可能在序列之外
                else:
                    atg = [k for k in range(left + 1, right) if cod[k] == "ATG"]
                    if not atg:
                        continue
                    k0 = atg[0]                  # 取最上游的 ATG，即最长 ORF
                if r_edge and not partial:
                    continue
                aa = right - k0
                if aa < min_aa:
                    continue
                a = f + 3 * k0
                b = f + 3 * right + (0 if r_edge else 3)   # 完整 ORF 含终止密码子
                kind = {(False, False): "complete", (True, False): "5'partial",
                        (False, True): "3'partial", (True, True): "both"}[(l_edge and partial, r_edge)]
                a, b = (a, b) if strand == "+" else (L - b, L - a)
                out.append((strand, a, b, aa, kind))
    return sorted(out, key=lambda o: o[1])

def random_seq(n, p=(0.25, 0.25, 0.25, 0.25)):
    return "".join(rng.choice(list("ACGT"), size=n, p=p))

def planted_gene(n_codons):
    sense = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT" if a + b + c not in STOP]
    return "ATG" + "".join(rng.choice(sense, size=n_codons)) + "TAA"

# ---------- 1. ORF：两个完整基因（正、负链各一）+ 一个被序列右端截断的基因 ----------
g1, g2, g3 = planted_gene(150), planted_gene(120), planted_gene(200)
parts = [random_seq(300), g1, random_seq(400), revcomp(g2), random_seq(300), g3[:330]]
seq = "".join(parts)
pos = np.cumsum([0] + [len(x) for x in parts])
print("序列长度 %d；植入: + 链 [%d,%d)，- 链 [%d,%d)，+ 链 [%d,%d) 右端截断"
      % (len(seq), pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]))
for o in find_orfs(seq):
    print("  %s %5d %5d %4d aa  %s" % o)
n_rand = len(find_orfs(random_seq(100000), partial=False))
print("10 万 bp 随机序列中 >=80 aa 的完整 ORF:", n_rand)

# ---------- 2. CpG 岛：甲基化导致的 CpG 缺乏背景 + 一段 600 bp 的 GC 富集区 ----------
def deplete_cpg(s, keep=0.25):
    s = list(s)
    for i in range(len(s) - 1):
        if s[i] == "C" and s[i + 1] == "G" and rng.random() > keep:
            s[i] = "T"                          # 甲基化 C 脱氨成 T，CpG 逐渐流失
    return "".join(s)

def cpg_islands(seq, w=200, step=20, gc_min=0.5, oe_min=0.6):
    """Gardiner-Garden & Frommer (1987) 的窗口判据，合并相邻达标窗口。"""
    hits = []
    for i in range(0, len(seq) - w + 1, step):
        s = seq[i:i + w]
        c, g, cg = s.count("C"), s.count("G"), s.count("CG")
        if (c + g) / w > gc_min and c * g and cg * w / (c * g) > oe_min:
            if hits and i <= hits[-1][1]:
                hits[-1][1] = i + w
            else:
                hits.append([i, i + w])
    return hits

def obs_exp(s):
    return s.count("CG") * len(s) / (s.count("C") * s.count("G"))

bg = deplete_cpg(random_seq(6000, (0.29, 0.21, 0.21, 0.29)))
island = random_seq(600, (0.17, 0.33, 0.33, 0.17))
genome = bg[:3000] + island + bg[3000:]
print("背景 CpG 观测/期望 = %.2f；岛内 = %.2f" % (obs_exp(bg), obs_exp(island)))
print("检出的 CpG 岛（植入区间 [3000,3600)）:", cpg_islands(genome))

# ---------- 3. 内含子相位（GENSCAN 的下标）与 GFF3 第 8 列 phase ----------
cds_len = [100, 152, 88, 62]                    # 一个基因四段 CDS 的长度，总长 402 = 134 个密码子
cum = np.cumsum([0] + cds_len)
intron_phase = [int(c % 3) for c in cum[1:-1]]  # 内含子插在密码子第几个碱基之后
gff3_phase = [int((3 - c % 3) % 3) for c in cum[:-1]]   # 每段 CDS 开头要跳过几个碱基
print("内含子相位:", intron_phase, " 各段 CDS 的 GFF3 phase:", gff3_phase)
