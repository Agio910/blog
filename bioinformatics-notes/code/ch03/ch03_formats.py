"""第 3 章：FASTA、FASTQ 与 Phred 质量值（标准库 + numpy，离线运行）。

运行：python ch03_formats.py
内容：
  1. 式（3.1）的双向换算，逐行验算教材表 3.2；Solexa 分数与 Phred 分数的差别
  2. 拆解教材图 3.2 的读段名与质量字符
  3. 严格的 FASTQ 解析器、质量编码（+33/+64）的判断、Q20/Q30 统计
  4. FASTA 解析器：原笔记版本在空描述行上的 bug 与修正
"""
import io
import math
import numpy as np


# ---------------------------------------------------------------- 1. Phred
def q_phred(p):            # 式（3.1）：Q = -10 lg e
    return -10 * math.log10(p)


def p_from_q(q):
    return 10 ** (-q / 10)


def q_solexa(p):           # Solexa/早期 Illumina：Q = -10 lg[p/(1-p)]
    return -10 * math.log10(p / (1 - p))


print("== 1a. 教材表 3.2 逐行验算（Phred+33）")
table32 = [(5, 13, "."), (1, 20, "5"), (0.1, 30, "?"), (0.01, 40, "I")]
print("错误率%   精确Q     取整Q  字符  教材Q/字符  一致")
for err_pct, q_book, ch_book in table32:
    q = q_phred(err_pct / 100)
    qi = round(q)
    ch = chr(qi + 33)
    print(f"{err_pct:6}  {q:8.3f}  {qi:6d}  {ch:>4}  {q_book:>5}/{ch_book:<4}  {qi == q_book and ch == ch_book}")

print("\n== 1b. 同一错误率下 Phred 与 Solexa 分数（低质量处差别大，高质量处几乎相同）")
for p in [0.5, 0.25, 0.1, 0.01, 0.001]:
    print(f"p={p:<6} Q_phred={q_phred(p):6.2f}  Q_solexa={q_solexa(p) + 0.0:6.2f}")

print("\n== 1c. 自测题：错误率 0.5% 的 Q 值与字符；字符 '#' 的 Q 值与错误率")
q = q_phred(0.005)
print(f"Q = {q:.2f} -> 取整 {round(q)} -> 字符 {chr(round(q) + 33)!r}")
print(f"'#': Q = {ord('#') - 33}, 错误率 = {p_from_q(ord('#') - 33):.3f}")

# ---------------------------------------------------------------- 2. 图 3.2
print("\n== 2. 教材图 3.2：Casava 1.8 之后的读段名，以及图中出现的质量字符")
header = "@ST-E00169:117:HG2T3CCXX:4:1101:23794:1432 1:N:0:NTGTCA"
left, right = header[1:].split(" ")
names = ["仪器", "运行号", "流动槽", "泳道", "tile", "x", "y"]
print(dict(zip(names, left.split(":"))))
print(dict(zip(["读段(1/2)", "是否被过滤(Y/N)", "控制位", "样本条码"], right.split(":"))))
for c in "#<AFK":
    qq = ord(c) - 33
    print(f"字符 {c}: Q{qq:<3d} 错误率 {p_from_q(qq):.2e}")


# ---------------------------------------------------------------- 3. FASTQ
def parse_fastq(handle):
    """严格的四行 FASTQ 解析：返回 (id, seq, qual_string)。不处理多行 FASTQ（规范上允许，但几乎没人用）。"""
    n = 0
    while True:
        h = handle.readline()
        if not h:
            return
        h = h.rstrip("\r\n")
        if not h:                       # 允许文件末尾空行
            continue
        seq = handle.readline().rstrip("\r\n")
        plus = handle.readline().rstrip("\r\n")
        qual = handle.readline().rstrip("\r\n")
        n += 1
        if not h.startswith("@"):
            raise ValueError(f"第 {n} 条：第一行应以 @ 开头：{h[:30]!r}")
        if not plus.startswith("+"):
            raise ValueError(f"第 {n} 条：第三行应以 + 开头")
        if len(seq) != len(qual):
            raise ValueError(f"第 {n} 条：序列 {len(seq)} 与质量 {len(qual)} 长度不等")
        yield h[1:].split()[0], seq, qual


def guess_offset(quals):
    """只看出现过的字符范围判断编码：出现 ASCII < 59 必为 +33；全部 >= 64 且出现 > 74('J') 多半是 +64。"""
    lo = min(min(map(ord, q)) for q in quals)
    hi = max(max(map(ord, q)) for q in quals)
    if lo < 59:
        return 33, (lo, hi)
    if lo >= 64 and hi > 74:
        return 64, (lo, hi)
    return None, (lo, hi)               # 无法判断，需要元数据


rng = np.random.default_rng(3)
n_reads, L = 2000, 100
# 模拟 Illumina 式的质量曲线：前段高、末端下降，再加噪声；Q 取整并截在 [2, 41]
base = 38 - 12 * (np.arange(L) / L) ** 3
Q = np.clip(np.rint(base + rng.normal(0, 4, size=(n_reads, L))), 2, 41).astype(int)
seqs = ["".join(rng.choice(list("ACGT"), L)) for _ in range(n_reads)]


def to_fastq(Qmat, offset):
    buf = io.StringIO()
    for i, (s, q) in enumerate(zip(seqs, Qmat)):
        buf.write(f"@read{i} sim\n{s}\n+\n{''.join(chr(x + offset) for x in q)}\n")
    buf.seek(0)
    return buf


for offset in (33, 64):
    recs = list(parse_fastq(to_fastq(Q, offset)))
    quals = [q for _, _, q in recs]
    guessed, rng_ascii = guess_offset(quals)
    qv = np.array([[ord(c) - 33 for c in q] for q in quals])   # 故意一律按 +33 解读
    print(f"\n== 3. 写成 Phred+{offset}：{len(recs)} 条读段，ASCII 范围 {rng_ascii}，判断为 +{guessed}")
    print(f"   一律按 +33 解读：平均 Q = {qv.mean():.2f}；Q30 比例 = {(qv >= 30).mean():.4f}")
    qv_ok = qv - (guessed - 33)
    print(f"   按判断出的偏移解读：平均 Q = {qv_ok.mean():.2f}；Q20 = {(qv_ok >= 20).mean():.4f}；Q30 = {(qv_ok >= 30).mean():.4f}")

bad = io.StringIO("@r1\nACGT\n+\nIII\n")
try:
    list(parse_fastq(bad))
except ValueError as e:
    print("\n   坏记录被拦下：", e)


# ---------------------------------------------------------------- 4. FASTA
def parse_fasta_old(lines):             # 原笔记的写法
    name, seq = None, []
    for line in lines:
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                yield name, "".join(seq)
            name, seq = line[1:], []
        else:
            seq.append(line)
    if name:
        yield name, "".join(seq)


def parse_fasta(lines):
    """返回 (id, description, seq)。id 取描述行第一个空白前的部分；忽略空行和以 ; 开头的旧式注释行。"""
    header, seq = None, []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if header is not None:
                yield (*_split_header(header), "".join(seq))
            header, seq = line[1:], []
        else:
            if header is None:
                raise ValueError("序列行出现在第一个描述行之前")
            seq.append(line)
    if header is not None:
        yield (*_split_header(header), "".join(seq))


def _split_header(h):
    parts = h.split(maxsplit=1)
    return (parts[0] if parts else ""), (parts[1] if len(parts) > 1 else "")


fasta_text = """>gi|332309159|ref|NM_181652.2| Homo sapiens peroxiredoxin 5 (PRDX5), mRNA
CGCGCCTGCGCAGTGGAGGCGGCCCAGGCCCGCCTTCC
GCAGGGTGTCGCCGCTGTGCC

>
NNNNACGT
>NM_181652.3 Homo sapiens peroxiredoxin 5 (PRDX5), transcript variant 3, mRNA
acgtACGTnn
""".splitlines()
print("\n== 4. FASTA：原写法与修正版对比（第二条记录的描述行是空的）")
print("原写法:", [(n[:20], len(s)) for n, s in parse_fasta_old(fasta_text)])
for sid, desc, s in parse_fasta(fasta_text):
    acc = sid.split("|")[3] if sid.startswith("gi|") else sid
    print(f"修正版: id={sid!r:32} accession.version={acc!r:16} len={len(s)} 小写字母={sum(c.islower() for c in s)}")
