"""第 3 章：手写 GenBank flat file 解析器，并把特征转成 GFF3 / BED；验算 GFF3 的 phase 与 GTF 的 frame。

运行：python ch03_genbank_gff.py
输入：本目录的 U49845.1.gb（教材图 3.4 那条酵母记录，由 ch03_eutils_ena.py 从 NCBI 下载缓存）。
只用标准库。装了 Biopython 时，最后用 Bio.SeqIO 交叉核对一遍翻译结果。
"""
from pathlib import Path
import io
import re

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------ GenBank 解析
def parse_genbank(text):
    """逐条产出记录（以 // 切分）。返回 dict：locus、accession、version、definition、features、seq。"""
    for block in text.split("\n//"):
        if "LOCUS" not in block:
            continue
        lines = block.strip("\n").splitlines()
        rec = {"features": []}
        i = 0
        while i < len(lines):
            line = lines[i]
            key = line[:12].strip()
            if key == "LOCUS":
                # 按空白分词，不按固定列（GenBank 发布说明 3.4.4.1 的建议）
                tok = line.split()
                rec["locus"] = {"name": tok[1], "length": int(tok[2]), "mol": tok[4],
                                "topology": tok[5], "division": tok[6], "date": tok[7]}
            elif key == "DEFINITION":
                d = [line[12:].strip()]
                while i + 1 < len(lines) and lines[i + 1].startswith(" " * 12):
                    i += 1
                    d.append(lines[i].strip())
                rec["definition"] = " ".join(d)
            elif key == "ACCESSION":
                rec["accession"] = line[12:].split()[0]
            elif key == "VERSION":
                rec["version"] = line[12:].split()[0]
            elif key == "FEATURES":
                i = _parse_features(lines, i + 1, rec["features"]) - 1
            elif key == "ORIGIN":
                rec["seq"] = "".join(re.sub(r"[^A-Za-z]", "", l) for l in lines[i + 1:]).upper()
                break
            i += 1
        yield rec


def _parse_features(lines, i, out):
    """FEATURES 区：第 6–20 列是特征键，第 22 列起是位置或 /限定词；续行以 21 个空格开头。"""
    cur = None
    while i < len(lines) and lines[i].startswith(" "):
        line = lines[i]
        if line[5] != " ":                                  # 新特征
            cur = {"key": line[5:21].strip(), "loc": line[21:].strip(), "q": {}}
            out.append(cur)
            last = "loc"
        else:
            body = line[21:].strip()
            if body.startswith("/"):
                k, _, v = body[1:].partition("=")
                cur["q"][k] = v
                last = k
            elif last == "loc":                             # 位置串跨行
                cur["loc"] += body
            else:                                           # 限定词值跨行（translation 不加空格）
                sep = "" if last == "translation" else " "
                cur["q"][last] += sep + body
        i += 1
    for f in out:
        f["q"] = {k: v.strip('"') for k, v in f["q"].items()}
    return i


def parse_location(loc):
    """INSDC 位置串 -> [(start, end, strand, partial5, partial3)]，按特征自身 5'->3' 的顺序排列；坐标 1-based 闭区间。"""
    loc = loc.replace(" ", "")
    if loc.startswith("complement(") and loc.endswith(")"):
        segs = parse_location(loc[len("complement("):-1])
        return [(s, e, -st, p3, p5) for (s, e, st, p5, p3) in reversed(segs)]
    for op in ("join(", "order("):
        if loc.startswith(op) and loc.endswith(")"):
            parts, depth, buf = [], 0, ""
            for ch in loc[len(op):-1]:
                if ch == "," and depth == 0:
                    parts.append(buf)
                    buf = ""
                    continue
                depth += (ch == "(") - (ch == ")")
                buf += ch
            parts.append(buf)
            return [seg for p in parts for seg in parse_location(p)]
    m = re.fullmatch(r"(<?)(\d+)(?:\.\.(>?)(\d+))?", loc)
    if not m:
        raise ValueError(f"不支持的位置写法：{loc}")
    s = int(m.group(2))
    e = int(m.group(4)) if m.group(4) else s
    return [(s, e, 1, bool(m.group(1)), bool(m.group(3)))]


COMP = str.maketrans("ACGTN", "TGCAN")


def extract(seq, segs):
    out = []
    for s, e, strand, _, _ in segs:
        piece = seq[s - 1:e]                                # 1-based 闭区间 -> Python 切片
        out.append(piece.translate(COMP)[::-1] if strand < 0 else piece)
    return "".join(out)


BASES = "TCAG"
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"   # NCBI 遗传密码表 1
CODON = {a + b + c: AA[16 * i + 4 * j + k]
         for i, a in enumerate(BASES) for j, b in enumerate(BASES) for k, c in enumerate(BASES)}


def translate(nt):
    return "".join(CODON.get(nt[i:i + 3], "X") for i in range(0, len(nt) - 2, 3))


text = (HERE / "U49845.1.gb").read_text(encoding="utf-8")
rec = next(parse_genbank(text))
print("== 1. LOCUS 与版本")
print(rec["locus"])
print("ACCESSION", rec["accession"], "| VERSION", rec["version"], "| ORIGIN 长度", len(rec["seq"]),
      "| 与 LOCUS 一致:", len(rec["seq"]) == rec["locus"]["length"])

print("\n== 2. 逐个 CDS：按位置取序列、按 codon_start 翻译，与记录里的 /translation 比较")
for f in rec["features"]:
    if f["key"] != "CDS":
        continue
    segs = parse_location(f["loc"])
    cs = int(f["q"].get("codon_start", 1))
    nt = extract(rec["seq"], segs)
    prot = translate(nt[cs - 1:])
    stop = prot.endswith("*")
    prot = prot.rstrip("*")
    same = prot == f["q"]["translation"]
    print(f"{f['loc']:24s} {f['q']['protein_id']:11s} nt={len(nt):5d} codon_start={cs} "
          f"末尾终止密码子={'有' if stop else '无'} aa={len(prot):4d} 与/translation一致={same}  {prot[:12]}...")

print("\n== 3. 转成 GFF3（1-based 闭区间，phase = codon_start - 1）与 BED（0-based 半开）")
gff, bed = ["##gff-version 3", f"##sequence-region {rec['version']} 1 {rec['locus']['length']}"], []
genes = {f["q"]["gene"] for f in rec["features"] if f["key"] == "gene"}
for n, f in enumerate(rec["features"], 1):
    segs = parse_location(f["loc"])
    lo, hi = min(s for s, *_ in segs), max(e for _, e, *_ in segs)
    strand = "+" if segs[0][2] > 0 else "-"
    typ = {"source": "region"}.get(f["key"], f["key"])
    name = f["q"].get("gene") or f["q"].get("product") or rec["version"]
    attrs = [f"ID={typ.lower()}{n}", f"Name={name}"]
    if f["key"] in ("mRNA", "CDS") and f["q"].get("gene") in genes:
        attrs.append(f"Parent=gene-{f['q']['gene']}")
    if f["key"] == "gene":
        attrs[0] = f"ID=gene-{name}"
        bed.append(f"{rec['version']}\t{lo - 1}\t{hi}\t{name}\t0\t{strand}")
    if any(p5 or p3 for *_, p5, p3 in segs):
        attrs.append("partial=true")            # 小写开头的属性可由应用自定义（GFF3 规范）
    phase = str(int(f["q"].get("codon_start", 1)) - 1) if f["key"] == "CDS" else "."
    gff.append("\t".join([rec["version"], "GenBank", typ, str(lo), str(hi), ".", strand, phase, ";".join(attrs)]))
print("\n".join(gff))
print("BED:")
print("\n".join(bed))


# ------------------------------------------------------------------ GFF3 / GTF 的 phase
def next_phase(length, phase):
    """GTF2.2 规范给出的公式：下一段的 frame/phase = (3 - ((length - phase) mod 3)) mod 3。"""
    return (3 - ((length - phase) % 3)) % 3


spec_example = """\
ctg123\t.\tCDS\t1201\t1500\t.\t+\t0\tID=cds00001;Parent=mRNA00001
ctg123\t.\tCDS\t3000\t3902\t.\t+\t0\tID=cds00001;Parent=mRNA00001
ctg123\t.\tCDS\t5000\t5500\t.\t+\t0\tID=cds00001;Parent=mRNA00001
ctg123\t.\tCDS\t7000\t7600\t.\t+\t0\tID=cds00001;Parent=mRNA00001
ctg123\t.\tCDS\t1201\t1500\t.\t+\t0\tID=cds00002;Parent=mRNA00002
ctg123\t.\tCDS\t5000\t5500\t.\t+\t0\tID=cds00002;Parent=mRNA00002
ctg123\t.\tCDS\t7000\t7600\t.\t+\t0\tID=cds00002;Parent=mRNA00002
ctg123\t.\tCDS\t3301\t3902\t.\t+\t0\tID=cds00003;Parent=mRNA00003
ctg123\t.\tCDS\t5000\t5500\t.\t+\t1\tID=cds00003;Parent=mRNA00003
ctg123\t.\tCDS\t7000\t7600\t.\t+\t1\tID=cds00003;Parent=mRNA00003
ctg123\t.\tCDS\t3391\t3902\t.\t+\t0\tID=cds00004;Parent=mRNA00003
ctg123\t.\tCDS\t5000\t5500\t.\t+\t1\tID=cds00004;Parent=mRNA00003
ctg123\t.\tCDS\t7000\t7600\t.\t+\t1\tID=cds00004;Parent=mRNA00003
"""
print("\n== 4. GFF3 规范示例（EDEN 基因）的 CDS phase：按片段长度推算 vs 文件中给出的值")
groups = {}
for line in spec_example.splitlines():
    c = line.split("\t")
    cid = dict(kv.split("=") for kv in c[8].split(";"))["ID"]
    groups.setdefault(cid, []).append((int(c[3]), int(c[4]), int(c[7])))
for cid, segs in groups.items():
    segs.sort()                                         # 正链：按坐标升序就是 5'->3'
    phase, calc = segs[0][2], [segs[0][2]]
    for s, e, _ in segs[:-1]:
        phase = next_phase(e - s + 1, phase)
        calc.append(phase)
    given = [p for *_, p in segs]
    total = sum(e - s + 1 for s, e, _ in segs)
    print(f"{cid}: 片段长度 {[e - s + 1 for s, e, _ in segs]} 给出 {given} 推算 {calc} "
          f"一致={given == calc}；CDS 总长 {total}，除以 3 余 {total % 3}")

print("\n== 5. GTF2.2 规范示例（Twinscan 基因）的 frame")
gtf = [(380, 401, 0), (501, 650, 2), (700, 707, 2)]
frame, calc = gtf[0][2], [gtf[0][2]]
for s, e, _ in gtf[:-1]:
    frame = next_phase(e - s + 1, frame)
    calc.append(frame)
print("给出", [f for *_, f in gtf], "推算", calc, "；CDS 总长", sum(e - s + 1 for s, e, _ in gtf),
      "（GTF 的 CDS 不含终止密码子，708-710 另记为 stop_codon）")


# ------------------------------------------------------------------ 原笔记 parse_gff 的 bug
def parse_gff_old(lines):                               # 原笔记 §4C 的写法
    cols = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        rec = dict(zip(cols, f))
        rec["start"], rec["end"] = int(rec["start"]), int(rec["end"])
        yield rec


def parse_gff3(lines):
    """修正版：遇到 ##FASTA 就停止解析特征；第 9 列拆成字典并做百分号解码；返回 (特征列表, 剩余 FASTA 行)。"""
    from urllib.parse import unquote
    feats, it = [], iter(lines)
    for line in it:
        if line.startswith("##FASTA") or line.startswith(">"):
            rest = [] if line.startswith("##FASTA") else [line]
            return feats, rest + list(it)
        if line.startswith("#") or not line.strip():
            continue
        c = line.rstrip("\n").split("\t")
        if len(c) != 9:
            raise ValueError(f"列数不是 9：{line[:40]!r}")
        attrs = {}
        for kv in filter(None, c[8].split(";")):
            k, _, v = kv.partition("=")
            attrs[unquote(k)] = [unquote(x) for x in v.split(",")]
        feats.append({"seqid": c[0], "type": c[2], "start": int(c[3]), "end": int(c[4]),
                      "strand": c[6], "phase": c[7], "attrs": attrs})
    return feats, []


gff_with_fasta = gff[:3] + ["##FASTA", ">U49845.1", rec["seq"][:60]]
print("\n== 6. 带 ##FASTA 段的 GFF3（Prokka 等注释工具的常见输出）")
try:
    list(parse_gff_old(gff_with_fasta))
except Exception as e:
    print("原写法报错：", type(e).__name__, e)
feats, fasta_lines = parse_gff3(gff_with_fasta)
print("修正版：特征", len(feats), "个；FASTA 行", len(fasta_lines), "行；第一个特征属性", feats[0]["attrs"])

try:
    from Bio import SeqIO, __version__ as biover
    ok = 0
    for r in SeqIO.parse(io.StringIO(text), "genbank"):
        for f in r.features:
            if f.type == "CDS":
                ok += str(f.translate(r.seq, cds=False, to_stop=True)) == f.qualifiers["translation"][0]
    print(f"\n== 7. Biopython {biover} 交叉核对：3 个 CDS 中 {ok} 个翻译与 /translation 一致")
except ImportError:
    print("\n== 7. 未安装 Biopython，跳过交叉核对")
