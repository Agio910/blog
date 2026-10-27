"""第 3 章：用 NCBI E-utilities 与 ENA API 程序化访问数据库（只用标准库）。

运行：python ch03_eutils_ena.py
需要联网。数字随数据库每日更新而变化，笔记里贴的是 2026-10-23 的运行结果。
同时把 GenBank 记录 U49845.1 缓存到本目录的 U49845.1.gb，供 ch03_genbank_gff.py 离线使用。

NCBI 的限速：不带 API key 每秒最多 3 次请求（响应头 X-RateLimit-Limit: 3），
带 key 每秒 10 次。这里每次请求后停 0.4 秒。
"""
from pathlib import Path
import json
import time
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
ENA_PORTAL = "https://www.ebi.ac.uk/ena/portal/api/"
ENA_BROWSER = "https://www.ebi.ac.uk/ena/browser/api/"
TOOL = "bioinfo_notes_ch03"          # NCBI 建议带 tool 参数；这里不带邮箱等个人信息


def get(url, params=None, pause=0.4):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as r:
        body = r.read().decode("utf-8")
    time.sleep(pause)
    return body


def einfo(db):
    d = json.loads(get(EUTILS + "einfo.fcgi", {"db": db, "retmode": "json", "tool": TOOL}))
    info = d["einforesult"]["dbinfo"][0]
    return int(info["count"]), info["lastupdate"]


def esearch_count(db, term):
    d = json.loads(get(EUTILS + "esearch.fcgi",
                       {"db": db, "term": term, "retmax": 0, "retmode": "json", "tool": TOOL}))
    return int(d["esearchresult"]["count"])


def efetch(db, uid, rettype, retmode="text"):
    return get(EUTILS + "efetch.fcgi",
               {"db": db, "id": uid, "rettype": rettype, "retmode": retmode, "tool": TOOL})


print("== 1. einfo：各 Entrez 数据库的记录数")
for db in ["nuccore", "protein", "gene", "structure", "taxonomy"]:
    n, upd = einfo(db)
    print(f"{db:10s} {n:>13,d}  (last update {upd})")

print("\n== 2. esearch：只要计数（retmax=0）")
queries = [
    ("nuccore", "txid10239[Organism:exp]"),                      # 病毒界 Viruses 的全部核酸记录
    ("nuccore", "txid10239[Organism:exp] AND refseq[filter]"),   # 其中的 RefSeq 记录
    ("protein", "txid10239[Organism:exp] AND refseq[filter]"),
]
for db, term in queries:
    print(f"{db:8s} {term:48s} {esearch_count(db, term):>12,d}")

print("\n== 3. efetch：GenBank 格式（rettype=gb），缓存到 U49845.1.gb")
gb = efetch("nuccore", "U49845.1", "gb")
(HERE / "U49845.1.gb").write_text(gb, encoding="utf-8")
lines = [l for l in gb.splitlines() if l.strip()]
for line in lines[:5]:
    print(line)
print(f"... 共 {len(lines)} 个非空行，最后一行：{lines[-1]!r}")

print("\n== 4. 教材图 3.1 的记录今天长什么样（FASTA，rettype=fasta）")
fa = efetch("nuccore", "NM_181652", "fasta")
header, *seqlines = fa.strip().splitlines()
print(header)
print(f"序列长度 {sum(len(s) for s in seqlines)} nt；教材图 3.1 的描述行是 >gi|332309159|ref|NM_181652.2| ...")

print("\n== 5. ENA：Portal API 计数与 Browser API 取 EMBL 格式记录")
cnt = get(ENA_PORTAL + "count", {"result": "sequence", "query": "tax_tree(10239)"}, pause=0.2)
print("ENA sequence, tax_tree(10239):", cnt.strip().splitlines()[-1])
embl = get(ENA_BROWSER + "embl/U49845.1", {"lineLimit": 12}, pause=0.2)
for line in embl.splitlines()[:7]:
    print(line)
