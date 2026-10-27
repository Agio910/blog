"""第 3 章：把 PROSITE 模式（pattern）翻译成正则表达式，扫描蛋白序列，并估计随机命中的期望次数。

运行：python ch03_prosite.py
输入：本目录的 U49845.1.gb（取 AXL2 蛋白的 /translation；AXL2 是质膜糖蛋白）。只用标准库。
语法依据 PROSITE 用户手册：x 为任意残基；[..] 为可选集合；{..} 为排除集合；元素间用 - 分隔；
e(n) 或 e(n,m) 表示重复；< 与 > 表示锚定 N 端、C 端；句点结束模式。
"""
from pathlib import Path
import random
import re
from collections import Counter

HERE = Path(__file__).resolve().parent


def prosite_to_regex(pat):
    pat = pat.strip().rstrip(".")
    out = []
    for el in pat.split("-"):
        m = re.fullmatch(r"(<?)(.+?)(?:\((\d+)(?:,(\d+))?\))?(>?)", el)
        nterm, core, n1, n2, cterm = m.groups()
        if core == "x":
            rx = "."
        elif core.startswith("["):
            rx = "[" + core[1:-1].replace(">", "") + "]"   # 罕见的 [G>] 写法此处不处理
        elif core.startswith("{"):
            rx = "[^" + core[1:-1] + "]"
        else:
            rx = core
        if n1:
            rx += "{%s}" % (n1 if not n2 else f"{n1},{n2}")
        out.append(("^" if nterm else "") + rx + ("$" if cterm else ""))
    return "".join(out)


def scan(pattern, seq):
    rx = re.compile("(?=(%s))" % prosite_to_regex(pattern))     # 零宽前瞻：允许重叠命中
    return [(m.start() + 1, m.group(1)) for m in rx.finditer(seq)]


def expected_hits(pattern, seq):
    """按该序列自身的氨基酸组成、假设各位点独立，估计随机序列中每个起点命中的概率与期望次数。"""
    comp = Counter(seq)
    f = {a: c / len(seq) for a, c in comp.items()}
    p, k = 1.0, 0
    for el in pattern.rstrip(".").split("-"):
        if el == "x":
            q = 1.0
        elif el.startswith("["):
            q = sum(f.get(a, 0) for a in el[1:-1])
        elif el.startswith("{"):
            q = 1 - sum(f.get(a, 0) for a in el[1:-1])
        else:
            q = f.get(el, 0)
        p *= q
        k += 1
    return p, p * (len(seq) - k + 1)


text = (HERE / "U49845.1.gb").read_text(encoding="utf-8")
m = re.search(r'/gene="AXL2".*?/translation="([^"]+)"', text, re.S)
axl2 = re.sub(r"\s+", "", m.group(1))

for pat in ["N-{P}-[ST]-{P}.", "C-x(2,4)-C-x(3)-[LIVMFYWC]-x(8)-H-x(3,5)-H."]:   # PS00001、PS00028
    print("PROSITE 模式:", pat, " ->  正则:", prosite_to_regex(pat))

pat = "N-{P}-[ST]-{P}."           # PS00001，N-糖基化位点，条目带 /SKIP-FLAG=TRUE
hits = scan(pat, axl2)
p, e = expected_hits(pat, axl2)
print(f"\nAXL2 蛋白长 {len(axl2)} aa；PS00001 命中 {len(hits)} 处，前 8 处：{hits[:8]}")
print(f"按 AXL2 自身组成估计：每个起点随机命中概率 {p:.4f}，期望 {e:.1f} 处")

shuffled = list(axl2)
random.seed(1)
counts = []
for _ in range(200):
    random.shuffle(shuffled)
    counts.append(len(scan(pat, "".join(shuffled))))
counts.sort()
print(f"把 AXL2 打乱 200 次：命中数中位数 {counts[100]}，范围 {counts[0]}–{counts[-1]}；"
      f"打乱后命中数 >= {len(hits)} 的比例 {sum(c >= len(hits) for c in counts) / len(counts):.3f}")
