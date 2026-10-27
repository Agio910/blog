"""反向剪接（BSJ）读段识别：两端锚点拆分比对 + GT/AG 断点校正，以及 circseq_cup 的"2n"伪参考。

玩具基因组 + 精确字符串匹配代替比对器；只演示信号本身，不是可用的鉴定工具。
"""
import random
from collections import Counter

random.seed(3)
rnd = lambda n: "".join(random.choice("ACGT") for _ in range(n))
E1, E2, E3, E4 = rnd(80), rnd(60), rnd(70), rnd(70)
I1, I2, I3 = ("GT" + rnd(n) + "AG" for n in (150, 200, 120))
genome = rnd(100) + E1 + I1 + E2 + I2 + E3 + I3 + E4 + rnd(100)
e2_start, e3_end = genome.index(E2), genome.index(E3) + len(E3)
linear, circ = E1 + E2 + E3 + E4, E2 + E3          # 环：E3 的 3' 端（供体）接回 E2 的 5' 端（受体）

def sample(t, n, L=60, circular=False):
    s = t + t[:L - 1] if circular else t
    starts = [random.randrange(len(t) if circular else len(t) - L + 1) for _ in range(n)]
    return [s[i:i + L] for i in starts]

def split_align(r, A=20):
    """两端各取 A nt 锚点；返回 ('contig'|'linear'|'BSJ'|'none', (受体, 供体))。"""
    if r in genome:
        return "contig", None
    p5, p3 = genome.find(r[:A]), genome.find(r[-A:])
    if p5 < 0 or p3 < 0:
        return "none", None
    best = None
    for x in range(A, len(r) - A + 1):                 # 断点：r[:x] | r[x:]
        donor, acc = p5 + x, p3 + A - (len(r) - x)
        if r[:x] == genome[p5:donor] and r[x:] == genome[acc:acc + len(r) - x]:
            gtag = genome[donor:donor + 2] == "GT" and genome[acc - 2:acc] == "AG"
            if best is None or gtag:
                best = (acc, donor, gtag)
    if best is None:
        return "none", None
    acc, donor, gtag = best
    if not gtag:
        return "none", None
    return ("BSJ" if acc < donor else "linear"), (acc, donor)

reads = sample(circ, 300, circular=True) + sample(linear, 300)
calls = Counter()
kinds = Counter()
for r in reads:
    kind, j = split_align(r)
    kinds[kind] += 1
    if kind == "BSJ":
        calls[j] += 1
print(f"真实 BSJ：受体（E2 起点）{e2_start}，供体（E3 终点）{e3_end}")
print("600 条读段分类:", dict(kinds))
print("BSJ 调用（受体, 供体）: 支持读段数 =", dict(calls))
bsj_reads = [r for r in reads if split_align(r)[0] == "BSJ"]
two_n = circ + circ
print(f"BSJ 读段在 2n 伪参考上连续比对: {sum(r in two_n for r in bsj_reads)}/{len(bsj_reads)}；"
      f"在线性 mRNA 上: {sum(r in linear for r in bsj_reads)}/{len(bsj_reads)}")
