"""植物靶位点规则（教材版 vs 常用版）、eTM 规则（Wu et al. 2013）、动物种子位点分型。

位置编号一律按 miRNA 5'->3' 从 1 开始；靶位点序列按 mRNA 5'->3' 书写，
miRNA 第 i 位与靶位点倒数第 i 个碱基配对。示例中的靶位点都是人工构造的。
"""
COMP = {"A": "U", "U": "A", "G": "C", "C": "G"}
WOBBLE = {("G", "U"), ("U", "G")}
revcomp = lambda s: "".join(COMP[c] for c in reversed(s))

def duplex(mir, site):
    """逐位比较：'|' 配对，':' G:U，'x' 错配。"""
    if len(mir) != len(site):
        raise ValueError("此演示只支持已对齐的等长无空位位点")
    t = site[::-1]
    return "".join("|" if COMP[a] == b else ":" if (a, b) in WOBBLE else "x" for a, b in zip(mir, t))

def plant_rules(mir, site, forbidden):
    d = duplex(mir, site)
    mm = [i + 1 for i, c in enumerate(d) if c == "x"]
    ok = (len(mm) <= 3                                   # 1. 错配不超过 3 个
          and sum(p <= 10 for p in mm) <= 1              # 2. 5' 端前 10 位至多 1 个错配
          and not set(mm) & forbidden                    # 3. 关键位置不许错配
          and "xxx" not in d)                            # 4. 无 >=3 个连续错配
    return d, ok

def etm_ok(mir, site):
    """Wu et al. 2013：凸起只在 miRNA 第 9-12 位之间、恰为 3 nt；2-8 位完全配对（允许 G:U）；凸起外错配+G:U <= 3。"""
    n = len(mir)
    if len(site) != n + 3:
        return False
    for b in range(9, 13):              # 凸起插在 miRNA 第 b 位与第 b+1 位之间
        cut = n - b                     # 凸起在靶位点（5'->3'）中占 cut..cut+2
        core = site[:cut] + site[cut + 3:]
        d = duplex(mir, core)
        if "x" not in d[1:8] and sum(c != "|" for c in d) <= 3:
            return True
    return False

def animal_sites(mir, utr):
    core = revcomp(mir[1:7])                          # 与种子（2-7 位）互补的 6 nt
    off = revcomp(mir[2:8])                           # offset 6mer：与 3-8 位互补
    out = []
    for p in range(len(utr) - 5):
        if utr[p:p + 6] == core:
            m8 = p > 0 and utr[p - 1] == COMP[mir[7]]
            a1 = p + 6 < len(utr) and utr[p + 6] == "A"
            out.append((p, {(1, 1): "8mer", (1, 0): "7mer-m8", (0, 1): "7mer-A1", (0, 0): "6mer"}[(m8, a1)]))
        elif utr[p:p + 6] == off and utr[p + 1:p + 7] != core:
            out.append((p, "offset 6mer"))
    return out

if __name__ == "__main__":
    mir156 = "UGACAGAAGAGAGUGAGCAC"                   # ath-miR156a 成熟序列（20 nt）
    perfect = revcomp(mir156)
    mutate = lambda s, i, b: s[:len(s) - i] + b + s[len(s) - i + 1:]   # 改动与 miRNA 第 i 位相对的碱基
    cases = {"完全互补": perfect,
             "第1位错配": mutate(perfect, 1, "C"),
             "第10位错配": mutate(perfect, 10, "G"),
             "第6位G:U+第20位错配": mutate(mutate(perfect, 6, "U"), 20, "U")}
    print("miR156:", mir156)
    for name, site in cases.items():
        d, ok_tb = plant_rules(mir156, site, {1, 11, 12})
        _, ok_cm = plant_rules(mir156, site, {10, 11})
        print(f"{name:<14} {site}  {d}  教材规则 {'通过' if ok_tb else '不通过'}；第10-11位规则 {'通过' if ok_cm else '不通过'}")
    mimic = perfect[:10] + "CUA" + perfect[10:]      # 在 miRNA 第 10/11 位对面插入 3 nt 凸起
    print(f"eTM 构造位点 {mimic}：符合 Wu 2013 规则 = {etm_ok(mir156, mimic)}；"
          f"同处插 2 nt = {etm_ok(mir156, perfect[:10] + 'CU' + perfect[10:])}；"
          f"凸起移到第 5/6 位之间 = {etm_ok(mir156, perfect[:15] + 'CUA' + perfect[15:])}")

    let7a = "UGAGGUAGUAGGUUGUAUAGUU"                 # let-7a 成熟序列
    utr = "AAACUACCUCAUUU" + "GGCUACCUCGG" + "UUUACCUCAGG" + "GGUACCUCGG" + "AACUACCUGG"
    print("let-7a 种子(2-7):", let7a[1:7], " 3'UTR 位点:", animal_sites(let7a, utr))
