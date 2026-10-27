"""玩具版 McCaskill：配分函数、碱基配对概率与可及性 ED（对应教材式 9.10）。

能量模型是教学用的"每个碱基对独立计能"（GC -3、AU -2、GU -1 kcal/mol），
没有堆积能和环罚分，不是 Turner 最近邻模型；递推结构与 Nussinov 相同，
只是把 max/+1 换成 求和/乘玻尔兹曼因子。
"""
import math
from functools import lru_cache

RT = 0.0019872 * 310.15          # kcal/mol，37 摄氏度
E = {("G", "C"): -3.0, ("C", "G"): -3.0, ("A", "U"): -2.0, ("U", "A"): -2.0,
     ("G", "U"): -1.0, ("U", "G"): -1.0}

def Z_dp(seq, theta=3, forbid=frozenset()):
    """forbid 中的位置不允许配对（用于计算"区间不配对"的受限配分函数）。"""
    @lru_cache(None)
    def Z(i, j):
        if j - i <= theta:
            return 1.0
        z = Z(i, j - 1)
        for k in range(i, j - theta):
            if (seq[k], seq[j]) in E and k not in forbid and j not in forbid:
                z += Z(i, k - 1) * math.exp(-E[(seq[k], seq[j])] / RT) * Z(k + 1, j - 1)
        return z
    return Z(0, len(seq) - 1)

def enumerate_structs(seq, theta=3):
    """穷举全部结构（与 ch09_nussinov.py 同一文法，每个结构只生成一次）。"""
    @lru_cache(None)
    def S(i, j):
        if j - i <= theta:
            return [()]
        out = list(S(i, j - 1))
        for k in range(i, j - theta):
            if (seq[k], seq[j]) in E:
                out += [a + ((k, j),) + b for a in S(i, k - 1) for b in S(k + 1, j - 1)]
        return out
    return S(0, len(seq) - 1)

def energy(seq, s):
    return sum(E[(seq[k], seq[j])] for k, j in s)

if __name__ == "__main__":
    seq = "GGGACUUCGGUCCCAAUAUA"
    allS = enumerate_structs(seq)
    w = [math.exp(-energy(seq, s) / RT) for s in allS]
    Z_enum = sum(w)
    print(f"序列 {seq}（{len(seq)} nt），结构总数 {len(allS)}")
    print(f"Z（递推）= {Z_dp(seq):.6e}   Z（穷举）= {Z_enum:.6e}")

    mfe_s = min(allS, key=lambda s: energy(seq, s))
    db = ["."] * len(seq)
    for k, j in mfe_s:
        db[k], db[j] = "(", ")"
    e_mfe = energy(seq, mfe_s)
    print(f"MFE 结构 {''.join(db)}  E = {e_mfe:.1f}，"
          f"该结构在系综中的概率 {math.exp(-e_mfe / RT) / Z_enum:.3f}")
    print(f"系综自由能 -RT ln Z = {-RT * math.log(Z_enum):.2f}（低于 MFE）")

    P = {}
    for s, ws in zip(allS, w):
        for p in s:
            P[p] = P.get(p, 0.0) + ws / Z_enum
    top = sorted(P.items(), key=lambda x: -x[1])[:6]
    print("配对概率最高的 6 个碱基对:",
          ", ".join(f"{seq[k]}{k + 1}-{seq[j]}{j + 1}:{p:.3f}" for (k, j), p in top))

    Z = Z_dp(seq)
    for i, k in [(1, 4), (6, 9), (15, 18)]:          # 0 起编号的闭区间
        Zu = Z_dp(seq, forbid=frozenset(range(i, k + 1)))
        print(f"区间 {i + 1}-{k + 1} 保持单链：Z_unpaired/Z = {Zu / Z:.3e}，ED = {-RT * math.log(Zu / Z):.2f} kcal/mol")
