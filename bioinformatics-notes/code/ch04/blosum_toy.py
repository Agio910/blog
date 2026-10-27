"""第 4 章：按式（4.2）–（4.5）和 Henikoff 的聚类加权，在模拟的 DNA "模块"上构建 BLOSUM 式矩阵。

模拟：每个模块 10 条序列，从同一祖先沿星状树独立演化，枝长随机（0.02–0.6 次替换/位点），
用 K2P 模型（转换速率是每种颠换的 4 倍）。聚类阈值越低，越多近缘序列被并成一簇，
统计到的就越是远缘替换。
运行：python blosum_toy.py    依赖：numpy
"""
import numpy as np

rng = np.random.default_rng(0)
NT = "ACGT"
TS = np.array([2, 3, 0, 1])          # 每个碱基的转换伙伴：A<->G，C<->T


def k2p_evolve(seq, t, kappa=4.0):
    beta = 1 / (kappa + 2); alpha = kappa * beta          # 总速率 alpha + 2beta = 1
    p_tv = 0.25 - 0.25 * np.exp(-4 * beta * t)            # 每一种颠换
    p_ts = 0.25 + 0.25 * np.exp(-4 * beta * t) - 0.5 * np.exp(-2 * (alpha + beta) * t)
    u = rng.random(seq.size)
    out = seq.copy()
    ts = u < p_ts
    out[ts] = TS[seq[ts]]
    tv = (u >= p_ts) & (u < p_ts + 2 * p_tv)
    # 两种颠换：嘌呤 A/G 变成 C 或 T，嘧啶 C/T 变成 A 或 G
    pick = rng.integers(0, 2, tv.sum())
    out[tv] = np.where(np.isin(seq[tv], [0, 2]), np.array([1, 3])[pick], np.array([0, 2])[pick])
    return out


def make_blocks(n_blocks=2000, n_seq=10, width=40):
    blocks = []
    for _ in range(n_blocks):
        root = rng.integers(0, 4, width)
        blocks.append(np.array([k2p_evolve(root, rng.uniform(0.02, 0.6)) for _ in range(n_seq)]))
    return blocks


def clusters(block, thr):
    """单链聚类：与簇内任一序列一致度 >= thr 即并入（thr > 1 表示不聚类）。"""
    n = len(block)
    ident = (block[:, None, :] == block[None, :, :]).mean(2)
    lab = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if ident[i, j] >= thr and lab[i] != lab[j]:
                old = lab[j]
                lab = [lab[i] if x == old else x for x in lab]
    return np.array(lab)


def blosum(blocks, thr):
    f = np.zeros((4, 4))
    for blk in blocks:
        lab = clusters(blk, thr)
        ids, sizes = np.unique(lab, return_counts=True)
        size = dict(zip(ids, sizes))
        for a in range(len(blk)):
            for b in range(a + 1, len(blk)):
                if lab[a] == lab[b]:
                    continue                       # 同簇的序列对不计
                w = 1.0 / (size[lab[a]] * size[lab[b]])
                for x, y in zip(blk[a], blk[b]):
                    f[x, y] += w / 2; f[y, x] += w / 2   # 无序对，对称存放
    q = f / f.sum()                                # 式（4.2），q[i,j]+q[j,i] 才是无序对 (i,j) 的比例
    p = q.sum(1)                                   # 式（4.4）：p_i = q_ii + sum_{j!=i} q_ij/2
    e = np.outer(p, p)                             # 式（4.3）（对称存放后 i!=j 的 2p_ip_j 已分到两格）
    s = 2 * np.log2(q / e)                         # 式（4.5）：半比特
    H = (q * np.log2(q / e)).sum()                 # 相对熵（比特/配对）
    return np.rint(s).astype(int), H, (q.diagonal().sum())


blocks = make_blocks()
for thr, name in ((1.01, "不聚类"), (0.80, "BLOSUM80 式"), (0.62, "BLOSUM62 式"), (0.45, "BLOSUM45 式")):
    s, H, ident = blosum(blocks, thr)
    print(f"{name:12s} 计入配对的平均一致度 = {ident:.2f}，相对熵 H = {H:.3f} bit")
    print("   A  C  G  T\n" + "\n".join(NT[i] + " " + " ".join(f"{v:2d}" for v in s[i]) for i in range(4)))
