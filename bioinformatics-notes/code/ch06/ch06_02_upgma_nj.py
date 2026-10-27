# -*- coding: utf-8 -*-
"""第 6 章 代码 2：从零实现 UPGMA 与邻接法（NJ），复现教材表 6.5、6.6、6.8 和图 6.7、6.12；
再用一个不满足分子钟的可加树说明 UPGMA 何时出错。只依赖 numpy。"""
import numpy as np

NAMES = ["hu", "ch", "go", "or", "gi"]
D_BOOK = np.array([[0.000, 0.015, 0.045, 0.143, 0.198],     # 教材表 6.4 上三角（三位小数）
                   [0.015, 0.000, 0.030, 0.126, 0.179],
                   [0.045, 0.030, 0.000, 0.092, 0.179],
                   [0.143, 0.126, 0.092, 0.000, 0.179],
                   [0.198, 0.179, 0.179, 0.179, 0.000]])


def upgma(D, names, verbose=True):
    """返回 Newick（超度量树，枝长 = 节点高度差）。"""
    D = D.astype(float).copy()
    nodes = [(n, 1, 0.0) for n in names]            # (newick, 簇大小, 节点高度)
    labels = list(names)
    while len(nodes) > 1:
        n = len(nodes)
        iu = np.triu_indices(n, 1)
        k = int(np.argmin(D[iu]))
        i, j = int(iu[0][k]), int(iu[1][k])
        h = D[i, j] / 2
        (ni, si, hi), (nj, sj, hj) = nodes[i], nodes[j]
        new = (f"({ni}:{h - hi:.4f},{nj}:{h - hj:.4f})", si + sj, h)
        dnew = (si * D[i] + sj * D[j]) / (si + sj)  # 按成员数加权 = 原始成员两两距离的平均
        keep = [x for x in range(n) if x not in (i, j)]
        if verbose:
            rest = ", ".join(f"{labels[x]}={dnew[x]:.4f}" for x in keep)
            print(f"  合并 {labels[i]} + {labels[j]}（d={D[i, j]:.4f}，节点高度 {h:.4f}）；新类到其余：{rest}")
        D = np.vstack([np.hstack([D[np.ix_(keep, keep)], dnew[keep][:, None]]),
                       np.hstack([dnew[keep], [0.0]])])
        nodes = [nodes[x] for x in keep] + [new]
        labels = [labels[x] for x in keep] + [labels[i] + "-" + labels[j]]
    return nodes[0][0] + ";"


def nj(D, names, verbose=True):
    """Saitou–Nei 邻接法（Studier–Keppler 的写法）。新节点放在矩阵最后，与教材表 6.8 的顺序一致；
    M 并列时取行优先的第一个。剩 3 个节点时按星形树解出三条枝长。返回无根树的 Newick。"""
    D = D.astype(float).copy()
    nodes, labels = list(names), list(names)
    step, k_new = 1, 1
    while len(nodes) > 3:
        n = len(nodes)
        r = D.sum(axis=1)                                    # 式 (6.7) 净分歧度
        M = D - (r[:, None] + r[None, :]) / (n - 2)          # 式 (6.8)
        np.fill_diagonal(M, np.inf)
        i, j = sorted(np.unravel_index(int(np.argmin(M)), M.shape))
        if verbose:
            print(f"  第 {step} 步（N={n}）  r = " + ", ".join(f"{a}:{b:.3f}" for a, b in zip(labels, r)))
            low = [f"{labels[a]}-{labels[b]}:{M[a, b]:.3f}" for a in range(n) for b in range(a)]
            print("    M(下三角) = " + "  ".join(low))
            ties = [(labels[a], labels[b]) for a in range(n) for b in range(a + 1, n)
                    if abs(M[a, b] - M[i, j]) < 1e-12]
            print(f"    最小 M = {M[i, j]:.4f}，取到最小值的对：{ties}")
        bi = D[i, j] / 2 + (r[i] - r[j]) / (2 * (n - 2))     # 式 (6.9)
        bj = D[i, j] - bi                                    # 式 (6.10)
        dnew = (D[i] + D[j] - D[i, j]) / 2                   # 式 (6.11)
        new_label = f"节点{k_new}"
        if verbose:
            rest = ", ".join(f"{labels[x]}={dnew[x]:.4f}" for x in range(n) if x not in (i, j))
            print(f"    合并 {labels[i]}({bi:.4f}) + {labels[j]}({bj:.4f}) -> {new_label}；到其余：{rest}")
        keep = [x for x in range(n) if x not in (i, j)]
        D = np.vstack([np.hstack([D[np.ix_(keep, keep)], dnew[keep][:, None]]),
                       np.hstack([dnew[keep], [0.0]])])
        nodes = [nodes[x] for x in keep] + [f"({nodes[i]}:{bi:.4f},{nodes[j]}:{bj:.4f})"]
        labels = [labels[x] for x in keep] + [new_label]
        step, k_new = step + 1, k_new + 1
    a, b, c = 0, 1, 2                                        # 三个节点连到同一中心
    la = (D[a, b] + D[a, c] - D[b, c]) / 2
    lb = (D[a, b] + D[b, c] - D[a, c]) / 2
    lc = (D[a, c] + D[b, c] - D[a, b]) / 2
    if verbose:
        print(f"  最后三点成星形：{labels[a]}={la:.4f}, {labels[b]}={lb:.4f}, {labels[c]}={lc:.4f}")
    return f"({nodes[a]}:{la:.4f},{nodes[b]}:{lb:.4f},{nodes[c]}:{lc:.4f});"


print("== UPGMA，教材表 6.4（三位小数）==")
print(upgma(D_BOOK, NAMES))
print("\n== NJ，教材表 6.4（三位小数）==")
print(nj(D_BOOK, NAMES))

diffs = np.array([[0, 1, 3, 9, 12], [1, 0, 2, 8, 11], [3, 2, 0, 6, 11], [9, 8, 6, 0, 11], [12, 11, 11, 11, 0]])
D_jc = -0.75 * np.log(1 - 4 * (diffs / 69) / 3)              # 未舍入的 JC 距离（L = 69，见代码 1）
print("\n== NJ，未舍入的 JC 距离（L=69）==")
print(nj(D_jc, NAMES, verbose=False))
print("== UPGMA，未舍入的 JC 距离（L=69）==")
print(upgma(D_jc, NAMES, verbose=False))

# ---------- UPGMA 何时出错：可加但不满足分子钟的树 ----------
# 真树 ((A:1,B:5):1,(C:1,D:5))，B、D 演化快。树上路径长度给出距离（完全可加，无噪声）
print("\n== 不满足分子钟的可加距离：真树 ((A,B),(C,D))，B、D 是长枝 ==")
D4 = np.array([[0, 6, 3, 7], [6, 0, 7, 11], [3, 7, 0, 6], [7, 11, 6, 0]], float)
print("UPGMA:", upgma(D4, list("ABCD"), verbose=False))
print("NJ   :", nj(D4, list("ABCD"), verbose=True))
