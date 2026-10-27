import numpy as np
from itertools import product

# 图 10.17B 的真值表：键是 x1x2x3，值是各布尔函数的输出
states = [''.join(s) for s in product('01', repeat=3)]
f1 = {'a': '01110111', 'b': '01100111'}          # 基因 1 的两个函数 f1(1)、f2(1)，选择概率 0.6、0.4
f2 = {'a': '01101101'}                            # 基因 2 只有一个函数 f1(2)，概率 1
f3 = {'a': '00010111', 'b': '00000001'}          # 基因 3 的两个函数 f1(3)、f2(3)，各 0.5
c1, c3 = {'a': 0.6, 'b': 0.4}, {'a': 0.5, 'b': 0.5}

# PBN 由 2×1×2 = 4 个布尔网络组成，网络概率 = 各函数选择概率之积
nets = [((g1, 'a', g3), c1[g1] * c3[g3]) for g1 in 'ab' for g3 in 'ab']
A = np.zeros((8, 8))
for (g1, g2, g3), prob in nets:
    for i, s in enumerate(states):
        nxt = f1[g1][i] + f2[g2][i] + f3[g3][i]
        A[i, states.index(nxt)] += prob
print('网络概率 p1..p4 =', [p for _, p in nets])
print('转移矩阵 A（行=当前状态 000..111）:')
for s, row in zip(states, A):
    print(' ', s, np.round(row, 2).tolist())

# 每个组成网络单独运行时的吸引子
for (g1, g2, g3), _ in nets:
    step = {s: f1[g1][i] + f2[g2][i] + f3[g3][i] for i, s in enumerate(states)}
    attractors = set()
    for s in states:
        seen = []
        while s not in seen:
            seen.append(s); s = step[s]
        cycle = seen[seen.index(s):]
        attractors.add(tuple(sorted(cycle)))
    print(f'网络 ({g1},{g2},{g3}) 的吸引子:', sorted(attractors))

# 加入翻转概率 P 的随机扰动（式 10.8），整个系统变成各态遍历的马尔可夫链
P = 0.01
flip = np.array([[P ** sum(a != b for a, b in zip(s, t)) * (1 - P) ** sum(a == b for a, b in zip(s, t))
                  for t in states] for s in states])
no_flip = (1 - P) ** 3
A_p = flip.copy()
np.fill_diagonal(A_p, 0)                          # 至少有一个基因翻转：x(t+1) = x(t) ⊕ r
A_p += no_flip * A                                # 没有翻转：按布尔网络走一步
vals, vecs = np.linalg.eig(A_p.T)
pi = np.real(vecs[:, np.argmax(np.real(vals))]); pi /= pi.sum()
print('行和:', np.round(A_p.sum(1), 6).tolist())
print('稳态分布:', dict(zip(states, np.round(pi, 3).tolist())))
