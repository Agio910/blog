import numpy as np

rng = np.random.default_rng(0)

def count_ffl(edges):                       # 前馈环 X→Y, Y→Z, X→Z
    out, inn = {}, {}
    for a, b in edges:
        out.setdefault(a, set()).add(b); inn.setdefault(b, set()).add(a)
    return sum(len(out.get(x, set()) & inn.get(z, set())) for x, z in edges)

def switch_randomize(edges, n_swaps):       # 教材的“交换算法”：保持每个节点的入度和出度
    edges = list(edges); es = set(edges)
    for _ in range(n_swaps):
        i, j = rng.integers(len(edges), size=2)
        (x1, x2), (y1, y2) = edges[i], edges[j]
        if len({x1, x2, y1, y2}) < 4 or (x1, y2) in es or (y1, x2) in es:
            continue                        # 会产生自回链接或多重链接，取消这次交换
        es -= {(x1, x2), (y1, y2)}; es |= {(x1, y2), (y1, x2)}
        edges[i], edges[j] = (x1, y2), (y1, x2)
    return edges

# 造一个 300 节点的有向网络：随机边 + 人为植入 30 个前馈环
n = 300
edges = {(int(a), int(b)) for a, b in rng.integers(n, size=(500, 2)) if a != b}
for _ in range(30):
    x, y, z = rng.choice(n, 3, replace=False)
    edges |= {(int(x), int(y)), (int(y), int(z)), (int(x), int(z))}
edges = list(edges)
real = count_ffl(edges)
rand = [count_ffl(switch_randomize(edges, 10 * len(edges))) for _ in range(200)]
print(f'真实网络 FFL = {real}，随机网络 {np.mean(rand):.1f} ± {np.std(rand):.1f}，'
      f'Z = {(real - np.mean(rand)) / np.std(rand):.1f}')

# 表 10.2 的数字直接算 Z
for name, nr, mu, sd in [('E. coli FFL', 40, 7, 3), ('E. coli bi-fan', 203, 47, 12),
                         ('酵母 FFL', 70, 11, 4), ('酵母 bi-fan', 1812, 300, 40)]:
    print(f'{name}: Z = ({nr} − {mu}) / {sd} = {(nr - mu) / sd:.1f}')
