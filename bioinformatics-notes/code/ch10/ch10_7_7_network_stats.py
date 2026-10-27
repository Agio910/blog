import numpy as np
import networkx as nx

rng = np.random.default_rng(0)

# 1) 图 10.14：节点 i 有 3 个邻居，邻居之间分别有 3、1、0 条边
print('C_i =', [round(nx.clustering(nx.Graph([(0, 1), (0, 2), (0, 3)] + extra), 0), 3)
                for extra in ([(1, 2), (1, 3), (2, 3)], [(2, 3)], [])])

# 2) 无标度（BA 优先联结）vs 随机（ER），节点数、边数相同
n = 3000
ba = nx.barabasi_albert_graph(n, 2, seed=0)
er = nx.gnm_random_graph(n, ba.number_of_edges(), seed=0)

def mean_path(g, n_src=200):                    # 从 200 个随机源点做 BFS，估计平均最短路
    giant = g.subgraph(max(nx.connected_components(g), key=len))
    nodes = list(giant)
    d = [l for s in rng.choice(nodes, n_src, replace=False)
         for l in nx.single_source_shortest_path_length(giant, s).values() if l > 0]
    return np.mean(d)

for name, g in [('BA', ba), ('ER', er)]:
    deg = np.array([d for _, d in g.degree()])
    print(f'{name}: 平均度 {deg.mean():.2f}  最大度 {deg.max():3d}  度方差 {deg.var():5.1f}  '
          f'平均聚合系数 {nx.average_clustering(g):.4f}  平均最短路 ≈ {mean_path(g):.2f}')

# 3) 鲁棒性：随机删除 vs 定点删除度最大的节点后，最大连通片占全网比例
def giant_after(g, nodes):
    h = g.copy(); h.remove_nodes_from(nodes)
    return len(max(nx.connected_components(h), key=len)) / g.number_of_nodes()

for f in (0.05, 0.10, 0.20):
    k = int(f * n)
    row = []
    for name, g in [('BA', ba), ('ER', er)]:
        hubs = [v for v, _ in sorted(g.degree(), key=lambda t: -t[1])[:k]]
        row.append(f'{name} 随机 {giant_after(g, rng.choice(n, k, replace=False)):.2f} / 攻击 {giant_after(g, hubs):.2f}')
    print(f'删除 {f:.0%}:  ' + '   '.join(row))

# 4) 阶层网络（Ravasz–Barabási 确定性模型）：C(k) ∝ 1/k
def rb_hierarchical(levels):
    g = nx.complete_graph(5); periph = [1, 2, 3, 4]          # 5 节点全连通模块，0 号是中心
    for _ in range(levels):
        size, base, new_periph = g.number_of_nodes(), g.copy(), []
        for r in range(1, 5):                                 # 复制 4 份，各份的外围节点连回中心 0
            off = r * size
            g.add_edges_from((u + off, v + off) for u, v in base.edges())
            g.add_edges_from((p + off, 0) for p in periph)
            new_periph += [p + off for p in periph]
        periph = new_periph
    return g

def ck_slope(g, min_nodes):                                  # log C(k) 对 log k 的回归斜率
    by_k = {}
    for v, c in nx.clustering(g).items():
        by_k.setdefault(g.degree(v), []).append(c)
    ks = np.array([k for k in sorted(by_k) if len(by_k[k]) >= min_nodes and np.mean(by_k[k]) > 0])
    return np.polyfit(np.log(ks), np.log([np.mean(by_k[k]) for k in ks]), 1)[0]

h = rb_hierarchical(3)
ba8 = nx.barabasi_albert_graph(n, 8, seed=0)
print(f'RB 阶层网络 ({h.number_of_nodes()} 节点): 平均聚合系数 {nx.average_clustering(h):.2f}，C(k) 斜率 {ck_slope(h, 1):.2f}')
print(f'BA 网络 (m=8): 平均聚合系数 {nx.average_clustering(ba8):.3f}，C(k) 斜率 {ck_slope(ba8, 20):.2f}')
