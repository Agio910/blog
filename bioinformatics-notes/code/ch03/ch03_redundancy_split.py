"""第 3 章：数据库冗余、CD-HIT 式贪心聚类，以及按聚类划分训练/测试集（numpy，离线运行）。

运行：python ch03_redundancy_split.py

模拟设定（只用替换突变、不考虑插入缺失，所以"一致性"就是逐位相同的比例，不需要比对）：
  20 个蛋白家族，每个家族有一条祖先序列（长 150）；
  每个家族分出 5 个亚家族，亚家族祖先相对家族祖先有 75% 的位点被替换（亚家族之间一致性约 0.09）；
  每个亚家族 6 条成员序列，相对亚家族祖先再替换 5%（同一亚家族成员之间约 0.90）。
任务：给一条新序列预测它属于哪个家族；"模型"是按一致性找最近邻（1-NN）。
对照两种划分（随机划分、按 50% 一致性聚类后整簇划分），并把同一个模型拿到全新生成的亚家族上
（相当于部署时遇到的新数据）再测一次。重复 20 次取平均。
"""
import numpy as np

rng = np.random.default_rng(2026)
A, L = 20, 150                 # 字母表大小、序列长度
N_FAM, N_SUB, N_MEM = 20, 5, 6
P_SUB, P_MEM = 0.75, 0.05


def mutate(seq, p):
    s = seq.copy()
    hit = rng.random(len(s)) < p
    # 替换成另一个残基（不会"替换成自己"）
    s[hit] = (s[hit] + rng.integers(1, A, hit.sum())) % A
    return s


def make_subfamily(fam_anc):
    sub_anc = mutate(fam_anc, P_SUB)
    return [mutate(sub_anc, P_MEM) for _ in range(N_MEM)]


fam_anc = [rng.integers(0, A, L) for _ in range(N_FAM)]
X, fam, sub = [], [], []
for f in range(N_FAM):
    for s in range(N_SUB):
        for m in make_subfamily(fam_anc[f]):
            X.append(m); fam.append(f); sub.append(f * N_SUB + s)
X, fam, sub = np.array(X), np.array(fam), np.array(sub)
n = len(X)
ident = (X[:, None, :] == X[None, :, :]).mean(axis=2)          # n×n 一致性矩阵
iu = np.triu_indices(n, 1)
same_sub = (sub[:, None] == sub[None, :])[iu]
same_fam = (fam[:, None] == fam[None, :])[iu]
print(f"== 0. 数据：{n} 条序列。平均一致性：同亚家族 {ident[iu][same_sub].mean():.3f}，"
      f"同家族不同亚家族 {ident[iu][same_fam & ~same_sub].mean():.3f}，不同家族 {ident[iu][~same_fam].mean():.3f}")


# ---------------------------------------------------------------- CD-HIT 式贪心增量聚类 + 短词过滤
def kmers(s, k):
    return {tuple(s[i:i + k]) for i in range(len(s) - k + 1)}


def greedy_cluster(X, t, k):
    """按长度从长到短（这里等长，按输入顺序）依次处理；与已有代表序列比，一致性 >= t 就并入第一个命中的簇。
    短词过滤：只有替换时，m 个错配最多破坏 k*m 个 k-mer，所以一致性 >= t 的两条等长序列至少共享
    (L-k+1) - k*floor((1-t)*L) 个 k-mer；共享数达不到这个下界就不必再算一致性。"""
    reps, label = [], np.empty(len(X), int)
    bound = (L - k + 1) - k * int(np.floor((1 - t) * L + 1e-9))   # 加 1e-9 防止 (1-0.9)*150 被算成 14.999…
    km = [kmers(s, k) for s in X]
    n_checked = n_skipped = 0
    for i in range(len(X)):
        for c, r in enumerate(reps):
            if bound > 0 and len(km[i] & km[r]) < bound:
                n_skipped += 1
                continue
            n_checked += 1
            if (X[i] == X[r]).mean() >= t:
                label[i] = c
                break
        else:
            reps.append(i)
            label[i] = len(reps) - 1
    return label, reps, n_checked, n_skipped, bound


print("\n== 1. 贪心聚类（类似 CD-HIT 的默认快速模式）")
for t, k in [(0.9, 5), (0.5, 2)]:
    lab, reps, nc, ns, b = greedy_cluster(X, t, k)
    pure = np.mean([len(set(sub[lab == c])) == 1 for c in range(len(reps))])
    print(f"阈值 {t:.0%}、k={k}：{len(reps)} 个簇（真实亚家族 {N_FAM * N_SUB} 个），簇内只含一个亚家族的比例 {pure:.2f}；"
          f"短词下界 {b} 个共享 k-mer，算了 {nc} 次一致性、过滤掉 {ns} 次")
labels50 = greedy_cluster(X, 0.5, 2)[0]


# ---------------------------------------------------------------- 两种划分 + 部署数据
def one_nn(train_idx, Xq):
    Xt = X[train_idx]
    sim = (Xq[:, None, :] == Xt[None, :, :]).mean(axis=2)
    j = sim.argmax(axis=1)
    return fam[train_idx][j], sim.max(axis=1)


def random_split(test_frac=0.2):
    perm = rng.permutation(n)
    k = int(test_frac * n)
    return perm[k:], perm[:k]


def cluster_split(labels, test_frac=0.2):
    test = []
    for c in rng.permutation(np.unique(labels)):       # 整簇放进测试集，直到约 20%
        if len(test) >= test_frac * n:
            break
        test.extend(np.where(labels == c)[0])
    test = np.array(test)
    return np.setdiff1d(np.arange(n), test), test


def fresh_subfamilies():                               # 部署场景：每个已知家族各冒出一个新亚家族
    Xn, yn = [], []
    for f in range(N_FAM):
        for m in make_subfamily(fam_anc[f]):
            Xn.append(m); yn.append(f)
    return np.array(Xn), np.array(yn)


R = 20
res = {"随机划分": [], "按 50% 聚类整簇划分": []}
for r in range(R):
    new_X, new_fam = fresh_subfamilies()
    for name, (tr, te) in [("随机划分", random_split()), ("按 50% 聚类整簇划分", cluster_split(labels50))]:
        pred, best = one_nn(tr, X[te])
        pred_new, best_new = one_nn(tr, new_X)          # 同一个模型（同一训练集）拿去预测新亚家族
        res[name].append((np.median(best), np.mean(pred == fam[te]), np.median(best_new), np.mean(pred_new == new_fam)))

print(f"\n== 2. 1-NN 预测家族，重复 {R} 次（训练 480 条 / 测试 120 条；部署数据为每次新生成的 20 个亚家族共 120 条）")
print("划分方式              测试集与训练集最高一致性   测试集准确率      同一模型在新亚家族上的准确率")
for name, rows in res.items():
    a = np.array(rows)
    print(f"{name:20s}  {a[:, 0].mean():.3f}                    {a[:, 1].mean():.3f} ± {a[:, 1].std():.3f}     "
          f"{a[:, 3].mean():.3f} ± {a[:, 3].std():.3f}（新数据最高一致性 {a[:, 2].mean():.3f}）")


# ---------------------------------------------------------------- Mash：k-mer Jaccard 估计核苷酸分歧度
def mash_distance(a, b, k):
    ka = {a[i:i + k] for i in range(len(a) - k + 1)}
    kb = {b[i:i + k] for i in range(len(b) - k + 1)}
    j = len(ka & kb) / len(ka | kb)
    return (-np.log(2 * j / (1 + j)) / k if j > 0 else np.inf), j


print("\n== 3. Mash 距离 D = -(1/k) ln[2J/(1+J)] 与真实替换率（20 kb 随机基因组，精确 Jaccard，不抽样）")
g = "".join(rng.choice(list("ACGT"), 20000))
for d in [0.01, 0.05, 0.10, 0.20]:
    arr = np.array(list(g))
    hit = rng.random(len(arr)) < d
    arr[hit] = [rng.choice([c for c in "ACGT" if c != x]) for x in arr[hit]]
    h = "".join(arr)
    row = []
    for k in (16, 21):
        D, j = mash_distance(g, h, k)
        row.append(f"k={k}: J={j:.4f} D={D:.4f}")
    print(f"真实替换率 {hit.mean():.4f}  |  " + "  |  ".join(row))
