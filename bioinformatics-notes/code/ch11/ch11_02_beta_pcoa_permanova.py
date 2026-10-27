import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
D = 60                                                    # 物种数
p0 = rng.lognormal(0, 1.5, D); p0 /= p0.sum()             # 两组共同的平均组成（原假设为真）

def sample(n, theta, depth=5000):                         # Dirichlet-多项：theta 越小，样本间离散越大
    P = rng.dirichlet(theta * p0, size=n)
    return np.array([rng.multinomial(depth, p) for p in P])

def bray_curtis(X):
    P = X / X.sum(1, keepdims=True)
    return np.abs(P[:, None, :] - P[None, :, :]).sum(-1) / 2   # 相对丰度时 BC = 1 − Σmin = ½Σ|差|

def aitchison(X):                                         # CLR 后的欧氏距离（伪计数 0.5）
    L = np.log(X + 0.5)
    C = L - L.mean(1, keepdims=True)
    return np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))

def pcoa(Dm):                                             # 经典多维尺度：双中心化 + 特征分解
    n = len(Dm)
    J = np.eye(n) - 1 / n
    B = -0.5 * J @ (Dm ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]

def permanova(Dm, g, perms=499):                          # 伪 F = [SS_A/(a−1)] / [SS_W/(N−a)]，置换得 p
    N, a = len(g), len(set(g))
    D2 = Dm ** 2
    sst = D2[np.triu_indices(N, 1)].sum() / N
    def ssw(gr):
        return sum(D2[np.ix_(gr == k, gr == k)][np.triu_indices((gr == k).sum(), 1)].sum() / (gr == k).sum()
                   for k in set(gr))
    def F(gr):
        w = ssw(gr)
        return ((sst - w) / (a - 1)) / (w / (N - a))
    f0 = F(g)
    fp = np.array([F(rng.permutation(g)) for _ in range(perms)])
    return f0, (1 + (fp >= f0).sum()) / (perms + 1)

def dispersion_test(Dm, g):                               # betadisper 思路：到组中心的距离做方差分析
    w, V = pcoa(Dm)
    pos = w > 1e-10
    Y = V[:, pos] * np.sqrt(w[pos])
    dist = np.empty(len(g))
    for k in set(g):
        m = g == k
        dist[m] = np.linalg.norm(Y[m] - Y[m].mean(0), axis=1)
    return stats.f_oneway(*[dist[g == k] for k in set(g)]).pvalue

# 1) 一次实例：Bray–Curtis 的 PCoA 会出现负特征值（BC 不是欧氏距离）
X = np.vstack([sample(20, 300), sample(20, 300)])
w, _ = pcoa(bray_curtis(X))
print(f'BC 的 PCoA：最大特征值 {w[0]:.3f}，最小特征值 {w[-1]:.4f}（为负），负特征值个数 {(w < -1e-10).sum()}')
w2, _ = pcoa(aitchison(X))
print(f'Aitchison 的 PCoA：最小特征值 {w2[-1]:.1e}（欧氏距离不会出现明显负值）')

# 2) 欧氏空间里两组中心相同、只有离散程度不同（Anderson & Walsh 2013 的经典情形）
def euclid(X):
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))

def type1(make, n1, n2, reps=200):
    hits_perm = hits_disp = 0
    for _ in range(reps):
        X1, X2 = make(n1, n2)
        Dm = make.dist(np.vstack([X1, X2]))
        g = np.array([0] * n1 + [1] * n2)
        hits_perm += permanova(Dm, g, perms=199)[1] <= 0.05
        hits_disp += dispersion_test(Dm, g) <= 0.05
    return hits_perm / reps, hits_disp / reps

def gauss(n1, n2):                                        # 10 维正态，均值都为 0；组 2 的标准差是组 1 的 1.5 倍
    return rng.normal(0, 1, (n1, 10)), rng.normal(0, 1.5, (n2, 10))
gauss.dist = euclid
for n1, n2 in [(20, 20), (40, 10), (10, 40)]:
    tp, td = type1(gauss, n1, n2)
    print(f'正态+欧氏  n=({n1},{n2})，组2 更离散：PERMANOVA 判显著 {tp:4.0%}   离散度检验判显著 {td:4.0%}')

# 3) 群落数据：平均组成相同，但组 2 的 Dirichlet 离散更大（theta 300 vs 100），用 Bray–Curtis
def dirich(n1, n2):
    return sample(n1, 300), sample(n2, 100)
dirich.dist = bray_curtis
tp, td = type1(dirich, 20, 20, reps=100)
print(f'群落+BC    n=(20,20)，组2 更离散：PERMANOVA 判显著 {tp:4.0%}   离散度检验判显著 {td:4.0%}')
