import numpy as np

rng = np.random.default_rng(0)
n, m = 40, 30                                               # 40 个样点、30 个物种
x = np.sort(rng.uniform(0, 10, n))                          # 一条环境梯度（如深度）
z = rng.normal(0, 1, n)                                     # 一个与群落无关的环境变量（噪声）
u, t, h = rng.uniform(-1, 11, m), rng.uniform(0.8, 1.5, m), rng.lognormal(3, 0.5, m)
Y = rng.poisson(h * np.exp(-(x[:, None] - u) ** 2 / (2 * t ** 2)))   # 单峰（高斯）响应曲线
Y = Y[:, Y.sum(0) > 0]

def ca_matrix(Y):                                           # 对应分析的标准化残差矩阵（卡方距离）
    P = Y / Y.sum()
    r, c = P.sum(1), P.sum(0)
    return (P - np.outer(r, c)) / np.sqrt(np.outer(r, c)), r

def weighted_fit(Q, X, r):                                  # CCA：以样点权重 r 做多元回归，再对拟合值做 SVD
    Xc = X - r @ X                                          # 加权中心化
    Xw = np.sqrt(r)[:, None] * Xc
    H = Xw @ np.linalg.pinv(Xw)
    return H @ Q

Q, r = ca_matrix(Y)
total = (Q ** 2).sum()                                      # 总惯量
U, s, Vt = np.linalg.svd(Q, full_matrices=False)
site = U[:, :2] * s[:2] / np.sqrt(r)[:, None]               # CA 样点得分
quad = np.polyfit(site[:, 0], site[:, 1], 2)
r2 = 1 - np.var(site[:, 1] - np.polyval(quad, site[:, 0])) / np.var(site[:, 1])
print(f'CA：第 1 轴与梯度 x 的相关 {abs(np.corrcoef(site[:, 0], x)[0, 1]):.3f}；第 2 轴可由第 1 轴的二次函数解释 R² = {r2:.2f}（弓形效应）')

def cca_test(X, perms=999):
    fit = weighted_fit(Q, X, r)
    inert = (fit ** 2).sum()
    null = np.array([(weighted_fit(Q, X[rng.permutation(n)], r) ** 2).sum() for _ in range(perms)])
    return inert / total, (1 + (null >= inert).sum()) / (perms + 1), fit

for name, X in [('x', x[:, None]), ('z', z[:, None]), ('x+z', np.c_[x, z]), ('x+x²', np.c_[x, x ** 2])]:
    frac, p, fit = cca_test(X)
    U1 = np.linalg.svd(fit, full_matrices=False)[0][:, 0]
    print(f'CCA 约束变量 {name:<3}：解释惯量 {frac:5.1%}，置换 p = {p:.3f}，第一约束轴与 x 的相关 {abs(np.corrcoef(U1 / np.sqrt(r), x)[0, 1]):.3f}')

# RDA：Hellinger 变换后做普通（不加权）多元回归 + PCA
Hl = np.sqrt(Y / Y.sum(1, keepdims=True)); Hl -= Hl.mean(0)
def rda_frac(X):
    Xc = X - X.mean(0)
    return ((Xc @ np.linalg.pinv(Xc) @ Hl) ** 2).sum() / (Hl ** 2).sum()
for name, X in [('x', x[:, None]), ('z', z[:, None])]:
    f0 = rda_frac(X)
    p = (1 + sum(rda_frac(X[rng.permutation(n)]) >= f0 for _ in range(999))) / 1000
    print(f'RDA 约束变量 {name}：解释方差 {f0:5.1%}，置换 p = {p:.3f}')
