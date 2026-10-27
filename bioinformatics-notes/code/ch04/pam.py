"""第 4 章：由表 4.2 的替换计数重建 PAM1 → PAM250 → 对数概率矩阵，并与表 4.3 对比。

运行：python pam.py    依赖：numpy
相对突变率（Dayhoff 1978 表 21）和氨基酸频率（表 22）取自 PAML 软件包的 dat/dayhoff.dat。
"""
import numpy as np

AA = "ARNDCQEGHILKMFPSTWYV"
# 表 4.2（= Dayhoff 1978 图 80）：可接受点突变次数 ×10，下三角，行列顺序同 AA
T42 = """30
109 17
154 0 532
33 10 0 0
93 120 50 76 0
266 0 94 831 0 422
579 10 156 162 10 30 112
21 103 226 43 10 243 23 10
66 30 36 13 17 8 35 0 3
95 17 37 0 0 75 15 17 40 253
57 477 322 85 0 147 104 60 23 43 39
29 17 0 0 0 20 7 7 0 57 207 90
20 7 7 0 0 0 0 17 20 90 167 0 17
345 67 27 10 10 93 40 49 50 7 43 43 4 7
772 137 432 98 117 47 86 450 26 20 32 168 20 40 269
590 20 169 57 10 37 31 50 14 129 52 200 28 10 73 696
0 27 3 0 0 0 0 0 3 0 13 0 0 10 0 17 0
20 3 36 0 30 0 10 0 40 13 23 10 0 260 0 22 23 6
365 20 13 17 33 27 37 97 30 661 303 17 77 10 50 43 186 0 17"""
MUT = dict(A=100, R=65, N=134, D=106, C=20, Q=93, E=102, G=49, H=66, I=96, L=40, K=56,
           M=94, F=41, P=56, S=120, T=97, W=18, Y=41, V=74)
FREQ = dict(A=.087, R=.041, N=.040, D=.047, C=.033, Q=.038, E=.050, G=.089, H=.034, I=.037,
            L=.085, K=.081, M=.015, F=.040, P=.051, S=.070, T=.058, W=.010, Y=.030, V=.065)
# 表 4.3：PAM250 对数概率（10·log10），下三角，顺序 CSTPAGNDEQHRKMILVFYW
ORDER = "CSTPAGNDEQHRKMILVFYW"
T43 = """12
0 2
-2 1 3
-3 1 0 6
-2 1 1 1 2
-3 1 0 -1 1 5
-4 1 0 -1 0 0 2
-5 0 0 -1 0 1 2 4
-5 0 0 -1 0 0 1 3 4
-5 -1 -1 0 0 -1 1 2 2 4
-3 -1 -1 0 -1 -2 2 1 1 3 6
-4 0 -1 0 -2 -3 0 -1 -1 1 2 6
-5 0 0 -1 -1 -2 1 0 0 1 0 3 5
-5 -2 -1 -2 -1 -3 -2 -3 -2 -1 -2 0 0 6
-2 -1 0 -2 -1 -3 -2 -2 -2 -2 -2 -2 -2 2 5
-6 -3 -2 -3 -2 -4 -3 -4 -3 -2 -2 -3 -3 4 2 6
-2 -1 0 -1 0 -1 -2 -2 -2 -2 -2 -2 -2 2 4 2 4
-4 -3 -3 -5 -4 -5 -4 -6 -5 -5 -2 -4 -5 0 1 2 -1 9
0 -3 -3 -5 -3 -5 -2 -4 -4 -4 0 -4 -4 -2 -1 -1 -2 7 10
-8 -2 -5 -6 -6 -7 -4 -7 -7 -5 -3 2 -3 -4 -5 -2 -6 0 0 17"""


def lower_to_full(text, n, diag=False):
    X = np.zeros((n, n))
    for i, line in enumerate(text.splitlines(), start=0 if diag else 1):
        for j, v in enumerate(line.split()):
            X[i, j] = X[j, i] = float(v)
    return X


A = lower_to_full(T42, 20)
print("表 4.2 合计 = %.0f（注：次数已乘 10，对应约 1572 次替换）" % np.tril(A, -1).sum())

m = np.array([MUT[a] for a in AA], float)
f = np.array([FREQ[a] for a in AA])
f = f / f.sum()

# 1 PAM：M[i, j] = 氨基酸 j 在 1 PAM 内被 i 替换的概率（Dayhoff 的"列"约定）
lam = 0.01 / (f @ m)                       # 使期望变化比例 = 1%
M = lam * m[None, :] * A / A.sum(0)[None, :]
np.fill_diagonal(M, 1 - lam * m)
print("每列和为 1:", np.allclose(M.sum(0), 1), "；期望变化比例 = %.4f" % (1 - f @ np.diag(M)))
print("1 PAM 内保持不变的概率：W %.4f，C %.4f，N %.4f，S %.4f"
      % tuple(M[AA.index(c), AA.index(c)] for c in "WCNS"))

# 多重替换使"观测差异"跟不上"PAM 距离"
for n in (1, 10, 50, 100, 150, 250, 400):
    Mn = np.linalg.matrix_power(M, n)
    print("PAM%-4d 观测一致度 = %.1f%%" % (n, 100 * f @ np.diag(Mn)))

# PAM250 对数概率：10·log10( M250[i,j] / f[i] )，再对称化、取整
M250 = np.linalg.matrix_power(M, 250)
R = 10 * np.log10(M250 / f[:, None])
S = np.rint((R + R.T) / 2).astype(int)

book = lower_to_full(T43, 20, diag=True).astype(int)
perm = [AA.index(c) for c in ORDER]
S2 = S[np.ix_(perm, perm)]
d = (S2 - book)[np.tril_indices(20)]
print("与表 4.3 的 210 个元素比较：完全相同 %d 个，差 1 的 %d 个，最大差 %d"
      % ((d == 0).sum(), (abs(d) == 1).sum(), abs(d).max()))
print("例：W-W 重建 %d / 表 %d；C-C %d / %d；F-Y %d / %d；L-I %d / %d" % (
    S[17, 17], book[19, 19], S[4, 4], book[0, 0], S[13, 18], book[18, 17], S[10, 9], book[15, 14]))
