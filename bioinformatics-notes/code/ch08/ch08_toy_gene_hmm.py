# -*- coding: utf-8 -*-
"""第 8 章 §7.2：4 状态玩具基因 HMM（对应图 8.5 的思路）：模拟、Viterbi、前向-后向。"""
import numpy as np
from scipy.special import logsumexp

rng = np.random.default_rng(0)
S = ["N", "C1", "C2", "C3"]                      # 基因间区 + 密码子三个位置（3 周期）
A = np.array([[.995, .005, 0, 0],                # N  -> N / C1
              [0, 0, 1, 0],                      # C1 -> C2
              [0, 0, 0, 1],                      # C2 -> C3
              [.003, .997, 0, 0]])               # C3 -> N / C1：每个密码子后以 0.003 的概率结束
E = np.array([[.30, .20, .20, .30],              # 发射概率，列为 A C G T；基因间区偏 AT
              [.25, .20, .35, .20],              # 密码子第 1 位偏 G
              [.30, .25, .15, .30],
              [.15, .35, .35, .15]])             # 第 3 位偏 G/C
pi = np.array([1., 0, 0, 0])
with np.errstate(divide="ignore"):
    lpi, lA, lE = np.log(pi), np.log(A), np.log(E)

def simulate(L):
    z = [0]
    x = [rng.choice(4, p=E[0])]
    for _ in range(L - 1):
        z.append(rng.choice(4, p=A[z[-1]]))
        x.append(rng.choice(4, p=E[z[-1]]))
    return np.array(x), np.array(z)

def viterbi(x):
    L, K = len(x), len(lpi)
    V = np.empty((L, K)); B = np.zeros((L, K), int)
    V[0] = lpi + lE[:, x[0]]
    for t in range(1, L):
        sc = V[t - 1][:, None] + lA               # sc[j, k]: 由 j 转到 k
        B[t] = sc.argmax(0)
        V[t] = sc.max(0) + lE[:, x[t]]
    path = [int(V[-1].argmax())]
    for t in range(L - 1, 0, -1):
        path.append(B[t, path[-1]])
    return np.array(path[::-1]), V[-1].max()

def forward_backward(x):
    L, K = len(x), len(lpi)
    F = np.empty((L, K)); Bk = np.zeros((L, K))
    F[0] = lpi + lE[:, x[0]]
    for t in range(1, L):
        F[t] = logsumexp(F[t - 1][:, None] + lA, axis=0) + lE[:, x[t]]
    for t in range(L - 2, -1, -1):
        Bk[t] = logsumexp(lA + (lE[:, x[t + 1]] + Bk[t + 1])[None, :], axis=1)
    logP = logsumexp(F[-1])
    return np.exp(F + Bk - logP), logP           # 每个位置各状态的后验概率

def sn_sp(truth, pred):
    tp = (truth & pred).sum()
    return tp / truth.sum(), tp / pred.sum()

x, z = simulate(20000)
zv, best = viterbi(x)
post, logP = forward_backward(x)
coding = z > 0
print("编码碱基占比 %.3f" % coding.mean())
print("log P(x) = %.1f，最优路径 log P(x, pi*) = %.1f，最优路径占全部概率的 %.2e"
      % (logP, best, np.exp(best - logP)))
print("Viterbi     nt Sn = %.3f  Sp(precision) = %.3f" % sn_sp(coding, zv > 0))
pc = post[:, 1:].sum(1)
print("后验 > 0.5  nt Sn = %.3f  Sp(precision) = %.3f" % sn_sp(coding, pc > 0.5))
both = coding & (zv > 0)
print("两者都判为编码的位置中，密码子位置（读码框）也一致的比例 %.3f" % (zv[both] == z[both]).mean())
print("后验概率列之和为 1:", np.allclose(post.sum(1), 1))
