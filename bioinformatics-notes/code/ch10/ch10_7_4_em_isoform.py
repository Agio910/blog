import numpy as np
from collections import Counter

rng = np.random.default_rng(0)
# 三个异构体共享外显子：T1=E1E2E3，T2=E1E3（跳过 E2），T3=E3E4
exons = {'E1': 500, 'E2': 500, 'E3': 500, 'E4': 300}
tx = {'T1': ['E1', 'E2', 'E3'], 'T2': ['E1', 'E3'], 'T3': ['E3', 'E4']}
names = list(tx)
L = np.array([sum(exons[e] for e in tx[t]) for t in names])
R = 100                                            # 读长（单端，片段=读序）
eff = L - R + 1                                    # 有效长度：读序起点可落的位置数
tau = np.array([0.5, 0.3, 0.2])                    # 真实转录本分子比例（TPM/1e6）

def exon_path(t, start):                           # 读序覆盖了转录本上的哪几段外显子
    pos, out = 0, []
    for e in tx[t]:
        if pos < start + R and start < pos + exons[e]:
            out.append(e)
        pos += exons[e]
    return tuple(out)

def compatible(path):                              # 哪些转录本包含这条连续外显子路径
    return tuple(t for t in names if any(tx[t][i:i + len(path)] == list(path) for i in range(len(tx[t]))))

# 读序来源概率 ∝ 分子数 × 有效长度（长转录本产生更多读序）
theta_true = tau * eff / (tau * eff).sum()
src = rng.choice(3, size=20000, p=theta_true)
classes = Counter(compatible(exon_path(names[t], rng.integers(eff[t]))) for t in src)
print('等价类计数:', dict(classes))

theta = np.full(3, 1 / 3)
for it in range(200):                              # EM
    w = np.zeros(3)
    for c, n in classes.items():
        idx = [names.index(t) for t in c]
        resp = theta[idx] / eff[idx]               # E 步：按 θ_t/ℓ̃_t 分配多重比对读序
        w[idx] += n * resp / resp.sum()
    theta = w / w.sum()                            # M 步
tau_em = theta / eff / (theta / eff).sum()

def naive(split):                                  # 对照：丢掉多重比对读序 / 平均分给兼容转录本
    w = np.zeros(3)
    for c, n in classes.items():
        idx = [names.index(t) for t in c]
        if split or len(idx) == 1:
            w[idx] += n / len(idx)
    return w / eff / (w / eff).sum()

print('真实 τ        ', np.round(tau, 3).tolist())
print('EM            ', np.round(tau_em, 3).tolist())
print('只用唯一读序   ', np.round(naive(False), 3).tolist())
print('多重读序平分   ', np.round(naive(True), 3).tolist())
