import numpy as np

# 4 个基因、2 个样本。样本 B 里只有 g4 被上调 21 倍，g1–g3 的真实表达没有变
length = np.array([2000, 1000, 4000, 500])          # 基因长度 (bp)
counts = np.array([[400, 200, 800, 100],            # 样本 A
                   [400, 200, 800, 2100]])          # 样本 B

def rpkm(c, L):                                     # 式 10.1
    return c / (c.sum() / 1e6) / (L / 1e3)

def tpm(c, L):                                      # 式 10.3
    a = c / (L / 1e3)                               # 先除长度（每千碱基读序数）
    return a / a.sum() * 1e6                        # 再按长度校正后的总量缩放

for name, f in [('RPKM', rpkm), ('TPM', tpm)]:
    m = np.vstack([f(c, length) for c in counts])
    print(f'{name}: A={np.round(m[0]).tolist()}  B={np.round(m[1]).tolist()}  列和={np.round(m.sum(1)).tolist()}')

def size_factors(cnt):                              # DESeq2 的 median-of-ratios；cnt: 基因×样本
    logc = np.log(cnt)
    keep = np.isfinite(logc).all(axis=1)            # 含 0 的基因不参与
    log_ref = logc[keep].mean(axis=1)               # 每个基因的几何均值 = 伪参考样本
    return np.exp(np.median(logc[keep] - log_ref[:, None], axis=0))

sf = size_factors(counts.T.astype(float))
print('文库大小:', counts.sum(1).tolist(), ' median-of-ratios 大小因子:', np.round(sf, 3).tolist())
print('按大小因子归一化后:', np.round(counts / sf[:, None]).tolist())
