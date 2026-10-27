# -*- coding: utf-8 -*-
"""第 2 章 §6.1/§7.1：Phred 质量值、FASTQ 质量串解析与期望错误数。

只用标准库。FASTQ 内容写死在脚本里（4 条示例读长），输出可复现。
"""
import math

FASTQ = """\
@r1 illumina-like, quality decays toward the 3' end
ACGTTGCAAGGCTTACCGATTGCAGGCTTAACGGATCCATGG
+
IIIIIIIIIIIIIIIIIIIIIIIIIIIHHHGGFFEDDCB@?<
@r2 mostly Q37 with a low-quality tail
TTGACCGGTAAGCTTGCAATCGGATACGTTAGGCATCAGNNN
+
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF:,,#####
@r3 uniform Q20
GATTACAGATTACAGATTACAGATTACAGATTACAGATTACA
+
555555555555555555555555555555555555555555
@r4 mixture: half Q40, half Q10
CCCCCCCCCCCCCCCCCCCCCGGGGGGGGGGGGGGGGGGGGG
+
IIIIIIIIIIIIIIIIIIIII+++++++++++++++++++++
"""


def q_to_p(q):
    """Phred: Q = -10 log10(p)  =>  p = 10^(-Q/10)."""
    return 10.0 ** (-q / 10.0)


def p_to_q(p):
    return -10.0 * math.log10(p)


def parse_fastq(text):
    """按 4 行一组解析；不假设质量串里没有 '@'（质量字符可以是 '@'）。"""
    lines = text.strip("\n").split("\n")
    assert len(lines) % 4 == 0, "FASTQ 行数应为 4 的倍数"
    for i in range(0, len(lines), 4):
        h, s, plus, q = lines[i:i + 4]
        assert h.startswith("@") and plus.startswith("+") and len(s) == len(q)
        yield h[1:].split()[0], s, q


def guess_offset(qual_strings):
    """粗略判断编码：出现 ASCII < 59 的字符只可能是 Phred+33。"""
    lo = min(min(ord(c) for c in q) for q in qual_strings)
    hi = max(max(ord(c) for c in q) for q in qual_strings)
    if lo < 59:
        return 33, lo, hi
    return 64, lo, hi  # 全部 >= 64 时才可能是 Phred+64（老 Illumina 1.3–1.7）


print("== 1. Phred 换算表（Phred+33 编码）==")
print("  Q   错误概率 p      准确率      ASCII 字符")
for q in (2, 10, 20, 30, 40, 50, 60):
    print("%3d   %-12.6g  %-10.6f  %3d '%s'" % (q, q_to_p(q), 1 - q_to_p(q), q + 33, chr(q + 33)))
print("字符 'I' = ASCII %d -> Q = %d -> p = %g" % (ord("I"), ord("I") - 33, q_to_p(ord("I") - 33)))
print("Solexa 记分 Q_sol=-5 对应的 Phred 值 = %.2f" % (10 * math.log10(1 + 10 ** (-5 / 10))))

records = list(parse_fastq(FASTQ))
off, lo, hi = guess_offset([q for _, _, q in records])
print("\n== 2. 编码判断：质量字符 ASCII 范围 %d–%d，推断 Phred+%d ==" % (lo, hi, off))
print("若误当成 Phred+64，'#'(35) 会被读成 Q=%d，'5'(53) 会被读成 Q=%d（负值说明编码判断错了）" % (35 - 64, 53 - 64))

print("\n== 3. 逐条读长：平均 Q、期望错误数 EE = sum(p_i) ==")
print("%-4s %4s %7s %9s %13s %9s %6s" % ("id", "len", "meanQ", "EE", "L*p(meanQ)", "EE/L", "N数"))
for rid, seq, qual in records:
    qs = [ord(c) - off for c in qual]
    ps = [q_to_p(q) for q in qs]
    L = len(seq)
    mean_q = sum(qs) / L
    ee = sum(ps)
    naive = L * q_to_p(mean_q)
    print("%-4s %4d %7.2f %9.4f %13.4f %9.5f %6d" % (rid, L, mean_q, ee, naive, ee / L, seq.count("N")))

print("\n解读：r4 的平均 Q = 25，看上去不错，但一半碱基是 Q10，")
print("EE = 21*1e-4 + 21*0.1 ≈ 2.10，远大于按平均 Q 估算的 0.13。")
print("期望错误数要先把每个 Q 换成概率再求和，不能先对 Q 取平均（Jensen 不等式：10^(-x/10) 是凸函数）。")

print("\n== 4. 按 maxEE 过滤（Edgar & Flyvbjerg 2015 的做法）==")
for max_ee in (1.0, 0.5):
    kept = []
    for rid, seq, qual in records:
        ee = sum(q_to_p(ord(c) - off) for c in qual)
        if ee <= max_ee:
            kept.append(rid)
    print("maxEE = %.1f -> 保留 %s" % (max_ee, ", ".join(kept)))

print("\n== 5. 末端质量修剪：从 3' 端去掉碱基直到窗口平均 Q >= 20（窗口 4）==")
for rid, seq, qual in records:
    qs = [ord(c) - off for c in qual]
    end = len(qs)
    while end >= 4 and sum(qs[end - 4:end]) / 4 < 20:
        end -= 1
    print("%s: %d -> %d bp，修剪后 EE = %.4f" % (rid, len(qs), end, sum(q_to_p(q) for q in qs[:end])))
