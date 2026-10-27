# -*- coding: utf-8 -*-
"""第 8 章 §7.5：复现图 8.13B 的“polar contacts”——FOXC2 R121（对应 FOXL2 R103）与 DNA 的极性接触。
结构文件从 RCSB 下载 PDBx/mmCIF，例如 https://files.rcsb.org/download/6O3T.cif。
用法：python ch08_foxc2_r121_contacts.py 6o3t.cif A 121   
      FOXL2 自身的结构：python ch08_foxc2_r121_contacts.py 7vou.cif C 230（7VOU 作者编号 = UniProt 编号 + 127）"""
import sys
import numpy as np

def read_atoms(path):
    """只解析 _atom_site 循环；返回每个原子的 (链, 残基号, 残基名, 原子名, 元素, xyz)。"""
    cols, atoms = [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith("_atom_site."):
            cols.append(line.split(".", 1)[1].strip())
        elif cols and line.startswith(("ATOM", "HETATM")):
            f = dict(zip(cols, line.split()))
            atoms.append((f["auth_asym_id"], int(f["auth_seq_id"]), f["label_comp_id"], f["label_atom_id"].strip('"'),
                          f["type_symbol"], np.array([float(f["Cartn_x"]), float(f["Cartn_y"]), float(f["Cartn_z"])])))
        elif cols and atoms and line.startswith("#"):
            break
    return atoms

path, chain, res = (sys.argv[1:4] + ["6o3t.cif", "A", "121"][len(sys.argv) - 1:])[:3]
res = int(res)
atoms = read_atoms(path)
seq = {r: n for c, r, n, *_ in atoms if c == chain}
print("%s 链残基 %d–%d；%d 号 = %s；%d–%d = %s" % (chain, min(seq), max(seq), res, seq[res], res - 5, res + 2,
      " ".join(seq.get(i, "-") for i in range(res - 5, res + 3))))
dna = [a for a in atoms if a[2] in ("DA", "DC", "DG", "DT") and a[4] in ("N", "O")]
xyz = np.array([a[5] for a in dna])
for c, r, n, atom, el, x in atoms:
    if c == chain and r == res and el in ("N", "O") and atom not in ("N", "O"):  # 只看侧链 N/O
        d = np.linalg.norm(xyz - x, axis=1)
        for j in np.where(d <= 3.5)[0]:
            b = dna[j]
            print("  %s%d %-3s -> 链 %s %s%-3d %-4s %.2f Å" % (n, r, atom, b[0], b[2], b[1], b[3], d[j]))
