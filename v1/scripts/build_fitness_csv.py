"""
Reconstruct full protein sequences for T7 and TEV from WT + mutation strings,
and save fitness CSVs compatible with the active learning pipeline.

Output columns: protein, label, stage
"""
import re
import pandas as pd
import numpy as np

# ── WT sequences ─────────────────────────────────────────────────────────────
T7_WT = (
    "MNTINIAKNDFSDIELAAIPFNTLADHYGERLAREQLALEHESYEMGEARFRKMFERQLK"
    "AGEVADNAAAKPLITTLLPKMIARINDWFEEVKAKRGKRPTAFQFLQEIKPEAVAYITIK"
    "TTLACLTSADNTTVQAVASAIGRAIEDEARFGRIRDLEAKHFKKNVEEQLNKRVGHVYKK"
    "AFMQVVEADMLSKGLLGGEAWSSWHKEDSIHVGVRCIEMLIESTGMVSLHRQNAGVVGQD"
    "SETIELAPEYAEAIATRAGALAGISPMFQPCVVPPKPWTGITGGGYWANGRRPLALVRTH"
    "SKKALMRYEDVYMPEVYKAINIAQNTAWKINKKVLAVANVITKWKHCPVEDIPAIEREEL"
    "PMKPEDIDMNPEALTAWKRAAAAVYRKDKARKSRRISLEFMLEQANKFANHKAIWFPYNM"
    "DWRGRVYAVSMFNPQGNDMTKGLLTLAKGKPIGKEGYYWLKIHGANCAGVDKVPFPERIK"
    "FIEENHENIMACAKSPLENTWWAEQDSPFCFLAFCFEYAGVQHHGLSYNCSLPLAFDGSC"
    "SGIQHFSAMLRDEVGGRAVNLLPSETVQDIYGIVAKKVNEILQADAINGTDNEVVTVTDE"
    "NTGEISEKVKLGTKALAGQWLAYGVTRSVTKRSVMTLAYGSKEFGFRQQVLEDTIQPAID"
    "SGKGLMFTQPNQAAGYMAKLIWESVSVTVVAAVEAMNWLKSAAKLLAAEVKDKKTGEILR"
    "KRCAVHWVTPDGFPVWQEYKKPIQTRLNLMFLGQFRLQPTINTNKDSEIDAHKQESGIAP"
    "NFVHSQDGSHLRKTVVWAHEKYGIESFALIHDSFGTIPADAANLFKAVRETMVDTYESCD"
    "VLADFYDQFADQLHESQLDKMPALPAKGNLNLRDILESDFAFA"
)

TEV_WT = (
    "GESLFKGPRDYNPISSTICHLTNESDGHTTSLYGIGFGPFIITNKHLFRRNNGTLLVQSL"
    "HGVFKVKNTTTLQQHLIDGRDMIIIRMPKDFPPFPQKLKFREPQREERICLVTTNFQTK"
    "SMSSMVSDTSCTFPSSDGIFWKHWIQTKDGQCGSPLVSTRDGFIVGIHSASNFTNTNNYF"
    "TSVPKNFMELLTNQEAQQWVSGWRLNADSVLWGGHKVFMVKPEEPFQPVKEATQLMN"
)


def apply_mutations(wt: str, muts_str: str) -> str:
    seq = list(wt)
    for mut in muts_str.split(":"):
        wt_aa = mut[0]
        pos = int(re.search(r"\d+", mut).group())  # 1-indexed
        new_aa = mut[-1]
        assert seq[pos - 1] == wt_aa, f"WT mismatch at {mut}: got {seq[pos-1]}"
        seq[pos - 1] = new_aa
    return "".join(seq)


def build(name, wt, scale2max_csv, out_csv):
    df = pd.read_csv(scale2max_csv)
    seqs = []
    for _, row in df.iterrows():
        if pd.isna(row["muts"]) or row["muts"] in ("", "WT"):
            seqs.append(wt)
        else:
            seqs.append(apply_mutations(wt, row["muts"]))

    out = pd.DataFrame({
        "protein": seqs,
        "label":   df["fitness"].values,
        "stage":   "train",
    })
    # scale to [0, max] like GB1/TrpB (already scale2max)
    out.to_csv(out_csv, index=False)
    print(f"{name}: {len(out)} variants → {out_csv}")
    print(f"  label range: [{out['label'].min():.4f}, {out['label'].max():.4f}]")
    print(f"  seq length: {len(seqs[0])}")


build("T7",  T7_WT,  "data/li2024/data/T7/scale2max/T7.csv",   "data/t7/t7_fitness.csv")
build("TEV", TEV_WT, "data/li2024/data/TEV/scale2max/TEV.csv", "data/tev/tev_fitness.csv")
