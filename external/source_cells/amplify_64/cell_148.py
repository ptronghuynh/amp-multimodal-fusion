# Chạy đúng với mo ta kiến trúc
# CELL A — Bộ 64 physicochemical features riêng
import numpy as np
import torch

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA20_SET = set(AA20)

# Kyte-Doolittle hydrophobicity
KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3
}

# Flexibility
FLEX = {
    "A": 0.360, "C": 0.350, "D": 0.510, "E": 0.500, "F": 0.310,
    "G": 0.540, "H": 0.320, "I": 0.460, "K": 0.470, "L": 0.370,
    "M": 0.300, "N": 0.460, "P": 0.510, "Q": 0.490, "R": 0.530,
    "S": 0.510, "T": 0.440, "V": 0.390, "W": 0.310, "Y": 0.420
}

# Bulkiness
BULK = {
    "A": 11.5, "C": 13.46, "D": 11.68, "E": 13.57, "F": 19.8,
    "G": 3.4, "H": 13.69, "I": 21.4, "K": 15.71, "L": 21.4,
    "M": 16.25, "N": 12.82, "P": 17.43, "Q": 14.45, "R": 14.28,
    "S": 9.47, "T": 15.77, "V": 21.57, "W": 21.67, "Y": 18.03
}

# Residue mass
MW = {
    "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13, "F": 165.19,
    "G": 75.07, "H": 155.16, "I": 131.18, "K": 146.19, "L": 131.18,
    "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
    "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19
}

# Boman-like scale
BOMAN = {
    "A": 0.17, "C": -0.24, "D": 3.00, "E": 2.68, "F": -1.13,
    "G": 0.01, "H": 2.06, "I": -1.12, "K": 3.77, "L": -1.25,
    "M": -0.67, "N": 0.42, "P": 1.44, "Q": 0.58, "R": 3.71,
    "S": 0.13, "T": 0.14, "V": -0.46, "W": 1.85, "Y": 1.61
}

PKA_POS = {"K": 10.5, "R": 12.4, "H": 6.0}
PKA_NEG = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}
PKA_NTERM = 8.0
PKA_CTERM = 3.1

BASIC = set("KRH")
ACIDIC = set("DE")
POLAR = set("STNQ")
NONPOLAR = set("AVLIMPG")
HYDROPHOBIC = set("AVILMFWYC")
AROMATIC = set("FWYH")
ALIPHATIC = set("AVIL")
SULFUR = set("CM")
AMIDE = set("NQ")
TINY = set("ACGST")
SMALL = set("ACDGNPSTV")

# reduced 4-group alphabet for 16 dipeptide features
GROUP_MAP = {}
for aa in "AVILMFWYC":
    GROUP_MAP[aa] = 0
for aa in "STNQGP":
    GROUP_MAP[aa] = 1
for aa in "KRH":
    GROUP_MAP[aa] = 2
for aa in "DE":
    GROUP_MAP[aa] = 3


def _clean_seq(seq: str) -> str:
    seq = str(seq).upper().strip()
    return "".join([a for a in seq if a in AA20_SET])


def _safe_moments(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return 0.0, 0.0, 0.0, 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-8:
        return mu, sd, 0.0, 0.0
    z = (x - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)
    return mu, sd, skew, kurt


def _net_charge(seq, ph=7.0):
    pos = 0.0
    neg = 0.0

    pos += 1.0 / (1.0 + 10 ** (ph - PKA_NTERM))
    neg += 1.0 / (1.0 + 10 ** (PKA_CTERM - ph))

    for aa in seq:
        if aa in PKA_POS:
            pos += 1.0 / (1.0 + 10 ** (ph - PKA_POS[aa]))
        if aa in PKA_NEG:
            neg += 1.0 / (1.0 + 10 ** (PKA_NEG[aa] - ph))

    return pos - neg


def _estimate_pI(seq):
    lo, hi = 0.0, 14.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _net_charge(seq, ph=mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _hydrophobic_moment(seq, window=11, angle_deg=100.0):
    if len(seq) == 0:
        return 0.0

    theta = np.deg2rad(angle_deg)

    def moment(fragment):
        vals = np.array([KD[a] for a in fragment], dtype=np.float32)
        x = np.sum(vals * np.cos(theta * np.arange(len(vals))))
        y = np.sum(vals * np.sin(theta * np.arange(len(vals))))
        return float(np.sqrt(x * x + y * y) / max(1, len(vals)))

    if len(seq) < window:
        return moment(seq)

    vals = [moment(seq[i:i+window]) for i in range(len(seq) - window + 1)]
    return float(np.mean(vals)) if len(vals) else 0.0


def physchem_features_64(seq: str) -> torch.Tensor:
    seq = _clean_seq(seq)
    n = len(seq)

    if n == 0:
        return torch.zeros(64, dtype=torch.float32)

    # 1) 20 AA composition
    aa_comp = np.array([seq.count(a) / n for a in AA20], dtype=np.float32)

    # 2) 16 reduced dipeptide composition
    red_dipep = np.zeros(16, dtype=np.float32)
    if n >= 2:
        for i in range(n - 1):
            g1 = GROUP_MAP[seq[i]]
            g2 = GROUP_MAP[seq[i + 1]]
            red_dipep[g1 * 4 + g2] += 1.0
        red_dipep /= (n - 1)

    # 3) 28 global descriptors
    hyd = np.array([KD[a] for a in seq], dtype=np.float32)
    flex = np.array([FLEX[a] for a in seq], dtype=np.float32)
    bulk = np.array([BULK[a] for a in seq], dtype=np.float32)
    mw = np.array([MW[a] for a in seq], dtype=np.float32)
    boman = np.array([BOMAN[a] for a in seq], dtype=np.float32)

    hyd_mean, hyd_std, hyd_skew, hyd_kurt = _safe_moments(hyd)
    flex_mean, flex_std, _, _ = _safe_moments(flex)
    bulk_mean, bulk_std, _, _ = _safe_moments(bulk)
    mw_mean, mw_std, _, _ = _safe_moments(mw)

    probs = np.array([seq.count(a) / n for a in AA20 if seq.count(a) > 0], dtype=np.float32)
    entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0

    net_charge = _net_charge(seq, ph=7.0)
    pI = _estimate_pI(seq)
    muH = _hydrophobic_moment(seq, window=11, angle_deg=100.0)

    global_feats = np.array([
        float(n),                               # 1
        float(np.log1p(n)),                     # 2
        float(net_charge),                      # 3
        float(abs(net_charge)),                 # 4
        float(net_charge / n),                  # 5
        sum(a in BASIC for a in seq) / n,       # 6
        sum(a in ACIDIC for a in seq) / n,      # 7
        sum(a in POLAR for a in seq) / n,       # 8
        sum(a in NONPOLAR for a in seq) / n,    # 9
        sum(a in HYDROPHOBIC for a in seq) / n, # 10
        sum(a in AROMATIC for a in seq) / n,    # 11
        sum(a in ALIPHATIC for a in seq) / n,   # 12
        sum(a in SULFUR for a in seq) / n,      # 13
        sum(a in AMIDE for a in seq) / n,       # 14
        sum(a in TINY for a in seq) / n,        # 15
        sum(a in SMALL for a in seq) / n,       # 16
        entropy,                                # 17
        hyd_mean,                               # 18
        hyd_std,                                # 19
        hyd_skew,                               # 20
        hyd_kurt,                               # 21
        flex_mean,                              # 22
        flex_std,                               # 23
        bulk_mean,                              # 24
        bulk_std,                               # 25
        mw_mean,                                # 26
        mw_std,                                 # 27
        float(np.mean(boman)),                  # 28
    ], dtype=np.float32)

    feat = np.concatenate([aa_comp, red_dipep, global_feats], axis=0)
    assert feat.shape[0] == 64, feat.shape
    return torch.tensor(feat, dtype=torch.float32)