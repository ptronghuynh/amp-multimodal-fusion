# CELL 16 — Physicochemical features (7-dim, reviewer-proof)

import torch
import numpy as np

# Feature names in order (MUST match physchem_features output)
PHYS_NAMES = [
    "KD_hydrophobicity",
    "is_basic",        # K,R,H
    "is_acidic",       # D,E
    "is_aromatic",     # F,W,Y
    "is_hydrophobic",  # A,V,I,L,M,F,W,Y
    "is_aliphatic",    # A,V,I,L
    "is_sulfur",       # C,M
]

# Kyte–Doolittle hydropathy index
KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3
}

BASIC = set(list("KRH"))
ACIDIC = set(list("DE"))
AROMATIC = set(list("FWY"))
HYDROPHOBIC = set(list("AVILMFWY"))
ALIPHATIC = set(list("AVIL"))
SULFUR = set(list("CM"))

def physchem_features(seq: str) -> torch.Tensor:
    seq = [a for a in seq if a in KD]
    if len(seq) == 0:
        return torch.zeros(7, dtype=torch.float32)

    kd = np.array([KD[a] for a in seq], dtype=np.float32)

    # per-residue binary indicators
    is_basic = np.array([a in BASIC for a in seq], dtype=np.float32)
    is_acidic = np.array([a in ACIDIC for a in seq], dtype=np.float32)
    is_aromatic = np.array([a in AROMATIC for a in seq], dtype=np.float32)
    is_hydrophobic = np.array([a in HYDROPHOBIC for a in seq], dtype=np.float32)
    is_aliphatic = np.array([a in ALIPHATIC for a in seq], dtype=np.float32)
    is_sulfur = np.array([a in SULFUR for a in seq], dtype=np.float32)

    # aggregate by mean across positions -> length-normalized
    feat = np.array([
        kd.mean(),
        is_basic.mean(),
        is_acidic.mean(),
        is_aromatic.mean(),
        is_hydrophobic.mean(),
        is_aliphatic.mean(),
        is_sulfur.mean()
    ], dtype=np.float32)

    return torch.tensor(feat, dtype=torch.float32)
