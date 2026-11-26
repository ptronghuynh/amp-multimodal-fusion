# CELL 4 — Physicochemical features + Dataset
# 7-dim physchem theo notebook hiện tại của anh
KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3
}
BASIC = set("KRH")
ACIDIC = set("DE")
AROMATIC = set("FWY")
HYDROPHOBIC = set("AVILMFWY")
ALIPHATIC = set("AVIL")
SULFUR = set("CM")

def physchem_features(seq: str) -> torch.Tensor:
    seq = [a for a in seq if a in KD]
    if len(seq) == 0:
        return torch.zeros(7, dtype=torch.float32)

    kd = np.array([KD[a] for a in seq], dtype=np.float32)
    is_basic = np.array([a in BASIC for a in seq], dtype=np.float32)
    is_acidic = np.array([a in ACIDIC for a in seq], dtype=np.float32)
    is_aromatic = np.array([a in AROMATIC for a in seq], dtype=np.float32)
    is_hydrophobic = np.array([a in HYDROPHOBIC for a in seq], dtype=np.float32)
    is_aliphatic = np.array([a in ALIPHATIC for a in seq], dtype=np.float32)
    is_sulfur = np.array([a in SULFUR for a in seq], dtype=np.float32)

    feat = np.array([
        kd.mean(),
        is_basic.mean(),
        is_acidic.mean(),
        is_aromatic.mean(),
        is_hydrophobic.mean(),
        is_aliphatic.mean(),
        is_sulfur.mean(),
    ], dtype=np.float32)

    return torch.tensor(feat, dtype=torch.float32)

class UnifiedPeptideDataset(Dataset):
    def __init__(self, samples, esm_cache):
        self.samples = samples
        self.esm_cache = esm_cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        seq = s["sequence"]
        return {
            "seq_emb": torch.tensor(self.esm_cache[seq], dtype=torch.float32),
            "phys": physchem_features(seq),
            "label": torch.tensor(int(s["label"]), dtype=torch.long),
        }

train_ds = UnifiedPeptideDataset(train, esm_cache)
val_ds   = UnifiedPeptideDataset(val, esm_cache)
test_ds  = UnifiedPeptideDataset(test, esm_cache)

sample = train_ds[0]
print(sample.keys())
print("seq_emb:", sample["seq_emb"].shape)
print("phys   :", sample["phys"].shape)
print("label  :", sample["label"])