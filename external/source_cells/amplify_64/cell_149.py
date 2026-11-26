# CELL B — Dataset dùng bộ 64 chiều
from torch.utils.data import Dataset

class UnifiedPeptideDataset64(Dataset):
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
            "phys": physchem_features_64(seq),
            "label": torch.tensor(int(s["label"]), dtype=torch.long),
        }

train_ds = UnifiedPeptideDataset64(train, esm_cache)
val_ds   = UnifiedPeptideDataset64(val, esm_cache)
test_ds  = UnifiedPeptideDataset64(test, esm_cache)

sample = train_ds[0]
print(sample.keys())
print("seq_emb:", sample["seq_emb"].shape)
print("phys   :", sample["phys"].shape)
print("label  :", sample["label"])