# CELL 6 — 4 model
class SeqOnlyModel(nn.Module):
    def __init__(self, seq_dim=320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, batch):
        return self.net(batch["seq_emb"]).squeeze(-1)


class PhysOnlyModel(nn.Module):
    def __init__(self, phys_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phys_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, batch):
        return self.net(batch["phys"]).squeeze(-1)


class FusionConcatModel(nn.Module):
    def __init__(self, seq_dim=320, phys_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(seq_dim + phys_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, batch):
        x = torch.cat([batch["seq_emb"], batch["phys"]], dim=1)
        return self.fc(x).squeeze(-1)


class CrossModalModel(nn.Module):
    def __init__(self, seq_dim=320, phys_dim=64, hidden_dim=128, num_heads=4):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.phys_proj = nn.Linear(phys_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, batch):
        seq = self.seq_proj(batch["seq_emb"]).unsqueeze(1)
        phys = self.phys_proj(batch["phys"]).unsqueeze(1)

        seq_attn, _ = self.attn(seq, phys, phys)
        phys_attn, _ = self.attn(phys, seq, seq)

        joint = torch.cat([seq_attn.squeeze(1), phys_attn.squeeze(1)], dim=-1)
        return self.fc(joint).squeeze(-1)

seq_dim = train_ds[0]["seq_emb"].shape[0]
phys_dim = train_ds[0]["phys"].shape[0]

print("seq_dim =", seq_dim)
print("phys_dim =", phys_dim)