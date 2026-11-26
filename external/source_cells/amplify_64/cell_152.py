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