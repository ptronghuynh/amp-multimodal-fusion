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