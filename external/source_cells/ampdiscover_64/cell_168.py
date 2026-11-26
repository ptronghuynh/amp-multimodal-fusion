# CELL 8 — Train loop + predict
def train_one_model(model, train_ds, val_ds, epochs=12, lr=1e-3, batch_size=64, patience=4):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            y = batch["label"].float()

            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                y = batch["label"].float()
                logits = model(batch)
                loss = loss_fn(logits, y)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

@torch.no_grad()
def predict_model(model, dataset, batch_size=128):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true, y_prob = [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()

        y_true.extend(batch["label"].cpu().numpy())
        y_prob.extend(probs)

    return np.array(y_true), np.array(y_prob)