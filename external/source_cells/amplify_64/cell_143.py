# CELL 6 — Train/eval utilities + metrics không dùng sklearn
def roc_auc_np(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.min() == y_true.max():
        return np.nan
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def average_precision_np(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    if tp[-1] == 0:
        return np.nan
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    recall_prev = np.r_[0.0, recall[:-1]]
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)

def binary_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)

    denom = math.sqrt(max(1, (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mcc = ((tp * tn) - (fp * fn)) / denom

    return {
        "AUC": roc_auc_np(y_true, y_prob),
        "AP": average_precision_np(y_true, y_prob),
        "ACC": acc,
        "MCC": mcc,
        "Sensitivity": sens,
        "Specificity": spec,
        "Precision": prec,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn
    }

def train_one_model(model, train_ds, val_ds, epochs=12, lr=1e-3, batch_size=64, patience=4):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            y = batch["label"].float()

            opt.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

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

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        print(f"Epoch {epoch:02d} | train_loss={mean_train:.4f} | val_loss={mean_val:.4f}")

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
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