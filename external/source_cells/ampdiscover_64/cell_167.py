# CELL 7 — Metrics không dùng sklearn
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
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }