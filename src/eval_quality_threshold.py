import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

def find_best_threshold(y_true, y_prob):
    best = None
    for t in np.linspace(0.05, 0.95, 19):
        y_hat = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        p = precision_score(y_true, y_hat, zero_division=0)
        r = recall_score(y_true, y_hat, zero_division=0)
        score = f1
        if best is None or score > best["score"]:
            best = {"threshold": float(t), "f1": float(f1), "precision": float(p), "recall": float(r), "score": float(score)}
    return best

def report_quality(y_true, y_prob):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)

    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    best = find_best_threshold(y_true, y_prob)

    print(f"QUALITY ROC-AUC: {roc:.4f}")
    print(f"QUALITY PR-AUC:  {ap:.4f}")
    print("Best threshold (max F1):", best)
