from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}")
    cm = contingency_matrix(y_true, y_pred)
    rows, cols = linear_sum_assignment(-cm)
    return float(cm[rows, cols].sum() / y_true.size)


def rare_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean recall of true classes smaller than sqrt(n), after Hungarian matching.

    Used to show whether a minority class is recovered rather than dissolved.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    cut = np.sqrt(n)
    cm = contingency_matrix(y_true, y_pred)
    rows, cols = linear_sum_assignment(-cm)
    pred_for_true = {int(r): int(c) for r, c in zip(rows, cols)}
    recalls = []
    for t, count in zip(*np.unique(y_true, return_counts=True)):
        if count >= cut:
            continue
        t = int(t)
        if t not in pred_for_true:
            recalls.append(0.0)
            continue
        p = pred_for_true[t]
        true_ids = np.flatnonzero(y_true == t)
        recalls.append(float(np.mean(y_pred[true_ids] == p)))
    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "ACC": cluster_acc(y_true, y_pred),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "n_clusters": float(len(np.unique(y_pred))),
        "n_true": float(len(np.unique(y_true))),
        "rare_recall": rare_class_recall(y_true, y_pred),
    }
