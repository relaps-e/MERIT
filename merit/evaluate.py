"""Ranking and selected candidate performance."""

import numpy as np
from sklearn.metrics import ndcg_score, roc_auc_score


def evaluate(truth, prediction, tie_break="last"):
    rows, selected = [], []
    for values, scores in zip(truth, prediction):
        valid = np.flatnonzero(np.isfinite(values))
        index = (
            valid[np.argmax(scores[valid])]
            if tie_break == "first"
            else valid[np.argsort(scores[valid], kind="mergesort")[-1]]
        )
        selected.append(int(index))
        values, scores = values[valid], scores[valid]
        true_order, predicted_order = np.argsort(values)[::-1], np.argsort(scores)[::-1]
        metrics = {}
        for k in (1, 3, 5):
            metrics[f"nDCG@{k}"] = float(
                ndcg_score(values.reshape(1, -1), scores.reshape(1, -1), k=k)
            )
            metrics[f"HR@{k}"] = float(np.intersect1d(predicted_order[:k], true_order[:k]).size / k)
            metrics[f"Regret@{k}"] = float(
                np.mean(values[true_order[:k]] - values[predicted_order[:k]])
            )
            labels = np.zeros(len(values), dtype=int)
            labels[np.argpartition(values, -k)[-k:]] = 1
            metrics[f"AUC@{k}"] = (
                float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else np.nan
            )
        rows.append(metrics)
    metrics = {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]}
    metrics["Top1Actual"] = float(
        np.mean([values[index] for values, index in zip(truth, selected)])
    )
    metrics["Top1ActualPercent"] = metrics["Top1Actual"] * 100
    return metrics, np.asarray(selected, dtype=np.int32)
