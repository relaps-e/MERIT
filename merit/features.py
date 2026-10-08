"""Source performance embeddings and target neighbor interpolation."""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def fill_missing(performance):
    performance = np.asarray(performance, dtype=np.float64)
    finite = np.isfinite(performance)
    count = finite.sum(axis=0)
    mean = np.divide(
        np.where(finite, performance, 0).sum(axis=0),
        count,
        out=np.full(performance.shape[1], performance[finite].mean()),
        where=count > 0,
    )
    return np.where(finite, performance, mean[None, :])


def standardize_rows(performance):
    result = np.zeros_like(performance)
    for index, row in enumerate(performance):
        valid = np.isfinite(row)
        if valid.sum() > 1:
            mean, std = float(np.mean(row[valid])), float(np.std(row[valid]))
            result[index, valid] = (row[valid] - mean) / std if std > 1e-10 else row[valid] - mean
    return np.nan_to_num(result)


def build_features(source, target, performance, mask, width, neighbors, svd="rowwise"):
    source = np.asarray(source[:, mask], dtype=np.float32)
    target = np.asarray(target[:, mask], dtype=np.float32)
    performance = fill_missing(performance)
    if svd == "vectorized":
        centered = performance - np.nanmean(performance, axis=1, keepdims=True)
        std = np.nanstd(performance, axis=1, keepdims=True)
        normalized = np.divide(centered, std, out=centered.copy(), where=std > 1e-10)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        normalized = standardize_rows(performance)
    components = max(1, min(width, performance.shape[0] - 1, performance.shape[1] - 1))
    decomposition = TruncatedSVD(n_components=components, random_state=42)
    projected = decomposition.fit_transform(normalized)
    root = np.sqrt(decomposition.singular_values_ + 1e-10)
    source_latent = projected / root if svd == "vectorized" else projected * (1.0 / root)
    model_latent = decomposition.components_.T * root
    source_latent = np.pad(source_latent, ((0, 0), (0, width - components)))
    model_latent = np.pad(model_latent, ((0, 0), (0, width - components)))

    similarity = cosine_similarity(target, source)
    k = min(neighbors, len(source))
    if svd == "vectorized":
        indices = np.argsort(similarity, axis=1)[:, -k:]
        weights = np.take_along_axis(similarity, indices, axis=1)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
        target_latent = np.sum(source_latent[indices] * weights[:, :, None], axis=1)
    else:
        target_latent = np.zeros((len(target), width), dtype=np.float64)
        for index, row in enumerate(similarity):
            indices = np.argsort(row, kind="mergesort")[-k:]
            weights = row[indices]
            denominator = float(weights.sum())
            weights = weights / denominator if abs(denominator) >= 1e-10 else np.full(k, 1.0 / k)
            target_latent[index] = np.sum(source_latent[indices] * weights[:, None], axis=0)
    return {
        "source_task": np.hstack([source, source_latent]).astype(np.float32),
        "target_task": np.hstack([target, target_latent]).astype(np.float32),
        "model": model_latent.astype(np.float32),
    }
