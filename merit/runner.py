"""Train a scenario and write one readable results log."""

import copy
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_data
from .evaluate import evaluate
from .factory import build_merit
from .features import build_features
from .utils import set_seed

ROOT = Path(__file__).resolve().parents[1]


def run_line(metrics, index, total, actual=False):
    text = f"Run {index}/{total} | Actual Top-1 {metrics['Top1ActualPercent']:.6f}%"
    return text if actual else text + f" | nDCG@5 {metrics['nDCG@5']:.6f}"


def result_lines(results, actual=False):
    """Format individual runs and their mean and sample standard deviation."""
    lines = [
        run_line(metrics, i, len(results), actual)
        for i, metrics in enumerate(results, 1)
    ]
    lines.append("\nResults (mean ± std)")
    keys = ["Top1ActualPercent"] if actual else [key for key in results[0] if key != "Top1Actual"]
    for key in keys:
        values = [metrics[key] for metrics in results]
        mean = np.mean(values)
        deviation = f" ± {np.std(values, ddof=1):.6f}" if len(values) > 1 else ""
        label = "Actual Top-1 (%)" if key == "Top1ActualPercent" else key
        lines.append(f"{label:<16} {mean:.6f}{deviation}")
    return lines


def prepare(config, path):
    data, mask = load_data(config, path)
    model = build_merit(config, "cpu")
    features = build_features(
        data["M_train"],
        data["M_test"],
        data["P_train"],
        mask,
        model.hid_dim,
        config.get("feature_k", model.k_nn),
        config.get("svd", "rowwise"),
    )
    return data, mask, features


def merge_settings(config, overrides):
    config = copy.deepcopy(config)
    for key, value in (overrides or {}).items():
        if key in ("model", "erl", "eil"):
            if value is None:
                config.pop(key, None)
            else:
                config.setdefault(key, {}).update(value)
        else:
            config[key] = value
    return config


def fold_configs(config, overrides=None, fold=None):
    """Resolve the configured graph splits and their training parameters."""
    base = {key: value for key, value in config.items() if key != "folds"}
    cases = [merge_settings(base, item) for item in config.get("folds", [{}])]
    if fold is not None:
        cases = [case for case in cases if case.get("fold") == fold]
        if not cases:
            raise ValueError(f"No configured fold {fold}")
    return [merge_settings(case, overrides) for case in cases]


def run(config_path, device="cuda:0", out=None, seed=None, overrides=None, fold=None):
    """Train the configured runs and write results/<scenario>.log."""
    path = Path(config_path)
    actual = path.stem.startswith("actual__")
    out = Path(out) if out is not None else ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / (path.stem + ".log")
    config = json.loads(path.read_text("utf-8"))
    cases = fold_configs(config, overrides, fold)
    prepared = [(case, *prepare(case, path)) for case in cases]
    seeds = [seed] if seed is not None else cases[0]["seeds"]
    results = []
    for index, seed in enumerate(seeds, 1):
        fold_results = []
        for case, data, mask, features in prepared:
            set_seed(seed, device)
            method = build_merit(case, device)
            method.set_node_features(features)
            method.fit(data["M_train"][:, mask], data["P_train"])
            method.k_nn = case.get("prediction_k", method.k_nn)
            prediction = method.predict(data["M_test"][:, mask])
            metrics, _ = evaluate(data["P_test"], prediction, case.get("tie_break", "last"))
            fold_results.append(metrics)
            if "fold" in case:
                print(f"Run {index}/{len(seeds)} | Fold {case['fold']} | nDCG@5 {metrics['nDCG@5']:.6f}", flush=True)
            del method
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        metrics = {
            key: float(np.mean([row[key] for row in fold_results]))
            for key in fold_results[0]
        }
        results.append(metrics)
        lines = ["MERIT | " + path.stem, *result_lines(results, actual)]
        log_path.write_text("\n".join(lines) + "\n", "utf-8")
        print(run_line(metrics, index, len(seeds), actual), flush=True)
    summary = dict(
        seeds=seeds,
        metrics={
            key: dict(
                mean=float(np.mean([r[key] for r in results])),
                sample_std=(
                    float(np.std([r[key] for r in results], ddof=1))
                    if len(results) > 1
                    else None
                ),
            )
            for key in results[0]
        },
    )
    print("\n".join(result_lines(results, actual)[len(results):]), flush=True)
    print("Results: " + str(log_path), flush=True)
    return summary


def main(batch=False, actual=False):
    from .cli import parse_jobs

    args, paths, parameters = parse_jobs(batch, actual=actual)
    for path in paths:
        print("Running " + path.stem, flush=True)
        run(path, args.device, args.out, args.seed, parameters, args.fold)
