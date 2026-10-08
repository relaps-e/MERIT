"""Scenario input arrays and feature masks."""

import numpy as np

from .feature_selection import select_features


def load_data(config, config_path):
    with np.load(config_path.parent / config["input"], allow_pickle=False) as archive:
        data = {}
        for key in ("M_train", "M_test", "P_train", "P_test", "model_ids"):
            entry = config["entry"] + "/" + key
            data[key] = archive[entry if entry in archive else key].copy()
    if "feature_selection" in config or "mask" not in config:
        mask = select_features(data["M_train"], data["P_train"], **config.get("feature_selection", {}))
    else:
        mask = np.asarray(config["mask"], dtype=np.int32)
    return data, mask
