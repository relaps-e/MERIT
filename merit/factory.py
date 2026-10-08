"""Build MERIT from model, EIL and ERL parameters."""

from .model import MERIT


def build_merit(config, device="cuda:0"):
    parameters = dict(config["model"])
    for component in ("eil", "erl"):
        parameters["use_" + component] = component in config
        parameters.update(
            {
                "lambda_" + component if key == "weight" else component + "_" + key: value
                for key, value in config.get(component, {}).items()
            }
        )
    return MERIT(**parameters, device=device)
