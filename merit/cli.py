"""Scenario selection and command-line parameter overrides."""

import argparse
from inspect import signature
from pathlib import Path

from .model import MERIT

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DATASETS = {
    "nc": ("Graph-NC", "nc", "node-class"),
    "lp": ("Graph-LP", "lp", "link-pred"),
    "nas": ("NAS-BenchGraph", "NAS-Bench-Graph", "nas"),
    "graphgym": ("GraphGym", "gg"),
}
PREFIXES = ("AttributedGraph_", "CitationFull_", "Amazon_", "Coauthor_", "HeterophilousGraph_")


def name_key(name):
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def configurations(objective):
    """Describe the bundled configurations using their scenario filenames."""
    entries = []
    for path in sorted(CONFIGS.glob("*.json")):
        if path.stem.endswith("_top1"):
            continue
        scenario, task = path.stem.split("_", 1)
        benchmark = next(
            benchmark for benchmark, names in DATASETS.items()
            if task.startswith({"nc": "node-class", "lp": "link-pred"}.get(benchmark, benchmark))
        )
        rho = float(task.split("rho", 1)[1].replace("p", ".")) if scenario == "incomplete" else None
        top1 = path.with_name(path.stem + "_top1.json")
        selected = top1 if objective == "top1" and top1.is_file() else path
        entries.append((scenario, benchmark, DATASETS[benchmark], rho, selected))
    for path in sorted((CONFIGS / "actual").glob("*.json")):
        benchmark, dataset = path.stem.split("__", 1)[1].split("_", 1)
        short_name = dataset
        for prefix in PREFIXES:
            if dataset.startswith(prefix):
                short_name = dataset[len(prefix):]
                break
        entries.append(("actual", benchmark, (short_name, dataset), None, path))
    return entries


def parser(batch=False, actual=False):
    description = ("Run MERIT Top-1 actual performance with each target's best parameters."
                   if actual else "Run MERIT with the selected scenario's best parameters.")
    arguments = argparse.ArgumentParser(description=description)
    selection = arguments.add_argument_group("targets" if actual else "scenario")
    if actual:
        arguments.set_defaults(scenario=["actual"] if batch else "actual", rho=None, objective="top1")
    else:
        selection.add_argument("--scenario", required=True, choices=("complete", "incomplete", "cross-domain", "cross-scale", "cross-task", "actual"),
                               nargs="+" if batch else None)
        selection.add_argument("--rho", type=float, choices=(0.1, 0.3, 0.5, 0.7),
                               nargs="+" if batch else None, help="Missing-performance ratio for incomplete")
        selection.add_argument("--objective", choices=("ranking", "top1"), default="ranking",
                               help="Choose ranking or Top-1 parameters; actual always uses its Top-1 configuration")
    selection.add_argument("--dataset", required=not batch, nargs="+" if batch else None,
                           help=("Target names, e.g. Photo; omit to run all targets in the selected benchmark"
                                 if actual else "Graph-NC, Graph-LP, NAS or GraphGym for ranking; target name for actual performance"))
    selection.add_argument("--benchmark", type=str.lower, choices=("nc", "lp", "nas", "graphgym"),
                           help="Benchmark or candidate pool")
    execution = arguments.add_argument_group("execution")
    execution.add_argument("--device", default="cuda:0")
    execution.add_argument("--out", type=Path)
    execution.add_argument("--fold", type=int, help="Run one LOGO fold instead of all configured folds")
    seeds = execution.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, help="Run one seed instead of the configured seeds")
    seeds.add_argument("--seeds", type=int, nargs="+", help="Override the configured seed list")
    model = arguments.add_argument_group("parameter overrides (omitted values use the selected configuration)")
    for name, parameter in signature(MERIT).parameters.items():
        if name == "device":
            continue
        flag = name[4:] if name in ("use_erl", "use_eil") else name
        flag = flag.replace("_", "-")
        if isinstance(parameter.default, bool):
            switch = model.add_mutually_exclusive_group()
            switch.add_argument("--" + flag, dest=name, action="store_true")
            switch.add_argument("--no-" + flag, dest=name, action="store_false")
            arguments.set_defaults(**{name: None})
        else:
            model.add_argument("--" + flag, dest=name, type=type(parameter.default))
    model.add_argument("--feature-k", type=int)
    model.add_argument("--prediction-k", type=int)
    model.add_argument("--tie-break", choices=("first", "last"))
    model.add_argument("--svd", choices=("rowwise", "vectorized"))
    model.add_argument("--select-features", type=int, metavar="COUNT",
                       help="Fit sparse feature weights on observed source records and select COUNT features")
    model.add_argument("--feature-penalty", type=float, default=0.001,
                       help="L2,1 regularization for --select-features")
    return arguments


def overrides(args):
    """Translate explicitly supplied arguments to the existing JSON structure."""
    values = {"model": {}}
    for name in signature(MERIT).parameters:
        if name in ("device", "use_erl", "use_eil"):
            continue
        value = getattr(args, name)
        if value is None:
            continue
        if name.startswith(("erl_", "eil_")):
            component, key = name.split("_", 1)
            values.setdefault(component, {})[key] = value
        elif name in ("lambda_erl", "lambda_eil"):
            values.setdefault(name[7:], {})["weight"] = value
        else:
            values["model"][name] = value
    for component in ("erl", "eil"):
        enabled = getattr(args, "use_" + component)
        if enabled is not None:
            values[component] = values.get(component, {}) if enabled else None
    for name in ("feature_k", "prediction_k", "tie_break", "svd", "seeds"):
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    if args.select_features is not None:
        values["feature_selection"] = {"count": args.select_features, "penalty": args.feature_penalty}
    return values


def parse_jobs(batch=False, argv=None, actual=False):
    arguments = parser(batch, actual)
    args = arguments.parse_args(argv)
    scenarios = args.scenario if batch else [args.scenario]
    datasets = args.dataset if batch else [args.dataset]
    ratios = args.rho if batch else ([args.rho] if args.rho is not None else None)
    if ratios is not None and "incomplete" not in scenarios:
        arguments.error("--rho applies to --scenario incomplete")
    if not batch and args.scenario == "incomplete" and args.rho is None:
        arguments.error("incomplete requires --rho")
    entries = configurations(args.objective)
    selected = [
        entry for entry in entries
        if entry[0] in scenarios
        and (args.benchmark is None or entry[1] == args.benchmark)
        and (datasets is None or any(name_key(dataset) in {name_key(name) for name in entry[2]}
                                     for dataset in datasets))
        and (entry[0] != "incomplete" or ratios is None or entry[3] in ratios)
    ]
    if not selected:
        arguments.error("No bundled configuration matches this scenario, dataset, benchmark and rho")
    if datasets is not None:
        matched = {name_key(name) for entry in selected for name in entry[2]}
        missing = [dataset for dataset in datasets if name_key(dataset) not in matched]
        if missing:
            arguments.error("No bundled configuration for: " + ", ".join(missing))
    if not batch and len(selected) != 1:
        arguments.error("This target appears in multiple candidate pools; specify --benchmark nc, lp or nas")
    return args, [entry[4] for entry in selected], overrides(args)
