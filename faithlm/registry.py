"""Component registry for FaithLM.

Every dataset, predictor, explainer and metric registers itself here under a
short name. Configs refer to components by that name, so adding a new model is
a matter of writing one function with a decorator rather than editing the
if/elif chains that used to live in main_local.py and main_global.py.
"""

from typing import Any, Callable, Dict

_REGISTRIES: Dict[str, Dict[str, Callable]] = {
    "dataset": {},
    "predictor": {},
    "explainer": {},
    "metric": {},
}


def _register(kind: str, name: str) -> Callable:
    if kind not in _REGISTRIES:
        raise KeyError(f"Unknown registry kind '{kind}'. Valid: {sorted(_REGISTRIES)}")

    def decorator(fn: Callable) -> Callable:
        if name in _REGISTRIES[kind]:
            raise KeyError(f"{kind} '{name}' is already registered")
        _REGISTRIES[kind][name] = fn
        return fn

    return decorator


def register_dataset(name: str) -> Callable:
    return _register("dataset", name)


def register_predictor(name: str) -> Callable:
    return _register("predictor", name)


def register_explainer(name: str) -> Callable:
    return _register("explainer", name)


def register_metric(name: str) -> Callable:
    return _register("metric", name)


def get(kind: str, name: str) -> Any:
    """Look up a registered component, with a helpful error when it is missing."""
    if kind not in _REGISTRIES:
        raise KeyError(f"Unknown registry kind '{kind}'. Valid: {sorted(_REGISTRIES)}")
    if name not in _REGISTRIES[kind]:
        available = ", ".join(sorted(_REGISTRIES[kind])) or "(none registered)"
        raise KeyError(f"No {kind} named '{name}'. Available: {available}")
    return _REGISTRIES[kind][name]


def available(kind: str) -> list:
    if kind not in _REGISTRIES:
        raise KeyError(f"Unknown registry kind '{kind}'. Valid: {sorted(_REGISTRIES)}")
    return sorted(_REGISTRIES[kind])
