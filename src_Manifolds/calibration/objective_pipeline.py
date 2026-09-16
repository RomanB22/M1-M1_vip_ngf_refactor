"""Evaluate configured calibration objectives after one NetPyNE simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import traceback
from typing import Any, Mapping

import numpy as np

from .objective_config import get_active_objectives, get_objectives
from .objectives import OBJECTIVE_NETWORK_VALIDATORS, OBJECTIVE_SCORERS, OBJECTIVE_VALIDATORS


@dataclass
class ObjectiveContext:
    sim: Any
    cfg: Any
    guard_summary: Mapping[str, Any] | None = None
    guard_penalty: float = 0.0
    cache: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.cache is None:
            self.cache = {}


def _objective_specs(cfg: Any) -> dict[str, dict[str, Any]]:
    configured = getattr(cfg, "objectives", None)
    return get_active_objectives(get_objectives() if configured is None else configured)


def validate_objective_setup(cfg: Any, strict_files: bool = True) -> None:
    """Fail before network construction when a configured objective is invalid."""

    for name, spec in _objective_specs(cfg).items():
        kind = str(spec.get("kind", ""))
        if kind not in OBJECTIVE_VALIDATORS:
            raise ValueError(f"Objective {name!r} has unknown kind {kind!r}")
        direction = str(spec.get("direction", "minimize"))
        if direction not in {"minimize", "maximize"}:
            raise ValueError(f"Objective {name!r} has invalid direction {direction!r}")
        failure_value = float(spec.get("failure_value", np.nan))
        if not np.isfinite(failure_value):
            raise ValueError(f"Objective {name!r} must define a finite failure_value")
        OBJECTIVE_VALIDATORS[kind](spec.get("config", {}), strict_files=strict_files)


def validate_instantiated_objectives(sim: Any, cfg: Any) -> None:
    """Run cheap checks that require instantiated cells, before connections."""

    for spec in _objective_specs(cfg).values():
        validator = OBJECTIVE_NETWORK_VALIDATORS.get(str(spec.get("kind", "")))
        if validator is not None:
            validator(sim, cfg, spec.get("config", {}))


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def evaluate_objectives(
    sim: Any,
    cfg: Any,
    guard_summary: Mapping[str, Any] | None = None,
    guard_penalty: float = 0.0,
) -> dict[str, Any]:
    """Return BatchTK scalar metrics plus structured diagnostics.

    Any objective failure or spike-guard blockade applies the finite sentinel
    to every optimized value. Raw successful scores remain in diagnostics, so
    an invalid candidate cannot become a useful Pareto point.
    """

    context = ObjectiveContext(
        sim=sim,
        cfg=cfg,
        guard_summary=guard_summary,
        guard_penalty=float(guard_penalty),
        cache={},
    )
    payload: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    failures: dict[str, str] = {}
    if guard_penalty > 0:
        failures["spike_guard"] = (
            f"blocked-network guard penalty is {float(guard_penalty):g}"
        )
    for name, spec in _objective_specs(cfg).items():
        kind = str(spec["kind"])
        try:
            result = OBJECTIVE_SCORERS[kind](context, spec.get("config", {}))
            value = float(result.value)
            if not np.isfinite(value):
                raise ValueError(f"Objective returned non-finite value {value}")
            payload[name] = value
            diagnostics[name] = {
                **_json_safe(result.diagnostics),
                "raw_value": value,
            }
        except Exception as exc:
            payload[name] = float(spec["failure_value"])
            failures[name] = f"{type(exc).__name__}: {exc}"
            diagnostics[name] = {
                "failed": True,
                "error": failures[name],
                "traceback": traceback.format_exc(limit=8),
            }

    if failures:
        for name, spec in _objective_specs(cfg).items():
            payload[name] = float(spec["failure_value"])

    payload["trial_valid"] = not failures
    payload["objective_failures"] = failures
    payload["failure_reason"] = "; ".join(f"{name}: {reason}" for name, reason in failures.items()) or None
    payload["objective_diagnostics"] = diagnostics
    return _json_safe(payload)


__all__ = [
    "ObjectiveContext",
    "evaluate_objectives",
    "validate_instantiated_objectives",
    "validate_objective_setup",
]
