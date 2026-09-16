"""Population firing-rate calibration objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RateScore:
    value: float
    diagnostics: dict[str, Any]


def _window_key(trange: Sequence[float]) -> str:
    return "%d_%d" % (trange[0], trange[1])


def _population_rate(pop_rates: Mapping[str, Any], pop: str, trange: Sequence[float]) -> float:
    if pop not in pop_rates:
        raise KeyError(f"Population {pop!r} is missing from popRates")
    value = pop_rates[pop]
    if isinstance(value, Mapping):
        key = _window_key(trange)
        if key not in value:
            raise KeyError(f"Time range {key!r} is missing for population {pop!r}")
        value = value[key]
    return float(value)


def _aggregate_group_rate(
    pop_rates: Mapping[str, Any],
    populations: Sequence[str],
    trange: Sequence[float],
    weighting: str,
    population_sizes: Mapping[str, int] | None,
) -> float:
    rates = np.asarray([_population_rate(pop_rates, pop, trange) for pop in populations], dtype=float)
    if len(rates) == 1 or weighting == "equal":
        return float(np.mean(rates))
    if weighting != "cell_count":
        raise ValueError(f"Unknown rate group weighting: {weighting!r}")
    if population_sizes is None:
        raise ValueError("population_sizes are required for cell_count weighting")
    weights = np.asarray([population_sizes.get(pop, 0) for pop in populations], dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError(f"Invalid population sizes for group {list(populations)!r}")
    return float(np.average(rates, weights=weights))


def validate_rate_config(config: Mapping[str, Any], **_: Any) -> None:
    scheme_name = config.get("scheme", "population")
    schemes = config.get("schemes", {})
    if scheme_name not in schemes:
        raise ValueError(f"Unknown rate scheme {scheme_name!r}; choose one of {sorted(schemes)}")
    groups = schemes[scheme_name].get("groups", {})
    if not groups:
        raise ValueError(f"Rate scheme {scheme_name!r} has no groups")
    for name, group in groups.items():
        if not group.get("populations"):
            raise ValueError(f"Rate group {name!r} has no populations")
        for key in ("target", "width", "min"):
            if key not in group:
                raise ValueError(f"Rate group {name!r} is missing {key!r}")
        if float(group["width"]) <= 0:
            raise ValueError(f"Rate group {name!r} width must be positive")


def calculate_rate_loss(
    pop_rates: Mapping[str, Any],
    tranges: Sequence[Sequence[float]],
    config: Mapping[str, Any],
    population_sizes: Mapping[str, int] | None = None,
) -> RateScore:
    """Calculate the configured group loss.

    With the ``population`` scheme this is algebraically identical to the old
    ``rateFitnessFuncTranges`` implementation.
    """

    validate_rate_config(config)
    scheme_name = str(config.get("scheme", "population"))
    groups = config["schemes"][scheme_name]["groups"]
    weighting = str(config.get("group_weighting", "cell_count"))
    max_fitness = float(config.get("max_fitness", 1000.0))

    group_losses: dict[str, float] = {}
    group_rates: dict[str, dict[str, float]] = {}
    for group_name, group in groups.items():
        populations = list(group["populations"])
        losses = []
        rates_by_window = {}
        for trange in tranges:
            rate = _aggregate_group_rate(
                pop_rates,
                populations,
                trange,
                weighting,
                population_sizes,
            )
            key = _window_key(trange)
            rates_by_window[key] = rate
            if rate > float(group["min"]):
                loss = min(
                    np.exp(abs(float(group["target"]) - rate) / float(group["width"])),
                    max_fitness,
                )
            else:
                loss = max_fitness
            losses.append(float(loss))
        group_rates[group_name] = rates_by_window
        group_losses[group_name] = float(np.mean(losses))

    value = float(np.mean(list(group_losses.values())))
    return RateScore(
        value=value,
        diagnostics={
            "scheme": scheme_name,
            "group_losses": group_losses,
            "group_rates": group_rates,
        },
    )


def _population_sizes(sim: Any) -> dict[str, int]:
    sizes = {}
    for name, pop in getattr(sim.net, "pops", {}).items():
        gids = getattr(pop, "cellGids", None)
        if gids is None and hasattr(pop, "tags"):
            gids = pop.tags.get("cellGids", [])
        sizes[name] = len([] if gids is None else gids)
    return sizes


def score_rate_objective(context: Any, config: Mapping[str, Any]) -> RateScore:
    if "pop_rates" not in context.cache:
        context.cache["pop_rates"] = context.sim.analysis.popAvgRates(
            tranges=context.cfg.printPopAvgRates,
            show=False,
        )
    result = calculate_rate_loss(
        context.cache["pop_rates"],
        context.cfg.printPopAvgRates,
        config,
        population_sizes=_population_sizes(context.sim),
    )
    value = result.value + float(context.guard_penalty)
    diagnostics = dict(result.diagnostics)
    diagnostics["spike_guard_penalty"] = float(context.guard_penalty)
    diagnostics["unpenalized_value"] = result.value
    return RateScore(value=value, diagnostics=diagnostics)


__all__ = ["RateScore", "calculate_rate_loss", "score_rate_objective", "validate_rate_config"]
