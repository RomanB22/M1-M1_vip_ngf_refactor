from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def compute_windowed_pop_rates(
    rate_trace,
    windows_ms: Sequence[tuple[int, int]],
    dt_ms: float = 1.0,
    xp=np,
):
    rates = xp.asarray(rate_trace, dtype=float)
    if rates.ndim != 2:
        raise ValueError("rate_trace must have shape [time, population]")
    windowed = []
    for start_ms, end_ms in windows_ms:
        start_idx = int(round(start_ms / dt_ms))
        end_idx = int(round(end_ms / dt_ms))
        if end_idx <= start_idx:
            raise ValueError(f"Invalid window {(start_ms, end_ms)}")
        windowed.append(rates[start_idx:end_idx].mean(axis=0))
    return xp.stack(windowed, axis=1)


def rate_fitness_tranges_jax(
    windowed_rates,
    targets,
    widths,
    mins,
    max_fitness: float = 1000.0,
    xp=np,
):
    rates = xp.asarray(windowed_rates, dtype=float)
    target_arr = xp.asarray(targets, dtype=float)
    width_arr = xp.asarray(widths, dtype=float)
    min_arr = xp.asarray(mins, dtype=float)

    if rates.ndim != 2:
        raise ValueError("windowed_rates must have shape [population, window]")
    penalties = xp.where(
        rates > min_arr[:, None],
        xp.minimum(xp.exp(xp.abs(target_arr[:, None] - rates) / width_arr[:, None]), max_fitness),
        max_fitness,
    )
    value = penalties.mean(axis=1).mean()
    if xp is np:
        return float(value)
    return value


def build_target_arrays(population_specs: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = []
    widths = []
    mins = []
    for spec in population_specs:
        target = spec["target"]
        targets.append(float(target["target"]))
        widths.append(float(target["width"]))
        mins.append(float(target["min"]))
    return np.asarray(targets, dtype=float), np.asarray(widths, dtype=float), np.asarray(mins, dtype=float)


def windowed_rates_to_dict(
    population_names: Sequence[str],
    windowed_rates: np.ndarray,
    windows_ms: Sequence[tuple[int, int]],
) -> dict[str, dict[str, float]]:
    rates = np.asarray(windowed_rates, dtype=float)
    if rates.shape != (len(population_names), len(windows_ms)):
        raise ValueError("windowed_rates shape does not match population_names/windows_ms")
    out: dict[str, dict[str, float]] = {}
    for pop_idx, pop_name in enumerate(population_names):
        out[pop_name] = {}
        for window_idx, (start_ms, end_ms) in enumerate(windows_ms):
            out[pop_name][f"{start_ms}_{end_ms}"] = float(rates[pop_idx, window_idx])
    return out
