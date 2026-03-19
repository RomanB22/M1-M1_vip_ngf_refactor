from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from math import pi
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from m1_model.jaxley_data import (
    DEFAULT_TAU_MS,
    M1FeatureSet,
    build_feature_set,
    cortical_population_specs,
    default_param_values,
    load_batch_param_space,
    long_range_populations,
    m1_defaults,
    project_root,
)
from m1_model.jaxley_loss import build_target_arrays, compute_windowed_pop_rates, rate_fitness_tranges_jax, windowed_rates_to_dict

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

try:
    import jaxley as jx
    from jaxley.channels import Rate
except ImportError:
    jx = None
    Rate = None


M1_PARAM_BOUNDS = OrderedDict(load_batch_param_space(project_root()))
M1_PARAM_ORDER = tuple(M1_PARAM_BOUNDS.keys())
M1_DEFAULT_PARAMS = OrderedDict((name, default_param_values()[name]) for name in M1_PARAM_ORDER)


def dict_to_vector(params: Mapping[str, float]) -> np.ndarray:
    values = []
    for name in M1_PARAM_ORDER:
        if name not in params:
            raise KeyError(f"Missing parameter {name}")
        values.append(float(params[name]))
    return np.asarray(values, dtype=float)


def vector_to_dict(vector: Iterable[float]) -> OrderedDict[str, float]:
    values = np.asarray(list(vector), dtype=float)
    if values.shape != (len(M1_PARAM_ORDER),):
        raise ValueError(f"Expected vector shape {(len(M1_PARAM_ORDER),)}, got {values.shape}")
    return OrderedDict((name, float(values[index])) for index, name in enumerate(M1_PARAM_ORDER))


def _sigmoid(x, xp):
    return 1.0 / (1.0 + xp.exp(-x))


def bounded_from_unconstrained(vector: Iterable[float], xp=np) -> np.ndarray:
    unconstrained = xp.asarray(vector, dtype=float)
    bounds = xp.asarray(list(M1_PARAM_BOUNDS.values()), dtype=float)
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    sigma = _sigmoid(unconstrained, xp)
    return lows + sigma * (highs - lows)


def unconstrained_from_bounded(vector: Iterable[float], xp=np) -> np.ndarray:
    bounded = xp.asarray(vector, dtype=float)
    bounds = xp.asarray(list(M1_PARAM_BOUNDS.values()), dtype=float)
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    eps = 1e-6
    ratio = (bounded - lows) / (highs - lows)
    ratio = xp.clip(ratio, eps, 1.0 - eps)
    return xp.log(ratio) - xp.log1p(-ratio)


def inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


class M1:
    def __init__(self, root: Path | None = None, use_jaxley: bool = True) -> None:
        self.root = Path(root) if root is not None else project_root()
        self.defaults = m1_defaults()
        if self.defaults.add_in_vivo_thalamus:
            raise NotImplementedError("The Jaxley M1 path does not support addInVivoThalamus=True")
        self.feature_set: M1FeatureSet = build_feature_set(self.root, M1_PARAM_ORDER)
        self.population_specs = self.feature_set.population_specs
        self.population_names = tuple(spec.name for spec in self.population_specs)
        self.long_range_names = tuple(long_range_populations())
        self.target_arrays = build_target_arrays(tuple(asdict(spec) for spec in self.population_specs))
        self.bias = np.asarray(
            [inverse_softplus(spec.target["target"]) for spec in self.population_specs],
            dtype=float,
        )
        self.dt_ms = self.defaults.dt_ms
        self.duration_ms = self.defaults.duration_ms
        self.rate_windows = self.defaults.rate_windows
        self.tau_ms = DEFAULT_TAU_MS
        self.recurrent_input_scale = 0.01
        self.external_input_scale = 0.01
        self._recurrent_stack_np = np.stack([self.feature_set.recurrent_features[name] for name in M1_PARAM_ORDER], axis=0)
        self._external_stack_np = np.stack([self.feature_set.external_features[name] for name in M1_PARAM_ORDER], axis=0)
        self.jaxley_network = self._build_jaxley_network() if use_jaxley else None

    def _build_jaxley_network(self):
        if jx is None or Rate is None:
            return None
        cell = jx.Cell()
        cell.set("length", 1.0 / (2.0 * pi * 1e-5))
        cell.set("radius", 1.0)
        cell.insert(Rate())
        return jx.Network([cell for _ in range(len(self.population_names) + len(self.long_range_names))])

    def default_bounded_vector(self) -> np.ndarray:
        return dict_to_vector(M1_DEFAULT_PARAMS)

    def bounded_from_unconstrained(self, vector: Iterable[float], xp=np):
        return bounded_from_unconstrained(vector, xp=xp)

    def unconstrained_from_bounded(self, vector: Iterable[float], xp=np):
        return unconstrained_from_bounded(vector, xp=xp)

    def dict_to_vector(self, params: Mapping[str, float]) -> np.ndarray:
        return dict_to_vector(params)

    def vector_to_dict(self, vector: Iterable[float]) -> OrderedDict[str, float]:
        return vector_to_dict(vector)

    def _stack(self, xp):
        if xp is np:
            return self._recurrent_stack_np, self._external_stack_np, self.bias
        return xp.asarray(self._recurrent_stack_np), xp.asarray(self._external_stack_np), xp.asarray(self.bias)

    def _simulate_numpy(self, params: np.ndarray) -> np.ndarray:
        recurrent_stack, external_stack, bias = self._stack(np)
        recurrent = np.tensordot(params, recurrent_stack, axes=1)
        external = np.tensordot(params, external_stack, axes=1)
        state = bias.copy()
        traces = np.zeros((self.duration_ms, len(self.population_names)), dtype=float)
        alpha = self.dt_ms / self.tau_ms
        for step in range(self.duration_ms):
            rates = np.logaddexp(state, 0.0)
            traces[step] = rates
            drive = bias + self.external_input_scale * external + self.recurrent_input_scale * (recurrent @ rates)
            state = np.clip(state + alpha * (-state + drive), -50.0, 50.0)
        return traces

    def _simulate_jax(self, params):
        recurrent_stack, external_stack, bias = self._stack(jnp)
        recurrent = jnp.tensordot(params, recurrent_stack, axes=1)
        external = jnp.tensordot(params, external_stack, axes=1)
        alpha = self.dt_ms / self.tau_ms

        def step_fn(state, _):
            rates = jnp.logaddexp(state, 0.0)
            drive = bias + self.external_input_scale * external + self.recurrent_input_scale * (recurrent @ rates)
            next_state = jnp.clip(state + alpha * (-state + drive), -50.0, 50.0)
            return next_state, rates

        _, traces = jax.lax.scan(step_fn, bias, xs=None, length=self.duration_ms)
        return traces

    def simulate(self, bounded_params: Iterable[float], backend: str = "auto"):
        if backend == "jax" or (backend == "auto" and jax is not None):
            if jax is None or jnp is None:
                raise ImportError("jax is required for backend='jax'")
            params = jnp.asarray(bounded_params, dtype=float)
            if params.shape != (len(M1_PARAM_ORDER),):
                raise ValueError(f"Expected {len(M1_PARAM_ORDER)} params, got {params.shape}")
            return self._simulate_jax(params)
        params = np.asarray(list(bounded_params), dtype=float)
        if params.shape != (len(M1_PARAM_ORDER),):
            raise ValueError(f"Expected {len(M1_PARAM_ORDER)} params, got {params.shape}")
        return self._simulate_numpy(params)

    def windowed_pop_rates(self, bounded_params: Iterable[float], backend: str = "auto"):
        traces = self.simulate(bounded_params, backend=backend)
        if backend == "jax" or (backend == "auto" and jax is not None):
            windowed = compute_windowed_pop_rates(traces, self.rate_windows, dt_ms=self.dt_ms, xp=jnp)
            return windowed
        windowed = compute_windowed_pop_rates(traces, self.rate_windows, dt_ms=self.dt_ms, xp=np)
        return windowed

    def loss(self, bounded_params: Iterable[float], backend: str = "auto") -> float:
        windowed = self.windowed_pop_rates(bounded_params, backend=backend)
        targets, widths, mins = self.target_arrays
        xp = jnp if (backend == "jax" or (backend == "auto" and jax is not None)) and jax is not None else np
        return rate_fitness_tranges_jax(windowed, targets, widths, mins, max_fitness=1000.0, xp=xp)

    def pop_rates_dict(self, bounded_params: Iterable[float], backend: str = "auto") -> dict[str, dict[str, float]]:
        windowed = self.windowed_pop_rates(bounded_params, backend=backend)
        return windowed_rates_to_dict(self.population_names, np.asarray(windowed), self.rate_windows)
