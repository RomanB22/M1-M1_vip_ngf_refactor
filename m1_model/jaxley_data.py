from __future__ import annotations

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, Iterable, Mapping
import pickle

import numpy as np


INH_BIN_RANGES = ((0.0, 0.37), (0.37, 0.8), (0.8, 1.0))
EE_TARGET = {"target": 5.0, "width": 5.0, "min": 0.5}
II_TARGET = {"target": 10.0, "width": 15.0, "min": 0.25}
DEFAULT_DT_MS = 1.0
DEFAULT_TAU_MS = 20.0
DEFAULT_EIGAIN = 1.0
DEFAULT_IEGAIN = 1.0
DEFAULT_IIGAIN = 1.0
DEFAULT_DURATION_MS = 2000
DEFAULT_RATE_WINDOWS = ((1000, 1250), (1250, 1500), (1500, 1750), (1750, 2000))


@dataclass(frozen=True)
class PopulationSpec:
    name: str
    cell_type: str
    ynorm: tuple[float, float]
    family: str
    layer_bin: int
    target_bin: int
    target: Mapping[str, float]

    @property
    def midpoint(self) -> float:
        return float((self.ynorm[0] + self.ynorm[1]) / 2.0)


@dataclass(frozen=True)
class M1Defaults:
    duration_ms: int
    dt_ms: float
    tau_ms: float
    rate_windows: tuple[tuple[int, int], ...]
    long_range_rate_ranges: Mapping[str, tuple[float, float]]
    long_range_rate_midpoints: Mapping[str, float]
    add_in_vivo_thalamus: bool


@dataclass(frozen=True)
class M1FeatureSet:
    population_specs: tuple[PopulationSpec, ...]
    population_index: Mapping[str, int]
    recurrent_features: Mapping[str, np.ndarray]
    external_features: Mapping[str, np.ndarray]
    feature_scales: Mapping[str, float]
    defaults: M1Defaults


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def load_conn_data(root: Path | None = None) -> dict:
    root = root or project_root()
    return _load_pickle(root / "conn" / "conn.pkl")


def load_conn_long_data(root: Path | None = None) -> dict:
    root = root or project_root()
    return _load_pickle(root / "conn" / "conn_long.pkl")


def load_batch_param_space(root: Path | None = None, percentage_change: float = 0.5) -> dict[str, tuple[float, float]]:
    root = root or project_root()
    batch_params_path = root / "src_test" / "batch_params.py"
    spec = spec_from_file_location("m1_batch_params", batch_params_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load batch params from {batch_params_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    min_chg = 1.0 - percentage_change
    max_chg = 1.0 + percentage_change
    return module.get_batch_params(min_chg, max_chg)


def default_param_values() -> dict[str, float]:
    values = {"weightLong.TPO": 0.5, "weightLong.TVL": 0.5, "weightLong.S1": 0.5, "weightLong.S2": 0.5}
    values.update({"weightLong.cM1": 0.5, "weightLong.M2": 0.5, "weightLong.OC": 0.5})
    values["EEGain"] = 1.0
    values["IEweights.0"] = 1.0
    values["IEweights.1"] = 1.0
    values["IEweights.2"] = 1.0
    values["IIweights.0"] = 1.0
    values["IIweights.1"] = 1.0
    values["IIweights.2"] = 1.0
    values["EICellTypeGain.PV"] = 1.0
    values["EICellTypeGain.SOM"] = 1.0
    values["EICellTypeGain.VIP"] = 1.0
    values["EICellTypeGain.NGF"] = 1.0
    return values


def m1_defaults() -> M1Defaults:
    long_ranges = {
        "TPO": (0.0, 2.5),
        "TVL": (0.0, 2.5),
        "S1": (0.0, 2.5),
        "S2": (0.0, 2.5),
        "cM1": (0.0, 2.5),
        "M2": (0.0, 2.5),
        "OC": (0.0, 2.5),
    }
    midpoints = {name: float(sum(bounds) / 2.0) for name, bounds in long_ranges.items()}
    return M1Defaults(
        duration_ms=DEFAULT_DURATION_MS,
        dt_ms=DEFAULT_DT_MS,
        tau_ms=DEFAULT_TAU_MS,
        rate_windows=DEFAULT_RATE_WINDOWS,
        long_range_rate_ranges=long_ranges,
        long_range_rate_midpoints=midpoints,
        add_in_vivo_thalamus=False,
    )


def cortical_population_specs() -> tuple[PopulationSpec, ...]:
    def make(name: str, cell_type: str, ynorm: tuple[float, float]) -> PopulationSpec:
        family = "exc" if cell_type in {"IT", "PT", "CT"} else "inh"
        layer_bin = find_bin_index((ynorm[0] + ynorm[1]) / 2.0, INH_BIN_RANGES)
        target = EE_TARGET if family == "exc" else II_TARGET
        return PopulationSpec(
            name=name,
            cell_type=cell_type,
            ynorm=ynorm,
            family=family,
            layer_bin=layer_bin,
            target_bin=layer_bin,
            target=target,
        )

    layer = {
        "1": (0.0, 0.1),
        "2": (0.1, 0.29),
        "4": (0.29, 0.37),
        "5A": (0.37, 0.47),
        "5B": (0.47, 0.8),
        "6": (0.8, 1.0),
    }
    return (
        make("NGF1", "NGF", layer["1"]),
        make("IT2", "IT", layer["2"]),
        make("PV2", "PV", layer["2"]),
        make("SOM2", "SOM", layer["2"]),
        make("VIP2", "VIP", layer["2"]),
        make("NGF2", "NGF", layer["2"]),
        make("IT4", "IT", layer["4"]),
        make("PV4", "PV", layer["4"]),
        make("SOM4", "SOM", layer["4"]),
        make("VIP4", "VIP", layer["4"]),
        make("NGF4", "NGF", layer["4"]),
        make("IT5A", "IT", layer["5A"]),
        make("PV5A", "PV", layer["5A"]),
        make("SOM5A", "SOM", layer["5A"]),
        make("VIP5A", "VIP", layer["5A"]),
        make("NGF5A", "NGF", layer["5A"]),
        make("IT5B", "IT", layer["5B"]),
        make("PT5B", "PT", layer["5B"]),
        make("PV5B", "PV", layer["5B"]),
        make("SOM5B", "SOM", layer["5B"]),
        make("VIP5B", "VIP", layer["5B"]),
        make("NGF5B", "NGF", layer["5B"]),
        make("IT6", "IT", layer["6"]),
        make("CT6", "CT", layer["6"]),
        make("PV6", "PV", layer["6"]),
        make("SOM6", "SOM", layer["6"]),
        make("VIP6", "VIP", layer["6"]),
        make("NGF6", "NGF", layer["6"]),
    )


def long_range_populations() -> tuple[str, ...]:
    return ("TPO", "TVL", "S1", "S2", "cM1", "M2", "OC")


def find_bin_index(value: float, bins: Iterable[Iterable[float]]) -> int:
    bins = tuple((float(low), float(high)) for low, high in bins)
    for index, (low, high) in enumerate(bins):
        if low <= value < high:
            return index
    if np.isclose(value, bins[-1][1]):
        return len(bins) - 1
    raise ValueError(f"{value} does not fall within bins {bins}")


def population_index_map(specs: Iterable[PopulationSpec]) -> dict[str, int]:
    return {spec.name: index for index, spec in enumerate(specs)}


def _population_midpoint(spec: PopulationSpec) -> float:
    return float((spec.ynorm[0] + spec.ynorm[1]) / 2.0)


def _ee_rule(spec: PopulationSpec) -> tuple[tuple[str, str, str], str, tuple[str, ...]] | None:
    if spec.name in {"IT2", "IT4"}:
        return (("W+AS_norm", "IT", "L2/3,4"), "W", ("IT",))
    if spec.name in {"IT5A", "IT5B"}:
        return (("W+AS_norm", "IT", "L5A,5B"), "AS", ("IT",))
    if spec.name == "PT5B":
        return (("W+AS_norm", "PT", "L5B"), "AS", ("IT", "PT"))
    if spec.name == "IT6":
        return (("W+AS_norm", "IT", "L6"), "W", ("IT", "CT"))
    if spec.name == "CT6":
        return (("W+AS_norm", "CT", "L6"), "W", ("IT", "CT"))
    return None


def _normalize_feature(feature: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(np.max(np.abs(feature)))
    if scale < 1.0:
        scale = 1.0
    return feature / scale, scale


def build_feature_set(root: Path | None = None, param_order: Iterable[str] | None = None) -> M1FeatureSet:
    root = root or project_root()
    conn = load_conn_data(root)
    conn_long = load_conn_long_data(root)
    defaults = m1_defaults()
    specs = cortical_population_specs()
    pop_index = population_index_map(specs)
    if param_order is None:
        param_order = tuple(load_batch_param_space(root).keys())
    param_order = tuple(param_order)

    recurrent_features = {name: np.zeros((len(specs), len(specs)), dtype=float) for name in param_order}
    external_features = {name: np.zeros((len(specs),), dtype=float) for name in param_order}

    for pre_idx, pre_spec in enumerate(specs):
        for post_idx, post_spec in enumerate(specs):
            if pre_spec.family == "exc" and post_spec.family == "exc":
                ee_rule = _ee_rule(post_spec)
                if ee_rule is None:
                    continue
                pmat_key, pre_bin_label, allowed_pre_types = ee_rule
                if pre_spec.cell_type not in allowed_pre_types:
                    continue
                pre_bin = find_bin_index(_population_midpoint(pre_spec), conn["bins"][pre_bin_label])
                post_bin = find_bin_index(_population_midpoint(post_spec), conn["bins"][("W+AS", pmat_key[1], pmat_key[2])])
                recurrent_features["EEGain"][post_idx, pre_idx] = (
                    float(conn["pmat"][pmat_key][post_bin, pre_bin]) * float(conn["wmat"][pmat_key][post_bin, pre_bin])
                )
                continue

            if pre_spec.family == "exc" and post_spec.family == "inh":
                param_name = f"EICellTypeGain.{post_spec.cell_type}"
                recurrent_features[param_name][post_idx, pre_idx] = (
                    float(conn["pmat"][("E", post_spec.cell_type)][post_spec.layer_bin, pre_spec.layer_bin])
                    * float(conn["wmat"][("E", post_spec.cell_type)][post_spec.layer_bin, pre_spec.layer_bin])
                    * DEFAULT_EIGAIN
                )
                continue

            if pre_spec.family == "inh" and post_spec.family == "exc":
                param_name = f"IEweights.{post_spec.target_bin}"
                recurrent_features[param_name][post_idx, pre_idx] = (
                    float(conn["pmat"][(pre_spec.cell_type, "E")][post_spec.target_bin, pre_spec.layer_bin]) * DEFAULT_IEGAIN
                )
                continue

            if pre_spec.family == "inh" and post_spec.family == "inh":
                if pre_spec.layer_bin != post_spec.layer_bin:
                    continue
                param_name = f"IIweights.{pre_spec.layer_bin}"
                recurrent_features[param_name][post_idx, pre_idx] = (
                    float(conn["pmat"][(pre_spec.cell_type, post_spec.cell_type)]) * DEFAULT_IIGAIN
                )

    for long_name in long_range_populations():
        param_name = f"weightLong.{long_name}"
        rate_midpoint = defaults.long_range_rate_midpoints[long_name]
        for post_idx, post_spec in enumerate(specs):
            bins = conn_long["bins"][(long_name, post_spec.cell_type)]
            bin_index = find_bin_index(post_spec.midpoint, bins)
            exc_values = conn_long["cmat"][(long_name, post_spec.cell_type, "exc")]
            inh_values = conn_long["cmat"][(long_name, post_spec.cell_type, "inh")]
            bin_index = min(bin_index, len(exc_values) - 1, len(inh_values) - 1)
            convergence_exc = float(exc_values[bin_index])
            convergence_inh = float(inh_values[bin_index])
            external_features[param_name][post_idx] = rate_midpoint * (convergence_exc - convergence_inh)

    normalized_recurrent: Dict[str, np.ndarray] = {}
    normalized_external: Dict[str, np.ndarray] = {}
    feature_scales: Dict[str, float] = {}
    for name in param_order:
        recurrent_norm, recurrent_scale = _normalize_feature(recurrent_features[name])
        external_norm, external_scale = _normalize_feature(external_features[name])
        normalized_recurrent[name] = recurrent_norm
        normalized_external[name] = external_norm
        feature_scales[name] = max(recurrent_scale, external_scale)

    return M1FeatureSet(
        population_specs=specs,
        population_index=pop_index,
        recurrent_features=normalized_recurrent,
        external_features=normalized_external,
        feature_scales=feature_scales,
        defaults=defaults,
    )
