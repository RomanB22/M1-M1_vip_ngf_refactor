"""Shared activity-matrix preparation for manifold objectives.

Both CEBRA and UMAP consume matrices shaped ``(time bins, features)``.  This
module is the only place that knows how experimental trials and NetPyNE spikes
are converted to that contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from .csd import resolve_event_time_ms


@dataclass(frozen=True)
class ActivityReference:
    activity: np.ndarray
    times_ms: np.ndarray
    cell_depths_um: np.ndarray | None
    metadata: dict[str, Any]


def _validate_activity(activity: np.ndarray, times_ms: np.ndarray, label: str) -> None:
    if activity.ndim != 2:
        raise ValueError(f"{label} activity must be 2D (time, features), got {activity.shape}")
    if times_ms.ndim != 1 or activity.shape[0] != times_ms.size:
        raise ValueError(
            f"{label} activity has {activity.shape[0]} bins but {times_ms.size} time values"
        )
    if not activity.size or not np.isfinite(activity).all() or not np.isfinite(times_ms).all():
        raise ValueError(f"{label} activity and times must be nonempty and finite")
    if times_ms.size > 1 and np.any(np.diff(times_ms) <= 0):
        raise ValueError(f"{label} times_ms must be strictly increasing")


def _crop_reference(
    activity: np.ndarray,
    times_ms: np.ndarray,
    window_ms: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    start, stop = map(float, window_ms)
    keep = (times_ms >= start) & (times_ms < stop)
    if not np.any(keep):
        raise ValueError(f"No experimental activity lies in [{start}, {stop}) ms")
    return activity[keep], times_ms[keep]


def load_legacy_pickle_reference(
    reference: Mapping[str, Any], window_ms: Sequence[float], bin_ms: float
) -> ActivityReference:
    activity_path = Path(str(reference["activity_path"])).expanduser()
    trial_index_path = Path(str(reference["trial_index_path"])).expanduser()
    metadata_path = Path(str(reference["metadata_path"])).expanduser()
    for path in (activity_path, trial_index_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Manifold reference file does not exist: {path}")

    with activity_path.open("rb") as handle:
        data = pickle.load(handle)
    labels = list(data["session_labels"])
    session = str(reference["session"])
    try:
        session_index = labels.index(session)
    except ValueError as exc:
        raise KeyError(f"Session {session!r} is not present in {activity_path}") from exc

    with trial_index_path.open() as handle:
        trial_ids = np.asarray(json.load(handle))
    session_activity = np.asarray(data["neural"][session_index], dtype=float)
    if session_activity.ndim != 2 or trial_ids.size != session_activity.shape[0]:
        raise ValueError("Legacy trial index and neural activity lengths do not match")
    trial = reference["trial"]
    activity = session_activity[trial_ids == trial]
    if activity.shape[0] == 0:
        raise KeyError(f"Trial {trial!r} is not present for session {session!r}")

    with metadata_path.open() as handle:
        metadata = json.load(handle)
    metadata_bin_ms = float(metadata.get("dt", bin_ms))
    if not np.isclose(metadata_bin_ms, float(bin_ms)):
        raise ValueError(
            f"Experimental bin width is {metadata_bin_ms} ms, configured bin_ms is {bin_ms}"
        )
    depths = np.asarray(metadata.get("cell_depths", []), dtype=float)
    if depths.size < activity.shape[1]:
        raise ValueError(
            f"Only {depths.size} cell depths are available for {activity.shape[1]} features"
        )
    # Preserve the behavior of the existing CEBRA path when one metadata cell
    # is absent from the preprocessed activity matrix.
    if depths.size > activity.shape[1]:
        depths = np.sort(depths)[: activity.shape[1]]

    start_ms = float(reference.get("reference_start_ms", 0.0))
    times_ms = start_ms + np.arange(activity.shape[0], dtype=float) * float(bin_ms)
    activity, times_ms = _crop_reference(activity, times_ms, window_ms)
    _validate_activity(activity, times_ms, "experimental")
    return ActivityReference(
        activity=activity,
        times_ms=times_ms,
        cell_depths_um=depths,
        metadata={
            "format": "legacy_pickle",
            "session": session,
            "trial": trial,
            "source_shape": list(session_activity.shape),
        },
    )


def load_npz_activity_reference(
    reference: Mapping[str, Any], window_ms: Sequence[float], bin_ms: float
) -> ActivityReference:
    path = Path(str(reference["activity_path"])).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Manifold reference NPZ does not exist: {path}")
    activity_key = str(reference.get("activity_key", "activity"))
    times_key = str(reference.get("times_key", "times_ms"))
    depths_key = str(reference.get("depths_key", "cell_depths_um"))
    with np.load(path, allow_pickle=False) as data:
        if activity_key not in data or times_key not in data:
            raise KeyError(f"{path} must contain {activity_key!r} and {times_key!r}")
        activity = np.asarray(data[activity_key], dtype=float)
        times_ms = np.asarray(data[times_key], dtype=float)
        depths = np.asarray(data[depths_key], dtype=float) if depths_key in data else None
    _validate_activity(activity, times_ms, "experimental")
    if depths is not None and depths.size != activity.shape[1]:
        raise ValueError("cell_depths_um length must match the experimental feature count")
    activity, times_ms = _crop_reference(activity, times_ms, window_ms)
    return ActivityReference(
        activity=activity,
        times_ms=times_ms,
        cell_depths_um=depths,
        metadata={"format": "npz", "path": str(path)},
    )


REFERENCE_LOADERS = {
    "legacy_pickle": load_legacy_pickle_reference,
    "npz": load_npz_activity_reference,
}


def load_activity_reference(config: Mapping[str, Any]) -> ActivityReference:
    reference = config["reference"]
    format_name = str(reference.get("format", "npz"))
    try:
        loader = REFERENCE_LOADERS[format_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown manifold reference format {format_name!r}; choose one of {sorted(REFERENCE_LOADERS)}"
        ) from exc
    return loader(reference, config["window_ms"], float(config["bin_ms"]))


def _to_numpy(vector: Any, dtype: Any = float) -> np.ndarray:
    if hasattr(vector, "as_numpy"):
        return np.asarray(vector.as_numpy(), dtype=dtype)
    if hasattr(vector, "to_python"):
        return np.asarray(vector.to_python(), dtype=dtype)
    return np.asarray(vector, dtype=dtype)


def _layer_for_depth(depth_um: float, layer_bounds: Mapping[str, Sequence[float]], size_y: float) -> str | None:
    normalized = float(depth_um) / float(size_y)
    for layer, bounds in layer_bounds.items():
        if layer.startswith("long") or layer == "24":
            continue
        low, high = map(float, bounds)
        if low <= normalized < high or (np.isclose(normalized, high) and np.isclose(high, 1.0)):
            return layer
    return None


def layer_feature_counts(reference: ActivityReference, cfg: Any, layer_order: Sequence[str]) -> dict[str, int]:
    if reference.cell_depths_um is None:
        raise ValueError("layer_depth_order selection requires experimental cell_depths_um")
    counts = {str(layer): 0 for layer in layer_order}
    for depth in reference.cell_depths_um:
        layer = _layer_for_depth(float(depth), cfg.layer, float(cfg.sizeY))
        if layer in counts:
            counts[layer] += 1
    if sum(counts.values()) != reference.activity.shape[1]:
        raise ValueError(
            "Experimental depth-to-layer mapping did not account for every activity feature: "
            f"mapped {sum(counts.values())}, expected {reference.activity.shape[1]}"
        )
    return counts


def configured_layer_feature_counts(
    config: Mapping[str, Any], layer_order: Sequence[str]
) -> dict[str, int]:
    """Return the reference contract used for pre-dynamics capacity checks."""

    reference = config.get("reference", {})
    raw_counts = reference.get("layer_feature_counts")
    if not isinstance(raw_counts, Mapping):
        raise ValueError(
            "Manifold reference must define reference.layer_feature_counts "
            "for pre-dynamics model validation"
        )
    counts = {str(layer): int(raw_counts.get(str(layer), 0)) for layer in layer_order}
    if any(value < 0 for value in counts.values()):
        raise ValueError("Manifold layer feature counts cannot be negative")
    feature_count = int(reference.get("feature_count", -1))
    if feature_count <= 0 or sum(counts.values()) != feature_count:
        raise ValueError(
            "reference.feature_count must be positive and equal the sum of "
            "reference.layer_feature_counts"
        )
    unknown = set(map(str, raw_counts)).difference(map(str, layer_order))
    if unknown:
        raise ValueError(f"Layer feature counts contain layers outside layer_order: {sorted(unknown)}")
    return counts


def validate_reference_feature_contract(
    reference: ActivityReference, cfg: Any, config: Mapping[str, Any]
) -> dict[str, int]:
    """Verify that explicit reference metadata describes the loaded matrix."""

    selection = config.get("preprocessing", {}).get("cell_selection", {})
    layer_order = [str(value) for value in selection.get("layer_order", [])]
    expected = configured_layer_feature_counts(config, layer_order)
    actual = layer_feature_counts(reference, cfg, layer_order)
    if reference.activity.shape[1] != int(config["reference"]["feature_count"]):
        raise ValueError(
            "Loaded manifold feature count does not match reference.feature_count: "
            f"{reference.activity.shape[1]} versus {config['reference']['feature_count']}"
        )
    if actual != expected:
        raise ValueError(
            "Loaded manifold layer counts do not match reference.layer_feature_counts: "
            f"{actual} versus {expected}"
        )
    return actual


def validate_model_feature_capacity(sim: Any, cfg: Any, config: Mapping[str, Any]) -> dict[str, int]:
    """Fail before connection/dynamics when the network is too small.

    ``sim.net.cells`` is local to an MPI rank.  The all-reduce makes the check
    use global instantiated counts while avoiding a full cell gather.
    """

    selection = config.get("preprocessing", {}).get("cell_selection", {})
    layer_order = [str(value) for value in selection.get("layer_order", [])]
    required = configured_layer_feature_counts(config, layer_order)
    local = {layer: 0 for layer in layer_order}
    cells = getattr(sim.net, "cells", None)
    if cells is None:
        cells = getattr(sim.net, "allCells", [])
    for cell in cells:
        tags = cell.get("tags", {}) if isinstance(cell, Mapping) else getattr(cell, "tags", {})
        ynorm = tags.get("ynorm")
        if ynorm is None:
            continue
        for layer in layer_order:
            low, high = map(float, cfg.layer[layer])
            if low <= float(ynorm) < high or (
                np.isclose(float(ynorm), high) and np.isclose(high, 1.0)
            ):
                local[layer] += 1
                break

    pc = getattr(sim, "pc", None)
    available = {
        layer: int(pc.allreduce(count, 1)) if pc is not None else int(count)
        for layer, count in local.items()
    }
    shortages = {
        layer: {"required": required[layer], "available": available[layer]}
        for layer in layer_order
        if available[layer] < required[layer]
    }
    if shortages:
        raise ValueError(f"Instantiated model cannot satisfy manifold feature contract: {shortages}")
    return available


def select_model_gids(
    sim: Any,
    cfg: Any,
    reference: ActivityReference,
    selection: Mapping[str, Any],
) -> tuple[list[int], dict[str, int]]:
    strategy = str(selection.get("strategy", "layer_depth_order"))
    if strategy != "layer_depth_order":
        raise ValueError(f"Unknown cell-selection strategy {strategy!r}")
    layer_order = [str(value) for value in selection.get("layer_order", [])]
    counts = layer_feature_counts(reference, cfg, layer_order)
    cells_by_layer = {layer: [] for layer in layer_order}
    for cell in sim.net.allCells:
        tags = cell.get("tags", {}) if isinstance(cell, Mapping) else getattr(cell, "tags", {})
        gid = cell.get("gid") if isinstance(cell, Mapping) else getattr(cell, "gid", None)
        ynorm = tags.get("ynorm")
        if gid is None or ynorm is None:
            continue
        for layer in layer_order:
            low, high = map(float, cfg.layer[layer])
            if low <= float(ynorm) < high or (np.isclose(float(ynorm), high) and np.isclose(high, 1.0)):
                cells_by_layer[layer].append(int(gid))
                break

    rng = np.random.default_rng(int(selection.get("seed", 4321)))
    selected: list[int] = []
    for layer in layer_order:
        candidates = np.asarray(sorted(cells_by_layer[layer]), dtype=int)
        required = counts[layer]
        if required > candidates.size:
            raise ValueError(
                f"Layer {layer} needs {required} model cells but only {candidates.size} are available"
            )
        if required:
            picked = rng.choice(candidates, size=required, replace=False)
            selected.extend(sorted(int(value) for value in picked))
    if len(selected) != reference.activity.shape[1]:
        raise ValueError("Selected model feature count does not match the experimental matrix")
    return selected, counts


def bin_model_activity(
    sim: Any,
    cfg: Any,
    selected_gids: Sequence[int],
    relative_times_ms: np.ndarray,
    bin_ms: float,
    event: str | float | int,
) -> np.ndarray:
    """Bin spikes on the exact time grid used by the experimental reference."""

    spike_times = _to_numpy(sim.allSimData["spkt"], dtype=float)
    spike_ids = _to_numpy(sim.allSimData["spkid"], dtype=int)
    if spike_times.shape != spike_ids.shape:
        raise ValueError("NetPyNE spkt and spkid arrays must have equal lengths")
    event_ms = resolve_event_time_ms(cfg, event)
    activity = np.zeros((relative_times_ms.size, len(selected_gids)), dtype=float)
    for feature, gid in enumerate(selected_gids):
        gid_spikes = spike_times[spike_ids == gid]
        starts = event_ms + relative_times_ms
        stops = starts + float(bin_ms)
        left = np.searchsorted(gid_spikes, starts, side="left")
        right = np.searchsorted(gid_spikes, stops, side="left")
        activity[:, feature] = right - left
    return activity


def gaussian_smooth(activity: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("smoothing kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("smoothing sigma must be positive")
    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=float)
    kernel = np.exp(-(x**2) / (2.0 * float(sigma) ** 2))
    kernel /= kernel.sum()
    def convolve_same_length(column: np.ndarray) -> np.ndarray:
        full = np.convolve(column, kernel, mode="full")
        start = (kernel.size - 1) // 2
        return full[start : start + column.size]

    return np.apply_along_axis(convolve_same_length, 0, activity)


def normalize_activity(activity: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return np.asarray(activity, dtype=float)
    if method == "feature_zscore":
        mean = activity.mean(axis=0, keepdims=True)
        std = activity.std(axis=0, keepdims=True)
        return (activity - mean) / np.where(std > 0, std, 1.0)
    raise ValueError(f"Unknown activity normalization method {method!r}")


def prepare_activity_pair(context: Any, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference_config = config.get("reference", {})
    reference_cache_key = "activity_reference:" + ":".join(
        str(reference_config.get(key, ""))
        for key in ("format", "activity_path", "session", "trial")
    )
    if reference_cache_key not in context.cache:
        context.cache[reference_cache_key] = load_activity_reference(config)
    reference = context.cache[reference_cache_key]
    validate_reference_feature_contract(reference, context.cfg, config)
    preprocessing = config.get("preprocessing", {})
    selection = preprocessing.get("cell_selection", {})
    selected_gids, layer_counts = select_model_gids(
        context.sim, context.cfg, reference, selection
    )
    model = bin_model_activity(
        context.sim,
        context.cfg,
        selected_gids,
        reference.times_ms,
        float(config["bin_ms"]),
        config.get("event", "tone_onset"),
    )
    smoothing = preprocessing.get("smoothing", {})
    if smoothing.get("enabled", False):
        model = gaussian_smooth(
            model,
            int(smoothing.get("kernel_size", 25)),
            float(smoothing.get("sigma", 2.5)),
        )
    experimental = np.asarray(reference.activity, dtype=float)
    if not config["reference"].get("already_smoothed", False) and smoothing.get("enabled", False):
        experimental = gaussian_smooth(
            experimental,
            int(smoothing.get("kernel_size", 25)),
            float(smoothing.get("sigma", 2.5)),
        )
    normalization = str(preprocessing.get("normalization", "none"))
    experimental = normalize_activity(experimental, normalization)
    model = normalize_activity(model, normalization)
    _validate_activity(experimental, reference.times_ms, "experimental")
    _validate_activity(model, reference.times_ms, "model")
    return experimental, model, {
        "reference": reference.metadata,
        "times_ms": [float(reference.times_ms[0]), float(reference.times_ms[-1])],
        "selected_gids": selected_gids,
        "layer_feature_counts": layer_counts,
        "normalization": normalization,
        "activity_shape": list(experimental.shape),
    }


__all__ = [
    "ActivityReference",
    "bin_model_activity",
    "configured_layer_feature_counts",
    "gaussian_smooth",
    "layer_feature_counts",
    "load_activity_reference",
    "normalize_activity",
    "prepare_activity_pair",
    "select_model_gids",
    "validate_model_feature_capacity",
    "validate_reference_feature_contract",
]
