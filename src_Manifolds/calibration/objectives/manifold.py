"""Manifold-model adapters and embedding comparators."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .activity import (
    REFERENCE_LOADERS,
    configured_layer_feature_counts,
    prepare_activity_pair,
    validate_model_feature_capacity,
)


@dataclass(frozen=True)
class ManifoldScore:
    value: float
    diagnostics: dict[str, Any]


def load_cebra_model(method_config: Mapping[str, Any]) -> Any:
    try:
        import cebra
    except ImportError as exc:
        raise ImportError("The CEBRA manifold method requires the 'cebra' package") from exc
    path = Path(str(method_config["model_path"])).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CEBRA model does not exist: {path}")
    return cebra.CEBRA.load(path)


def load_umap_model(method_config: Mapping[str, Any]) -> Any:
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("The UMAP manifold method requires the 'joblib' package") from exc
    path = Path(str(method_config["model_path"])).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"UMAP reducer does not exist: {path}")
    return joblib.load(path)


METHOD_LOADERS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "cebra": load_cebra_model,
    "umap": load_umap_model,
}


def transform_embedding(model: Any, activity: np.ndarray, label: str) -> np.ndarray:
    transform = getattr(model, "transform", None)
    if not callable(transform):
        raise TypeError("Manifold artifact must provide transform(activity)")
    embedding = np.asarray(transform(activity), dtype=float)
    if embedding.ndim != 2 or embedding.shape[0] != activity.shape[0]:
        raise ValueError(
            f"{label} embedding must have shape (time, latent dimensions); got {embedding.shape}"
        )
    if not np.isfinite(embedding).all():
        raise ValueError(f"{label} embedding contains non-finite values")
    return embedding


def load_reference_embedding(
    reference_config: Mapping[str, Any], method: str, time_bins: int
) -> np.ndarray | None:
    """Load an optional embedding produced by ``calibration.reference_preparation``."""

    raw_path = reference_config.get("embedding_path")
    if not raw_path:
        return None
    embedding_method = str(reference_config.get("embedding_method", method))
    if embedding_method != method:
        raise ValueError(
            f"Prepared embedding method {embedding_method!r} does not match {method!r}"
        )
    path = Path(str(raw_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Prepared manifold embedding does not exist: {path}")
    embedding = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if embedding.ndim != 2 or embedding.shape[0] != int(time_bins):
        raise ValueError(
            "Prepared manifold embedding must have one row per reference time bin; "
            f"got {embedding.shape}, expected {time_bins} rows"
        )
    if not np.isfinite(embedding).all():
        raise ValueError("Prepared manifold embedding contains non-finite values")
    return embedding


def paired_rms(reference: np.ndarray, model: np.ndarray) -> float:
    """RMS Euclidean separation between corresponding trajectory times."""

    if reference.shape != model.shape:
        raise ValueError(
            f"paired_rms requires equal embedding shapes, got {reference.shape} and {model.shape}"
        )
    return float(np.sqrt(np.mean(np.sum((reference - model) ** 2, axis=1))))


def procrustes_disparity(reference: np.ndarray, model: np.ndarray) -> float:
    try:
        from scipy.spatial import procrustes
    except ImportError as exc:
        raise ImportError("Procrustes comparison requires scipy") from exc
    if reference.shape != model.shape:
        raise ValueError("Procrustes comparison requires equal embedding shapes")
    _reference_aligned, _model_aligned, disparity = procrustes(reference, model)
    return float(disparity)


COMPARATORS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "paired_rms": paired_rms,
    "procrustes": procrustes_disparity,
}


def _validate_pair_shapes(
    reference: np.ndarray, model: np.ndarray, policy: Mapping[str, Any]
) -> None:
    if reference.ndim != 2 or model.ndim != 2:
        raise ValueError("Manifold inputs must both be 2D (time, features)")
    if policy.get("require_equal_time_bins", True) and reference.shape[0] != model.shape[0]:
        raise ValueError(
            f"Experimental/model time bins differ: {reference.shape[0]} versus {model.shape[0]}"
        )
    if policy.get("require_equal_features", True) and reference.shape[1] != model.shape[1]:
        raise ValueError(
            f"Experimental/model feature counts differ: {reference.shape[1]} versus {model.shape[1]}"
        )


def validate_manifold_setup(
    config: Mapping[str, Any], strict_files: bool = True, **_: Any
) -> None:
    method = str(config.get("method", "cebra"))
    if method not in METHOD_LOADERS:
        raise ValueError(f"Unknown manifold method {method!r}; choose one of {sorted(METHOD_LOADERS)}")
    comparator = str(config.get("comparator", "paired_rms"))
    if comparator not in COMPARATORS:
        raise ValueError(
            f"Unknown manifold comparator {comparator!r}; choose one of {sorted(COMPARATORS)}"
        )
    if float(config.get("bin_ms", 0)) <= 0:
        raise ValueError("Manifold bin_ms must be positive")
    window = np.asarray(config.get("window_ms", []), dtype=float)
    if window.shape != (2,) or window[0] >= window[1]:
        raise ValueError("Manifold window_ms must be [start, stop] with start < stop")
    reference = config.get("reference", {})
    reference_format = str(reference.get("format", "npz"))
    if reference_format not in REFERENCE_LOADERS:
        raise ValueError(f"Unknown manifold reference format {reference_format!r}")
    if method not in config.get("methods", {}):
        raise ValueError(f"No artifact configuration was supplied for manifold method {method!r}")
    selection = config.get("preprocessing", {}).get("cell_selection", {})
    layer_order = [str(value) for value in selection.get("layer_order", [])]
    if not layer_order:
        raise ValueError("Manifold cell selection must define a nonempty layer_order")
    configured_layer_feature_counts(config, layer_order)
    embedding_method = str(reference.get("embedding_method", method))
    if reference.get("embedding_path") and embedding_method != method:
        raise ValueError(
            f"Prepared embedding method {embedding_method!r} does not match {method!r}"
        )

    if strict_files:
        method_path = Path(str(config["methods"][method].get("model_path", ""))).expanduser()
        if not method_path.is_file():
            raise FileNotFoundError(f"{method.upper()} artifact does not exist: {method_path}")
        dependency = "cebra" if method == "cebra" else "joblib"
        if importlib.util.find_spec(dependency) is None:
            raise ImportError(f"Manifold method {method!r} requires the {dependency!r} package")
        required_reference_paths = [reference.get("activity_path")]
        if reference_format == "legacy_pickle":
            required_reference_paths += [reference.get("trial_index_path"), reference.get("metadata_path")]
        for raw_path in required_reference_paths:
            path = Path(str(raw_path or "")).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Manifold reference file does not exist: {path}")
        if reference.get("embedding_path"):
            embedding_path = Path(str(reference["embedding_path"])).expanduser()
            if not embedding_path.is_file():
                raise FileNotFoundError(
                    f"Prepared manifold embedding does not exist: {embedding_path}"
                )


def validate_manifold_capacity(sim: Any, cfg: Any, config: Mapping[str, Any]) -> None:
    validate_model_feature_capacity(sim, cfg, config)


def score_manifold_objective(context: Any, config: Mapping[str, Any]) -> ManifoldScore:
    method = str(config.get("method", "cebra"))
    comparator_name = str(config.get("comparator", "paired_rms"))
    experimental, model_activity, activity_diagnostics = prepare_activity_pair(context, config)
    _validate_pair_shapes(experimental, model_activity, config.get("shape_policy", {}))

    model_cache_key = f"manifold_model:{method}:{config['methods'][method].get('model_path', '')}"
    if model_cache_key not in context.cache:
        context.cache[model_cache_key] = METHOD_LOADERS[method](config["methods"][method])
    manifold_model = context.cache[model_cache_key]

    reference_config = config.get("reference", {})
    reference_embedding_key = (
        f"reference_embedding:{method}:"
        f"{reference_config.get('embedding_path') or reference_config.get('activity_path', '')}"
    )
    if reference_embedding_key not in context.cache:
        prepared_embedding = load_reference_embedding(
            config.get("reference", {}), method, experimental.shape[0]
        )
        context.cache[reference_embedding_key] = (
            prepared_embedding
            if prepared_embedding is not None
            else transform_embedding(manifold_model, experimental, "experimental")
        )
    reference_embedding = context.cache[reference_embedding_key]
    model_embedding = transform_embedding(manifold_model, model_activity, "model")
    value = COMPARATORS[comparator_name](reference_embedding, model_embedding)
    return ManifoldScore(
        value=value,
        diagnostics={
            **activity_diagnostics,
            "method": method,
            "comparator": comparator_name,
            "reference_embedding_source": (
                "prepared_artifact"
                if config.get("reference", {}).get("embedding_path")
                else "runtime_transform"
            ),
            "reference_embedding_shape": list(reference_embedding.shape),
            "model_embedding_shape": list(model_embedding.shape),
        },
    )


__all__ = [
    "COMPARATORS",
    "METHOD_LOADERS",
    "ManifoldScore",
    "load_reference_embedding",
    "paired_rms",
    "score_manifold_objective",
    "transform_embedding",
    "validate_manifold_capacity",
    "validate_manifold_setup",
]
