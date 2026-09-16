"""Current-source-density objective and replaceable reference loaders.

The objective depends on ``csd_quant`` for the distance calculation, but the
reference-loading decision is kept here.  Replacing the bundled template with
another experimental dataset therefore does not change the simulation code.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from optimization.run_manifest import git_revision


@dataclass(frozen=True)
class CSDScore:
    value: float
    diagnostics: dict[str, Any]


def resolve_event_time_ms(cfg: Any, event: str | float | int) -> float:
    """Resolve a named event to an absolute simulation time in milliseconds."""

    if isinstance(event, (float, int)):
        return float(event)
    if event == "tone_onset":
        return float(cfg.duration) - float(cfg.postTone)
    raise ValueError(f"Unknown event {event!r}; use 'tone_onset' or an absolute time")


def _load_csd_quant(reference: Mapping[str, Any]) -> Any:
    module_name = str(reference.get("backend_module", "csd_quant.wasserstein_dist"))
    backend_path = reference.get("backend_path")
    if backend_path:
        checkout = Path(str(backend_path)).expanduser().resolve()
        if checkout.exists():
            # Support both a package checkout (parent on sys.path) and the
            # repository's direct-module layout (checkout itself on sys.path).
            for import_root in (checkout.parent, checkout):
                root = str(import_root)
                if root not in sys.path:
                    sys.path.insert(0, root)
    try:
        return importlib.import_module(module_name)
    except Exception as package_error:
        # The upstream repository has historically also been usable as a
        # direct-module checkout without packaging metadata.
        try:
            return importlib.import_module(module_name.rsplit(".", 1)[-1])
        except Exception as exc:
            exc.__context__ = package_error
        location = f" at {backend_path}" if backend_path else ""
        raise ImportError(
            "Cannot load the csd_quant scoring backend"
            f"{location}. Clone https://github.com/smcelroy97/csd_quant to the "
            "configured backend_path at the pinned revision, and install its "
            "dependencies (including POT and h5py)."
        ) from exc


def load_csd_quant_template(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Load the template owned by csd_quant.

    The returned object is deliberately opaque to the scorer.  This function
    is the single seam to replace when the template-loading API changes.
    """

    backend = _load_csd_quant(reference)
    if not callable(getattr(backend, "wd_from_template", None)):
        raise AttributeError("csd_quant backend must define wd_from_template")
    return {"backend": backend, "loader": "csd_quant_template"}


def load_npz_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Load a future explicit experimental CSD reference from a simple NPZ."""

    path = Path(str(reference.get("npz_path", ""))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSD reference NPZ does not exist: {path}")
    csd_key = str(reference.get("csd_key", "csd"))
    times_key = str(reference.get("times_key", "times_ms"))
    with np.load(path, allow_pickle=False) as data:
        if csd_key not in data or times_key not in data:
            raise KeyError(f"{path} must contain {csd_key!r} and {times_key!r}")
        csd = np.asarray(data[csd_key], dtype=float)
        times_ms = np.asarray(data[times_key], dtype=float)
    _validate_csd_matrix(csd, times_ms, label="reference")
    return {
        "backend": _load_csd_quant(reference),
        "loader": "npz",
        "csd": csd,
        "times_ms": times_ms,
        "path": str(path),
    }


REFERENCE_LOADERS = {
    "csd_quant_template": load_csd_quant_template,
    "npz": load_npz_reference,
}


def load_csd_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    loader_name = str(reference.get("loader", "csd_quant_template"))
    try:
        loader = REFERENCE_LOADERS[loader_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown CSD reference loader {loader_name!r}; choose one of {sorted(REFERENCE_LOADERS)}"
        ) from exc
    return loader(reference)


def _validate_csd_matrix(csd: np.ndarray, times_ms: np.ndarray, label: str) -> None:
    if csd.ndim != 2:
        raise ValueError(f"{label} CSD must be 2D (depth, time), got {csd.shape}")
    if times_ms.ndim != 1 or csd.shape[1] != times_ms.size:
        raise ValueError(
            f"{label} CSD time dimension {csd.shape[1]} does not match "
            f"times_ms length {times_ms.size}"
        )
    if times_ms.size < 2 or not np.isfinite(csd).all() or not np.isfinite(times_ms).all():
        raise ValueError(f"{label} CSD and times must be finite and contain at least two samples")
    if np.any(np.diff(times_ms) <= 0):
        raise ValueError(f"{label} times_ms must be strictly increasing")


def validate_csd_setup(
    config: Mapping[str, Any], strict_files: bool = True, **_: Any
) -> None:
    window = np.asarray(config.get("window_ms", []), dtype=float)
    if window.shape != (2,) or window[0] >= window[1]:
        raise ValueError("CSD window_ms must be [start, stop] with start < stop")
    # wd_from_template currently owns this exact scoring interval.  Be explicit
    # instead of silently presenting a configurable setting that is ignored.
    if str(config.get("reference", {}).get("loader", "csd_quant_template")) == "csd_quant_template":
        if not np.allclose(window, [0.0, 200.0]):
            raise ValueError("csd_quant template scoring currently requires window_ms=[0, 200]")
    variant = str(config.get("score_variant", "raw"))
    if variant not in {"raw", "preprocessed"}:
        raise ValueError("CSD score_variant must be 'raw' or 'preprocessed'")
    reference = config.get("reference", {})
    loader_name = str(reference.get("loader", "csd_quant_template"))
    if loader_name not in REFERENCE_LOADERS:
        raise ValueError(f"Unknown CSD reference loader {loader_name!r}")
    if strict_files:
        backend_path = Path(str(reference.get("backend_path", ""))).expanduser()
        expected_revision = reference.get("revision")
        checkout_revision = git_revision(backend_path) if backend_path.is_dir() else None
        if expected_revision and checkout_revision and checkout_revision != expected_revision:
            raise RuntimeError(
                "csd_quant checkout revision does not match the configured revision: "
                f"{checkout_revision} versus {expected_revision}"
            )
        load_csd_reference(reference)


def _simulation_csd(context: Any, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    params = dict(config.get("prepare_csd", {}))
    prepared = context.sim.analysis.prepareCSD(
        sim=context.sim,
        timeRange=None,
        saveData=False,
        getAllData=True,
        **params,
    )
    if prepared is None or len(prepared) != 5:
        raise RuntimeError("NetPyNE prepareCSD did not return (CSD, LFP, sampr, spacing, dt)")
    csd, _lfp, _sampr, _spacing_um, dt_ms = prepared
    csd = np.asarray(csd, dtype=float)
    event_ms = resolve_event_time_ms(context.cfg, config.get("event", "tone_onset"))
    times_ms = np.arange(csd.shape[1], dtype=float) * float(dt_ms) - event_ms
    _validate_csd_matrix(csd, times_ms, label="simulation")
    return csd, times_ms


def _score_against_npz(
    reference: Mapping[str, Any], sim_csd: np.ndarray, sim_times_ms: np.ndarray, variant: str
) -> tuple[float, float]:
    backend = reference["backend"]
    ref = backend.poststimulus_csd(reference["csd"], reference["times_ms"])
    model = backend.poststimulus_csd(sim_csd, sim_times_ms)
    raw = float(backend.wasserstein_csd(ref, model, interpolate=True, sp_len=30, t_len=200))
    pp = float(
        backend.wasserstein_csd(
            backend.preprocess_csd(ref),
            backend.preprocess_csd(model),
            interpolate=True,
            sp_len=30,
            t_len=200,
        )
    )
    return raw, pp


def score_csd_objective(context: Any, config: Mapping[str, Any]) -> CSDScore:
    if "csd_reference" not in context.cache:
        context.cache["csd_reference"] = load_csd_reference(config["reference"])
    if "simulation_csd" not in context.cache:
        context.cache["simulation_csd"] = _simulation_csd(context, config)

    reference = context.cache["csd_reference"]
    sim_csd, sim_times_ms = context.cache["simulation_csd"]
    if reference["loader"] == "csd_quant_template":
        raw, preprocessed = reference["backend"].wd_from_template(sim_csd, sim_times_ms)
        raw, preprocessed = float(raw), float(preprocessed)
    else:
        raw, preprocessed = _score_against_npz(
            reference,
            sim_csd,
            sim_times_ms,
            str(config.get("score_variant", "raw")),
        )

    variant = str(config.get("score_variant", "raw"))
    value = raw if variant == "raw" else preprocessed
    return CSDScore(
        value=value,
        diagnostics={
            "score_variant": variant,
            "raw_distance": raw,
            "preprocessed_distance": preprocessed,
            "reference_loader": reference["loader"],
            "simulation_shape": list(sim_csd.shape),
            "event_time_ms": resolve_event_time_ms(context.cfg, config.get("event", "tone_onset")),
        },
    )


__all__ = [
    "CSDScore",
    "REFERENCE_LOADERS",
    "load_csd_reference",
    "resolve_event_time_ms",
    "score_csd_objective",
    "validate_csd_setup",
]
