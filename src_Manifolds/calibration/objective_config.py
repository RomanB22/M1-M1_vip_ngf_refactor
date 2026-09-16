"""Single source of truth for calibration objectives.

The dictionaries in this module are intentionally plain data.  They are used by
both the simulation process and the BatchTK launcher, so objective names cannot
silently drift between the two sides of the batch interface.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = MODULE_DIR.parent
PROJECT_ROOT = SOURCE_DIR.parent
MANIFOLD_DATA_ROOT = PROJECT_ROOT / "data" / "manifolds"
CEBRA_DATA_ROOT = MANIFOLD_DATA_ROOT / "cebra"
UMAP_DATA_ROOT = MANIFOLD_DATA_ROOT / "umap"

# These values are scientific inputs, not incidental simulator settings.  Keep
# them here so preprocessing, preflight validation, and run manifests use the
# same definitions as cfg.py.
MODEL_SEEDS = {
    "conn": 4321,
    "stim": 1234,
    "loc": 4321,
    "tvl_sampling": 1234,
    "m1_sampling": 4321,
    "cell": 1234,
}
CORTICAL_LAYER_BOUNDS = {
    "1": [0.0, 0.1],
    "2": [0.1, 0.29],
    "4": [0.29, 0.37],
    "5A": [0.37, 0.47],
    "5B": [0.47, 0.8],
    "6": [0.8, 1.0],
}
CORTICAL_SIZE_Y_UM = 1350.0


REGULAR_SPIKING_POPS = ["IT2", "IT4", "IT5A", "IT5B", "PT5B", "IT6", "CT6"]
INTERNEURON_POPS = [
    "NGF1",
    "PV2", "SOM2", "VIP2", "NGF2",
    "PV4", "SOM4", "VIP4", "NGF4",
    "PV5A", "SOM5A", "VIP5A", "NGF5A",
    "PV5B", "SOM5B", "VIP5B", "NGF5B",
    "PV6", "SOM6", "VIP6", "NGF6",
]

POPULATIONS_BY_LAYER_AND_CLASS = {
    "L1_interneurons": ["NGF1"],
    "L2_regular_spiking": ["IT2"],
    "L2_interneurons": ["PV2", "SOM2", "VIP2", "NGF2"],
    "L4_regular_spiking": ["IT4"],
    "L4_interneurons": ["PV4", "SOM4", "VIP4", "NGF4"],
    "L5A_regular_spiking": ["IT5A"],
    "L5A_interneurons": ["PV5A", "SOM5A", "VIP5A", "NGF5A"],
    "L5B_regular_spiking": ["IT5B", "PT5B"],
    "L5B_interneurons": ["PV5B", "SOM5B", "VIP5B", "NGF5B"],
    "L6_regular_spiking": ["IT6", "CT6"],
    "L6_interneurons": ["PV6", "SOM6", "VIP6", "NGF6"],
}

REGULAR_SPIKING_TARGET = {"target": 5.0, "width": 5.0, "min": 0.5}
INTERNEURON_TARGET = {"target": 10.0, "width": 15.0, "min": 0.25}


def _group(populations: list[str], target: dict[str, float]) -> dict[str, Any]:
    return {"populations": list(populations), **target}


def _population_groups() -> dict[str, dict[str, Any]]:
    groups = {pop: _group([pop], REGULAR_SPIKING_TARGET) for pop in REGULAR_SPIKING_POPS}
    groups.update({pop: _group([pop], INTERNEURON_TARGET) for pop in INTERNEURON_POPS})
    return groups


def _layer_class_groups() -> dict[str, dict[str, Any]]:
    groups = {}
    for name, populations in POPULATIONS_BY_LAYER_AND_CLASS.items():
        target = INTERNEURON_TARGET if name.endswith("interneurons") else REGULAR_SPIKING_TARGET
        groups[name] = _group(populations, target)
    return groups


RATE_CONFIG = {
    # Available schemes make the rate objective switchable without changing code.
    # "population" reproduces rateFitnessFuncTranges exactly.
    "scheme": "population",
    "max_fitness": 1000.0,
    "group_weighting": "cell_count",  # used only for multi-population groups
    "schemes": {
        "population": {"groups": _population_groups()},
        "cell_class": {
            "groups": {
                "regular_spiking": _group(REGULAR_SPIKING_POPS, REGULAR_SPIKING_TARGET),
                "interneurons": _group(INTERNEURON_POPS, INTERNEURON_TARGET),
            }
        },
        "layer_cell_class": {"groups": _layer_class_groups()},
    },
}


CSD_CONFIG = {
    "event": "tone_onset",  # resolved as cfg.duration - cfg.postTone
    "window_ms": [0.0, 200.0],
    "score_variant": "raw",  # recommended csd_quant return value; "preprocessed" is also available
    "reference": {
        "loader": "csd_quant_template",
        # Clone https://github.com/smcelroy97/csd_quant at the pinned revision here,
        # or set this to another checkout. The loader is isolated so a future
        # experimental NPZ can be selected with loader="npz".
        "backend_path": os.environ.get(
            "M1_CSD_QUANT_PATH", str(PROJECT_ROOT / "external" / "csd_quant")
        ),
        "backend_module": "csd_quant.wasserstein_dist",
        "revision": "676bf568bbcdd877e5a8d4f60c2a4b5788233020",
        "npz_path": None,
        "csd_key": "csd",
        "times_key": "times_ms",
    },
    "prepare_csd": {
        "minf": 0.05,
        "maxf": 300.0,
        "vaknin": True,
        "norm": False,
    },
}


# All scientific inputs to the manifold API live in this dictionary. Runtime
# paths are project-local under data/manifolds, independent of prototype trees.
MANIFOLD_CONFIG = {
    "method": "cebra",
    "comparator": "paired_rms",
    "event": "tone_onset",
    "window_ms": [-1000.0, 1500.0],
    "bin_ms": 20.0,
    "reference": {
        "format": "legacy_pickle",  # alternative: "npz"
        "activity_path": str(CEBRA_DATA_ROOT / "all_data.pkl"),
        "activity_key": "activity",
        "times_key": "times_ms",
        "depths_key": "cell_depths_um",
        "session": "230517_2759_1606VAL",
        "trial": 20,
        "trial_index_path": str(
            CEBRA_DATA_ROOT / "params" / "230517_2759_1606VAL_trial_idx.txt"
        ),
        "metadata_path": str(
            CEBRA_DATA_ROOT / "params" / "230517_2759_1606VAL_CEBRAparams.txt"
        ),
        "reference_start_ms": -1500.0,
        "already_smoothed": True,
        # These make model-capacity checks possible immediately after cell
        # creation, before the expensive simulation dynamics.  The reference
        # loader verifies that they still match the actual artifact.
        "feature_count": 91,
        "layer_feature_counts": {
            "1": 0,
            "2": 12,
            "4": 15,
            "5A": 7,
            "5B": 39,
            "6": 18,
        },
    },
    "preprocessing": {
        # Keep "none" for compatibility with the existing fitted CEBRA model.
        # "feature_zscore" is available for artifacts trained that way.
        "normalization": "none",
        "smoothing": {"enabled": True, "kernel_size": 25, "sigma": 2.5},
        "cell_selection": {
            "strategy": "layer_depth_order",
            "seed": 4321,
            "layer_order": ["1", "2", "4", "5A", "5B", "6"],
        },
    },
    "methods": {
        "cebra": {
            "model_path": str(CEBRA_DATA_ROOT / "models" / "model_stg-joy.pt")
        },
        "umap": {
            # The copied result files contain experimental activity and
            # embeddings, but not a fitted reducer with transform(). Keep
            # their paths explicit for offline reducer preparation.
            "model_path": str(UMAP_DATA_ROOT / "umap_reducer.joblib"),
            "source_data": {
                "parameters_path": str(UMAP_DATA_ROOT / "params.json"),
                "scaled_prep_results_path": str(
                    UMAP_DATA_ROOT / "scaled_prep" / "umap_results_n2_m1.pkl"
                ),
                "scaled_tone_results_path": str(
                    UMAP_DATA_ROOT / "scaled_tone" / "umap_results_n2_m1.pkl"
                ),
            },
        },
    },
    "shape_policy": {
        "require_equal_features": True,
        "require_equal_time_bins": True,
    },
}


OBJECTIVES = {
    "rate_loss": {
        "enabled": True,
        "kind": "population_rates",
        "direction": "minimize",
        "failure_value": 1.0e6,
        "config": RATE_CONFIG,
    },
    "csd_loss": {
        "enabled": True,
        "kind": "csd_wasserstein",
        "direction": "minimize",
        "failure_value": 1.0e6,
        "config": CSD_CONFIG,
    },
    "manifold_loss": {
        "enabled": True,
        "kind": "manifold",
        "direction": "minimize",
        "failure_value": 1.0e6,
        "config": MANIFOLD_CONFIG,
    },
}


def get_objectives() -> dict[str, dict[str, Any]]:
    """Return an isolated copy suitable for attaching to NetPyNE cfg."""

    return deepcopy(OBJECTIVES)


def get_active_objectives(
    objectives: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    specs = OBJECTIVES if objectives is None else objectives
    active = {name: spec for name, spec in specs.items() if spec.get("enabled", False)}
    if not active:
        raise ValueError("At least one optimization objective must be enabled")
    return active


def get_metrics(
    objectives: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    return {
        name: str(spec.get("direction", "minimize"))
        for name, spec in get_active_objectives(objectives).items()
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def objective_fingerprint(
    objectives: dict[str, dict[str, Any]] | None = None,
) -> str:
    active = get_active_objectives(objectives)
    payload = json.dumps(active, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
