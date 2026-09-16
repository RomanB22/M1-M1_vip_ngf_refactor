"""Prepare a compact, validated manifold reference outside optimization.

The default command converts the configured legacy CEBRA input to NPZ,
validates the fitted learner, and stores the experimental embedding once::

    python -m calibration.reference_preparation
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from calibration.objectives.activity import (
    gaussian_smooth,
    layer_feature_counts,
    load_activity_reference,
    normalize_activity,
)
from calibration.objectives.manifold import METHOD_LOADERS, transform_embedding
from calibration.objective_config import (
    CORTICAL_LAYER_BOUNDS,
    CORTICAL_SIZE_Y_UM,
    MANIFOLD_CONFIG,
)
from optimization.run_manifest import sha256_file, software_versions


def _ensure_writable(paths: list[Path], force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing manifold artifacts: {existing}; use --force"
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def prepare_manifold_reference(
    config: Mapping[str, Any],
    output_dir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Write compact activity, embedding, and provenance metadata artifacts."""

    configured = deepcopy(dict(config))
    method = str(configured.get("method", "cebra"))
    output = Path(output_dir).expanduser().resolve()
    activity_path = output / "experimental_activity.npz"
    embedding_path = output / f"{method}_embedding.npy"
    metadata_path = output / "reference_metadata.json"
    _ensure_writable([activity_path, embedding_path, metadata_path], force)

    reference = load_activity_reference(configured)
    preprocessing = configured.get("preprocessing", {})
    smoothing = preprocessing.get("smoothing", {})
    stored_activity = np.asarray(reference.activity, dtype=float)
    smoothing_applied = False
    if not configured["reference"].get("already_smoothed", False) and smoothing.get(
        "enabled", False
    ):
        stored_activity = gaussian_smooth(
            stored_activity,
            int(smoothing.get("kernel_size", 25)),
            float(smoothing.get("sigma", 2.5)),
        )
        smoothing_applied = True

    cfg_contract = SimpleNamespace(
        layer=CORTICAL_LAYER_BOUNDS,
        sizeY=CORTICAL_SIZE_Y_UM,
    )
    layer_order = [
        str(value)
        for value in preprocessing.get("cell_selection", {}).get("layer_order", [])
    ]
    counts = layer_feature_counts(reference, cfg_contract, layer_order)
    configured_count = int(configured["reference"].get("feature_count", -1))
    configured_layers = {
        str(key): int(value)
        for key, value in configured["reference"].get("layer_feature_counts", {}).items()
    }
    if stored_activity.shape[1] != configured_count or counts != configured_layers:
        raise ValueError(
            "Loaded reference disagrees with configured feature contract: "
            f"shape={stored_activity.shape}, layers={counts}, "
            f"configured_features={configured_count}, configured_layers={configured_layers}"
        )

    arrays: dict[str, np.ndarray] = {
        "activity": stored_activity,
        "times_ms": np.asarray(reference.times_ms, dtype=float),
    }
    if reference.cell_depths_um is not None:
        arrays["cell_depths_um"] = np.asarray(reference.cell_depths_um, dtype=float)
    normalized = normalize_activity(
        stored_activity, str(preprocessing.get("normalization", "none"))
    )
    model = METHOD_LOADERS[method](configured["methods"][method])
    embedding = transform_embedding(model, normalized, "experimental")
    # Do not leave a prepared activity artifact behind when model validation
    # fails.  All expensive/read-only validation happens before these writes.
    np.savez_compressed(activity_path, **arrays)
    np.save(embedding_path, embedding, allow_pickle=False)

    source_paths = {
        key: Path(str(configured["reference"][key])).expanduser().resolve()
        for key in ("activity_path", "trial_index_path", "metadata_path")
        if configured["reference"].get(key)
    }
    model_path = Path(str(configured["methods"][method]["model_path"])).expanduser().resolve()
    runtime_reference = {
        "format": "npz",
        "activity_path": str(activity_path),
        "activity_key": "activity",
        "times_key": "times_ms",
        "depths_key": "cell_depths_um",
        "already_smoothed": bool(
            configured["reference"].get("already_smoothed", False) or smoothing_applied
        ),
        "embedding_path": str(embedding_path),
        "embedding_method": method,
        "metadata_path": str(metadata_path),
        "feature_count": int(stored_activity.shape[1]),
        "layer_feature_counts": counts,
    }
    metadata = {
        "format_version": 1,
        "method": method,
        "activity_shape": list(stored_activity.shape),
        "embedding_shape": list(embedding.shape),
        "times_ms": [float(reference.times_ms[0]), float(reference.times_ms[-1])],
        "bin_ms": float(configured["bin_ms"]),
        "window_ms": list(map(float, configured["window_ms"])),
        "preprocessing": preprocessing,
        "smoothing_applied_during_preparation": smoothing_applied,
        "layer_feature_counts": counts,
        "runtime_reference_config": runtime_reference,
        "source_artifacts": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in source_paths.items()
        },
        "model_artifact": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "prepared_artifacts": {
            "activity": {"path": str(activity_path), "sha256": sha256_file(activity_path)},
            "embedding": {"path": str(embedding_path), "sha256": sha256_file(embedding_path)},
        },
        "software_versions": software_versions(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/calibration/manifolds",
        help="Directory for experimental_activity.npz and derived metadata",
    )
    parser.add_argument("--method", choices=sorted(METHOD_LOADERS), default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = deepcopy(MANIFOLD_CONFIG)
    if args.method is not None:
        config["method"] = args.method
    metadata = prepare_manifold_reference(config, args.output_dir, force=args.force)
    print(
        f"Prepared {metadata['method']} reference with activity "
        f"{tuple(metadata['activity_shape'])} and embedding {tuple(metadata['embedding_shape'])}"
    )


if __name__ == "__main__":
    main()


__all__ = ["prepare_manifold_reference"]
