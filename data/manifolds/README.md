# Manifold artifacts

This directory owns the experimental manifold inputs used by
`src_Manifolds/calibration/objective_config.py`. Runtime calibration must not
depend on the old prototype source tree.

## CEBRA

`cebra/` contains:

- `all_data.pkl`: experimental activity for all available sessions;
- `params/`: session metadata and trial-index files;
- `models/model_stg-joy.pt`: the model used by the default manifold objective;
- `models/model_joy.pt`: the companion fitted model retained with the source
  dataset.

The active configuration selects session `230517_2759_1606VAL`, trial 20, and
`model_stg-joy.pt`.

## UMAP

`umap/` contains the original parameters and the `scaled_prep` and
`scaled_tone` experimental results. Each result pickle contains representations,
activity matrices, session names, task progress, and cell depths.

Those result files do not contain a fitted UMAP estimator with a `transform()`
method. The runtime UMAP adapter intentionally expects a fixed reducer at
`umap/umap_reducer.joblib`; prepare and validate that artifact before selecting
UMAP as the active method. Optimization trials must transform through a fixed
reducer rather than fitting a new coordinate system per candidate.
