# Multi-objective M1 calibration

One NetPyNE simulation now returns three independently minimized objectives:

- `rate_loss`: population, cell-class, or layer/cell-class firing rates;
- `csd_loss`: the raw `csd_quant` template Wasserstein distance;
- `manifold_loss`: a CEBRA embedding trajectory distance by default.

Enable, disable, and configure objectives only in
`calibration/objective_config.py`. `init.py` and the BatchTK launcher derive
their metric names from that same dictionary. Invalid/blocked trials receive
each objective's finite failure value, so they cannot look useful on the
Pareto front.

## Directory map

```text
src_Manifolds/
├── cfg.py                         simulation parameter definitions only
├── netParams.py                   standard NetPyNE network specification
├── init.py                        short simulation + BatchTK entry point
├── calibration/
│   ├── objective_config.py        enabled objectives and scientific inputs
│   ├── objective_pipeline.py      validation and trial scoring
│   ├── reference_preparation.py   offline manifold-reference command
│   └── objectives/
│       ├── firing_rates.py        configurable rate-group scoring
│       ├── csd.py                 NetPyNE/CSD-quant comparison
│       ├── activity.py            activity matrix preparation
│       └── manifold.py            CEBRA/UMAP adapters and comparison
├── optimization/
│   ├── batch_runner.py            primary multi-objective launcher
│   ├── run_mocma.py               MO-CMA compatibility launcher
│   ├── cluster_backends.py        local, Slurm, and SGE execution settings
│   ├── search_space.py            optimized parameter bounds
│   ├── channelopathy_parameters.py named parameter presets
│   ├── sampler_factory.py         MO-CMA selection and NSGA-II fallback
│   ├── run_manifest.py            provenance and resume compatibility
│   ├── pareto_front.py            Pareto reconstruction/export command
│   └── grid_search.py             named-parameter seed grid search
├── simulation/
│   ├── runtime_config.py          BatchTK/env overrides and derived settings
│   ├── analysis_config.py         manual recording/plot configuration
│   ├── mutation_presets.py        optional mutation recipes
│   ├── network_helpers.py         network/data-loading helpers
│   └── spike_guard.py             blockade detection and diagnostics
├── requirements-optimization.txt optimization-only dependencies
└── MULTI_OBJECTIVE_REFACTOR_PLAN.md design decisions and implementation status
```

`cfg.py` performs no filesystem access, environment parsing, data loading, or
plot setup. Those runtime actions are explicitly applied by
`simulation/runtime_config.py` after BatchTK mappings are available and before
`netParams.py` constructs the network.

## NewBatchtk setup

The extra packages used by optimization and `csd_quant` are listed in
`requirements-optimization.txt`:

```bash
conda run -n NewBatchtk python -m pip install \
  -r src_Manifolds/requirements-optimization.txt
```

`csd_quant` has no package metadata, so keep it as an external pinned checkout:

```bash
mkdir -p external
git clone https://github.com/smcelroy97/csd_quant external/csd_quant
git -C external/csd_quant checkout 676bf568bbcdd877e5a8d4f60c2a4b5788233020
```

The default CSD loader uses the template owned by that checkout. Set
`M1_CSD_QUANT_PATH` to test another checkout. Changing later to a project CSD
file only requires selecting the `npz` loader and its keys in `CSD_CONFIG`.

## Objectives

### Firing rates

Set `RATE_CONFIG["scheme"]` to one of:

- `population`: exact behavior of the former `rateFitnessFuncTranges`;
- `cell_class`: regular-spiking cells versus interneurons;
- `layer_cell_class`: regular-spiking/interneuron groups within each layer.

Multi-population groups use cell-count weighting by default. Targets and group
membership are ordinary dictionaries and can be changed without scorer edits.

### Manifold

All inputs are configured through `MANIFOLD_CONFIG`. The shared API constructs
experimental and model matrices shaped `(time bins, neural features)`, checks
their shapes, and passes both through the selected fixed learner.

With CEBRA, the adapter loads the configured `.pt` model and transforms both
matrices into the same latent coordinate system. With UMAP, it loads a fitted
reducer and calls the same `transform` interface. In either case, the default
`paired_rms` comparator measures separation between event-aligned time points;
the learner and comparison rule are independent and can be changed separately.

The manifold artifacts are self-contained under `data/manifolds/`. The active
CEBRA configuration reads its activity, session metadata, trial index, and
fitted model from `data/manifolds/cebra/`; it no longer depends on the old
prototype tree. The copied UMAP parameters and experimental result datasets
live under `data/manifolds/umap/`.

The legacy UMAP result pickles contain activity matrices and embeddings, but
not a reusable fitted reducer exposing `transform()`. Consequently, switching
`MANIFOLD_CONFIG["method"]` to `"umap"` also requires preparing
`data/manifolds/umap/umap_reducer.joblib`. Its intended path and the source-data
paths are isolated in `MANIFOLD_CONFIG["methods"]["umap"]`.

Prepare and validate that compact NPZ outside an optimization run with:

```bash
PYTHONPATH="$PWD/src_Manifolds" \
conda run -n NewBatchtk python -m calibration.reference_preparation \
  --output-dir data/manifolds/prepared
```

The command loads the configured trial, validates the explicit feature and
per-layer counts, checks the fixed CEBRA/UMAP transform, and writes
`experimental_activity.npz`, the experimental embedding, and
`reference_metadata.json` with checksums. It refuses to overwrite existing
artifacts unless `--force` is supplied. Copy the emitted
`runtime_reference_config` into `MANIFOLD_CONFIG["reference"]` when promoting
the compact artifact to the optimization input.

`reference.feature_count` and `reference.layer_feature_counts` are deliberate
parts of the API. Immediately after cells are instantiated, every run checks
that the global network has enough cells in each layer; a too-small test or
production network stops before connections and simulation dynamics.

## Small smoke test

Production keeps `cfg.singleCellPops=False`. For local testing, this command
sets it to true and creates seven cells per population, enough to provide the
current reference's 91 layer-matched CEBRA features:

```bash
PYTHONPATH="$PWD:$PWD/src_Manifolds" \
M1_SINGLE_CELL_POPS=1 \
M1_TEST_CELLS_PER_POP=7 \
M1_TEST_DT_MS=0.05 \
M1_SMOKE_OBJECTIVES=rate_loss,manifold_loss \
conda run -n NewBatchtk nrniv -nogui -python src_Manifolds/init.py
```

`M1_TEST_CELLS_PER_POP` is configurable. Seven is the current minimum uniform
value because layer 5B needs 39 features across six cortical populations.
CoreNEURON is disabled only for this local smoke mode. `M1_SMOKE_OBJECTIVES`
can restrict a direct smoke run without changing production configuration;
omit it to exercise all three objectives. `M1_TEST_DT_MS` can override the
integration step for a quick mechanics test, but defaults to the production
value.

Run the fast objective tests with:

```bash
conda run -n NewBatchtk python -m pytest -q tests
```

## Local NetPyNE plots

Standalone simulations save NetPyNE plots by default under
`cfg.saveFolder` (`batchData/v103_manualTune` by default). The configured plots
are the spike raster, recorded voltage traces, LFP, dipole, CSD, and EEG.
The simulation data and configuration are also saved as JSON by default through
`cfg.saveJson = True`.
Set `cfg.plotSimResults = False` to disable them, or `cfg.showPlots = True` when
running with a graphical Matplotlib backend to display them interactively. The
entry point defaults to the headless `Agg` backend unless `MPLBACKEND` is set.

BatchTK trials automatically disable plotting to avoid producing figures for
every optimization evaluation. Set `M1_PLOT_RESULTS=1` explicitly if a selected
BatchTK trial should also save its plots.

## Multi-objective search

Both launchers below use BatchTK's `optuna_search`. They request OptunaHub
MO-CMA and automatically select Optuna NSGA-II if MO-CMA cannot be loaded or
constructed:

```bash
# Slurm default; requests MO-CMA and falls back to NSGA-II
PYTHONPATH="$PWD/src_Manifolds" \
conda run -n NewBatchtk python -m optimization.batch_runner

# Explicit MO-CMA compatibility entry point; SGE default
PYTHONPATH="$PWD/src_Manifolds" \
conda run -n NewBatchtk python -m optimization.run_mocma
```

Useful environment controls are `M1_BATCH_BACKEND` (`local`, `slurm`, `sge`),
`M1_NUM_TRIALS`, `M1_NUM_WORKERS`, `M1_POPULATION_SIZE`, and
`M1_OPTUNA_SAMPLER` (`mocma` or `nsgaii`).

Each run directory under `optimization/` contains:

- `manifest.json`, including the complete objective/search configuration,
  actual sampler, Git/source revision, seeds, package versions, pinned
  `csd_quant` revision, and reference/model checksums;
- the Optuna journal;
- `trials.csv`, with stable objective column names;
- `pareto_front.csv` and `pareto_front.json`;
- `pareto_objectives.png`, showing all trials and non-dominated solutions.

Pareto products can be rebuilt at any time from the journal and are checked
against Optuna's own `study.best_trials`:

```bash
PYTHONPATH="$PWD/src_Manifolds" \
conda run -n NewBatchtk python -m optimization.pareto_front \
  optimization/<run-directory>
```

Resume validation rejects changes to objectives, parameter bounds, sampler,
source tree, dependency versions, or reference checksums. Start a new run
directory after any of those scientific inputs changes.
