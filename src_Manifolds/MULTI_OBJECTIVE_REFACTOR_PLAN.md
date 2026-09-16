# Multi-objective calibration refactor plan

Status: **implementation and local integration complete**. The decisions below
were confirmed on 2026-09-16. The remaining acceptance step is a short
production/cluster run with all three objectives, including the computationally
expensive full-size `csd_quant` transport calculation.

## 1. Goal

Refactor `src_Manifolds/` so one NetPyNE simulation can emit an ordered, configurable set of independent calibration objectives:

1. population firing-rate loss;
2. model-versus-experimental CSD loss;
3. model-versus-experimental manifold loss, initially with UMAP and CEBRA backends.

BatchTK/Optuna minimizes the enabled objectives directly, without combining
them into a weighted scalar. Each run persists enough information to
reconstruct, export, and plot the Pareto front.

The design priorities are:

- one simulation path, rather than separate copies for rate, UMAP, and CEBRA calibration;
- small, pure scoring functions that can be unit tested without running NEURON;
- a single source of truth for enabled objectives and BatchTK metric names;
- deterministic preprocessing and neuron sampling;
- no expensive plotting, model fitting, or repeated reference-data preprocessing inside an optimization trial;
- clear failure behavior so invalid simulations cannot appear as good Pareto solutions.

## 2. Confirmed decisions

1. **Condition/event:** `SimulateBaseline=False`; tone onset is resolved as
   `cfg.duration - cfg.postTone`, and CSD scores `[0, 200)` ms relative to it.
2. **CSD reference:** use the `.npz` template bundled with the pinned
   `csd_quant` checkout. Reference loading is a registry so a later NPZ API is
   configuration-only.
3. **CSD score:** optimize raw `wd_from_template`; keep its preprocessed score
   in diagnostics.
4. **Manifold inputs:** session, trial, window, binning, smoothing, selection,
   and artifact paths are a dictionary passed to the shared manifold API.
5. **Comparator:** use event-time-paired RMS trajectory distance, isolated from
   the learner; Procrustes remains an alternative.
6. **Default learner:** enable one manifold objective and use CEBRA. UMAP is an
   alternate method, not a simultaneous default objective.
7. **Feature matching:** use deterministic layer/depth-count sampling for now,
   with strict matrix/embedding shape validation and an isolated selection API.
8. **Optimizer:** request OptunaHub MO-CMA and fall back to NSGA-II if its
   dependency, registry package, or constructor is unavailable.

## 3. Verified starting point

### Current project flow

- `init.py` runs and gathers the network, calculates `popAvgRates`, calls `defs.rateFitnessFuncTranges`, adds the spike-guard penalty, and sends one scalar named `loss`.
- `optunaBatch.py` and `cmaesBatch.py` both register only `{'loss': 'minimize'}`.
- `cfg.py` already records LFP at six contacts and enables NetPyNE CSD plotting. The objective should use the numerical CSD array, not a saved plot.
- `ClusterConfigs.py` and `gridSearch.py` still contain batch commands that point to `src_test/init.py`; the SGE command also activates `M1_dev` instead of `NewBatchtk`.
- Earlier UMAP and CEBRA prototypes exist in the ignored `M1-M1_vip_ngf_refactor-Codex_test/` tree. They provide useful preprocessing ideas, but duplicate the simulator, fit/score differently, write plots during trials, and return separate scalar losses.
- The only current automated coverage related to this folder is the spike-guard test suite.

### Authoritative runtime

The requested `NewBatchtk` environment currently contains:

| Package | Version |
|---|---:|
| BatchTK | 0.1.6 |
| NetPyNE | 1.1.1 |
| Optuna | 4.5.0 |
| cmaes | 0.12.0 |
| umap-learn | 0.5.9.post2 |
| CEBRA | 0.5.0 |
| POT | 0.9.6.post1 |
| scikit-learn | 1.7.2 |

BatchTK 0.1.6 behavior in this environment is important:

- `optuna_search` preserves the order of the supplied `metrics` dictionary, retrieves every named result, and passes all directions to `optuna.create_study`.
- `cmaes_search` explicitly selects only the first metric and is therefore single-objective.
- The multi-objective runner uses BatchTK `optuna_search`. It registers
  OptunaHub's `MoCmaSampler` in BatchTK's sampler registry and uses NSGA-II as
  a pre-study fallback. BatchTK's `cmaes_search` remains unsuitable because it
  selects only the first metric.

### `csd_quant` interface

The inspected upstream revision is `676bf568bbcdd877e5a8d4f60c2a4b5788233020`.

- `wasserstein_dist.wd_from_template(sim_csd, sim_times_ms)` validates and resamples the post-event `[0, 200)` ms interval and returns `(raw_distance, legacy_preprocessed_distance)`.
- `utils.wasserstein_csd(csd_a, csd_b, ...)` calculates the sum of 2-D Wasserstein distances for sinks and sources.
- Inputs use shape `(depth, time)`.
- The repository is not currently a Python package and has no package metadata.
  Its `utils.py` imports `h5py` at module import time. `h5py`, `optunahub`, and
  `pytest` were installed in `NewBatchtk` during implementation and are listed
  in `requirements-optimization.txt`.

## 4. Target architecture

The runtime flow should be:

```text
NetPyNE simulation and gather (rank 0)
                  |
          shared result context
       (rates, spikes, LFP, cfg)
                  |
       enabled-objective registry
        /          |           \
  rate loss     CSD loss    manifold loss
        \          |           /
       flat scalar result dictionary
                  |
       one sim.send(...) operation
                  |
 BatchTK Optuna / MO-CMA (NSGA-II fallback)
                  |
       journal + trials dataframe
                  |
      Pareto CSV/JSON/figure export
```

Use simple registries of callables rather than a large inheritance framework. The two extension points are:

- an **objective registry**, mapping an objective `kind` to a scoring function;
- a **manifold-method registry**, mapping `umap` or `cebra` to an artifact loader/transform function.

Adding a new objective should require one scorer plus one configuration entry. Adding a new manifold learner should require one adapter plus a registry entry, without changing `init.py` or the BatchTK runner.

## 5. Implemented file layout

```text
src_Manifolds/
├── init.py                         # short NetPyNE + BatchTK entry point
├── cfg.py                          # declarative simulation variables only
├── netParams.py                    # standard NetPyNE network specification
├── calibration/
│   ├── objective_config.py         # ordered objective specifications
│   ├── objective_pipeline.py       # validation and trial scoring
│   ├── reference_preparation.py    # offline compact-reference command
│   └── objectives/
│       ├── firing_rates.py         # pure firing-rate loss
│       ├── csd.py                  # NetPyNE CSD + csd_quant adapter
│       ├── activity.py             # deterministic sampling/binning
│       └── manifold.py             # learner adapters and comparator
├── optimization/
│   ├── batch_runner.py             # primary multi-objective runner
│   ├── run_mocma.py                # MO-CMA compatibility launcher
│   ├── cluster_backends.py         # local, Slurm, and SGE settings
│   ├── search_space.py             # parameter bounds
│   ├── channelopathy_parameters.py # named parameter presets
│   ├── sampler_factory.py          # MO-CMA registration + fallback
│   ├── run_manifest.py             # provenance and resume validation
│   ├── pareto_front.py             # journal reconstruction/export
│   └── grid_search.py              # fixed-parameter seed grid
├── simulation/
│   ├── runtime_config.py           # mappings, env overrides, derived data
│   ├── analysis_config.py          # manual recording/plot definitions
│   ├── mutation_presets.py         # optional mutation recipes
│   ├── network_helpers.py          # network/data helpers
│   └── spike_guard.py              # blockade detector and diagnostics
└── README.md                       # setup, contracts, and commands
```

The former ambiguous `defs.py` was renamed to
`simulation/network_helpers.py`. Runtime behavior was removed from `cfg.py`;
the file now contains imports and parameter assignments only.

## 6. One source of truth for objectives

Create an ordered plain-Python configuration in
`calibration/objective_config.py`. A representative shape is:

```python
OBJECTIVES = {
    "rate_loss": {
        "enabled": True,
        "kind": "population_rates",
        "direction": "minimize",
    },
    "csd_loss": {
        "enabled": True,
        "kind": "csd_wasserstein",
        "direction": "minimize",
        "event_time_cfg_key": "preTone",
        "window_ms": [0.0, 200.0],
        "score_variant": "raw",
    },
    "umap_manifold_loss": {
        "enabled": True,
        "kind": "manifold",
        "direction": "minimize",
        "method": "umap",
        "comparator": "paired_rms",
    },
    "cebra_manifold_loss": {
        "enabled": False,
        "kind": "manifold",
        "direction": "minimize",
        "method": "cebra",
        "comparator": "paired_rms",
    },
}
```

Exact paths and scientific preprocessing parameters also live in these entries
or in small referenced metadata files. Helper functions return:

- the active ordered objective specifications;
- the exact BatchTK `metrics` dictionary;
- a stable fingerprint of objective names, order, directions, and scientific configuration.

Both `init.py` and `optimization/batch_runner.py` use these helpers. There is
no separately maintained metric list.

Changing the active objective set or its order requires a new study. Write a run manifest before starting and refuse to resume a journal when its stored objective fingerprint differs.

## 7. Objective contracts and execution

Define a small result record:

```python
ObjectiveResult(value: float, diagnostics: dict)
```

Each scorer receives a shared context plus its specification and returns one
finite scalar and JSON-safe diagnostics. `calibration/objective_pipeline.py`:

1. validate all enabled specs and artifact paths before the expensive simulation starts;
2. construct a lazy shared context after `sim.gatherData()`;
3. run enabled scorers in configuration order on rank 0;
4. reject non-finite outputs;
5. apply the common invalid-simulation policy;
6. flatten the optimized metrics into the top level of the result payload;
7. call `sim.send` exactly once.

The context should cache derived data within a trial, so CSD and binned activity are each calculated at most once even if more than one objective uses them.

### Invalid simulations and spike guard

The current spike-guard penalty affects only the rate loss. In a multi-objective study, a blocked model could still appear non-dominated because of a good CSD or manifold score.

Implemented policy:

- preserve the current raw rate calculation and spike-guard diagnostics;
- if spike guard marks a trial invalid, or a runtime scorer fails for trial-specific numerical reasons, assign a configurable finite failure value (provisionally `1e6`) to **every** optimized objective;
- include raw objective values, `trial_valid`, `failure_reason`, and spike-guard summary as diagnostics;
- treat missing files, incompatible shapes, and invalid static configuration as preflight errors rather than running an expensive penalized trial.

This approximates a hard feasibility constraint without changing BatchTK internals and keeps invalid points off the useful Pareto front.

## 8. Population firing-rate objective

Preserve the exact current `rateFitnessFuncTranges` behavior:

- excitatory populations: target `5`, width `5`, minimum rate `0.5`;
- inhibitory populations: target `10`, width `15`, minimum rate `0.25`;
- maximum per-population penalty `1000`;
- evaluate over `cfg.printPopAvgRates`;
- average first across time ranges per population, then across populations.

Implementation steps:

1. Move the population target specification out of `init.py` into `calibration/objective_config.py`.
2. Make the mathematical scorer in `calibration/objectives/firing_rates.py` accept ordinary dictionaries and return both the scalar and per-population diagnostics.
3. Add a regression test proving equality with the existing function on fixed sample inputs before removing or aliasing the old definition.
4. Calculate `sim.analysis.popAvgRates` only when the rate objective or a diagnostic explicitly requires it.

## 9. CSD objective

### Dependency strategy

Pin `csd_quant` to the inspected commit as an external checkout under
`external/csd_quant/`, rather than copying its functions into project code.
The strict preflight verifies the checkout revision when Git metadata is
available. The missing runtime dependency (`h5py`) is part of the documented
`NewBatchtk` setup, and each run manifest records both expected and detected
upstream revisions plus the template checksum.

Because the upstream repository currently has no packaging metadata or explicit license file, confirm the intended vendoring/distribution arrangement before committing third-party source.

### Model CSD extraction

1. Keep `cfg.recordLFP` enabled whenever the CSD objective is active.
2. On rank 0, call NetPyNE 1.1.1 `sim.analysis.prepareCSD` using the gathered `sim.allSimData['LFP']`.
3. Calculate/filter the continuous CSD before event-window extraction to avoid filter-edge artifacts at 0 and 200 ms.
4. Build uniform model times relative to the configured event, with CSD shaped `(depth, time)`.
5. Validate contact order, cortical-depth direction, sign convention, complete time coverage, and finite values.

### Scoring

For the bundled reference, call upstream `wd_from_template(model_csd, model_times_ms)` directly. Optimize the configured element of its return tuple and retain both values in diagnostics.

For a project-specific target, load a versioned `.npz` containing at least:

```text
csd       float array, shape (depth, time)
times_ms  float array, shape (time,), relative to the event
```

Perform the same `[0, 200)` ms validation/resampling, then call upstream `wasserstein_csd`. Do not silently transpose, reverse depth, stretch an incomplete time window, or infer an event time.

## 10. Manifold objective

### Shared input contract

Both UMAP and CEBRA adapters should consume a common activity matrix:

```text
activity: time bins x neural features
times_ms: one value per time bin
```

The experimental reference metadata must define:

- session/trial selection;
- event-relative analysis window;
- bin width and smoothing parameters;
- normalization method;
- feature count and feature ordering;
- per-layer cell counts;
- random seed;
- fitted learner version and artifact checksum.

Model preprocessing deterministically constructs the configured features, bins
spikes into the identical time grid, and applies the recorded preprocessing.
Under the selected neuron-level strategy, it samples the same number of cells
per layer and orders them by layer/depth/GID. An instantiated model that cannot
supply the required cells fails before connections and simulation dynamics.

Do not select neurons based on whether they fired in the candidate simulation: that makes the observed feature set depend on the parameters being optimized. Use a fixed deterministic selection rule.

### Reference preparation outside optimization

`calibration/reference_preparation.py` is an explicit offline command. It:

1. loads and validates experimental activity;
2. applies and records preprocessing;
3. validates a fixed fitted UMAP reducer or trained CEBRA model;
4. transforms and stores the experimental embedding once;
5. saves metadata containing dimensions, versions, preprocessing,
   feature/layer counts, runtime reference configuration, and checksums.

The tool deliberately does not fit a learner: fitting policy remains a
separate scientific decision. Runtime trials load the stored experimental
embedding when configured and transform only model activity.

Optimization trials only load the fitted artifact and call `transform` for model activity. They must not fit UMAP/CEBRA or produce figures.

### Method adapters

- **UMAP:** load a fixed, seeded reducer fitted on the experimental reference; transform model activity expressed in the approved common feature space. This replaces the prototype's per-candidate joint fit, which is less reproducible and makes the coordinate system change between trials.
- **CEBRA:** load a fixed `.pt` model trained on the experimental reference; verify both input dimension and feature-space metadata, then transform model activity.

The manifold-method registry returns embeddings only. A separate comparator registry computes the scalar loss. Initial comparators should be limited to the selected primary metric plus one well-defined alternative; avoid carrying every exploratory metric into the production path.

The runtime diagnostics should include method, comparator, embedding shape, aligned time-bin count, and optional non-optimized comparison values. Plot embeddings only in a separate analysis command for selected trials/Pareto points.

## 11. BatchTK multi-objective runner

Implemented in `optimization/batch_runner.py`:

1. activate the objective configuration and obtain `metrics` from it;
2. request OptunaHub `MoCmaSampler(popsize=..., seed=...)`, register it in
   BatchTK 0.1.6, and fall back to `nsgaii` before study creation on failure;
3. keep the ordered metric names stable;
4. use a study label and output directory specific to that objective fingerprint;
5. write a manifest before starting and validate it on resume;
6. save the returned trials dataframe as CSV after the run.

Use `algo_kwargs={'seed': 42}` rather than BatchTK's separate `seed=` argument because the installed 0.1.6 study-name code concatenates that argument as a string and is unsafe for an integer.

`optimization/cluster_backends.py` ensures every optimization submission:

- activates `NewBatchtk`;
- adds the project root and `src_Manifolds` to `PYTHONPATH`;
- executes `src_Manifolds/init.py` rather than `src_test/init.py`;
- keeps local, Slurm, and SGE selection explicit and centralized.

`optimization/run_mocma.py` is a compatibility entry point that calls the same
multi-objective runner while explicitly requesting MO-CMA. It never calls
BatchTK's single-objective `cmaes_search`.

## 12. Pareto-front access

`optimization/pareto_front.py` renames Optuna's ordered `values_N` columns to the configured metric
names and computes the finite, completed non-dominated set in every direction.
It can also reopen BatchTK's Optuna journal from a run directory, rebuild all
outputs after an interrupted/completed run, and assert that independently
calculated trial numbers equal Optuna's `study.best_trials`.

Produce:

- `trials.csv`: state, parameters, and every named objective;
- `pareto_front.csv`: completed non-dominated trials with parameters and raw objective values;
- `pareto_front.json`: the same data for scripts/notebooks;
- a static Matplotlib figure:
  - two objectives: labeled 2-D scatter;
  - three objectives: pairwise projections;
  - more than three objectives: pairwise objective matrix with Pareto points highlighted.

The export must preserve raw objective units. Optional normalized axes may be used only for visualization and must be labeled. Do not automatically choose one "best" solution or hide the tradeoff behind a weighted sum.

The pure non-dominated-mask function is tested against known dominated, tied,
and failed points.

## 13. Efficiency and operational cleanup

1. Add an optimization-mode flag and skip `sim.analysis.plotData()` and unnecessary `sim.saveData()` work during batch trials. Keep plotting available for manual validation runs.
2. Load experimental arrays and fitted manifold artifacts only on rank 0 and only for enabled objectives.
3. Validate objective files and metadata before building/running the full network.
4. Save full plots/embeddings/CSD arrays only for explicitly selected trials after optimization; the per-trial BatchTK payload should remain small.
5. Use `pathlib.Path` anchored to the repository/module location rather than assuming the launch working directory.
6. Convert NumPy values to normal Python scalars/lists at the payload boundary.
7. Record package versions, Git revision, a hash of the complete local Python
   source tree (including untracked work), objective fingerprint, full active
   objective/search configuration, random seeds, reference checksums, and
   `csd_quant` revision in the run manifest. Resume validation includes these
   scientific inputs.

## 14. Implementation phases and acceptance criteria

### Phase A — Configuration and rate regression — complete

- Add the ordered objective configuration, pipeline, and rate scorer.
- Change the emitted metric from generic `loss` to `rate_loss`.
- Update BatchTK and cluster paths to use `NewBatchtk` and `src_Manifolds`.
- The original rate objective was isolated and all three approved objectives
  are enabled in the final configuration.

Acceptance:

- fixed inputs give exactly the old `rateFitnessFuncTranges` value;
- one local/mock BatchTK trial receives `rate_loss`;
- existing spike-guard tests pass;
- no plots are written in optimization mode.

### Phase B — CSD integration — adapter/preflight complete; production smoke pending

- Pin `csd_quant`, add its adapter, configure the selected event/reference, and add synthetic-array tests.
- Run one non-optimized production simulation and inspect model/reference CSD
  orientation and window before enabling the full batch search. This is the
  one remaining acceptance task.

Acceptance:

- identical CSD arrays score zero or numerical tolerance;
- shifted synthetic sinks/sources increase distance;
- incomplete time coverage and invalid shapes fail clearly;
- the simulation payload contains finite `rate_loss` and `csd_loss` values.

### Phase C — Shared manifold preprocessing/reference preparation — complete

- Extract deterministic sampling, binning, smoothing, normalization, and time alignment from the prototypes.
- Define the reference artifact/metadata format and prepare the selected experimental data.

Acceptance:

- repeated preprocessing with the same seed is identical;
- model and experimental matrices have validated feature/time dimensions;
- no preprocessing helper writes figures or mutates global `cfg` state.

### Phase D — UMAP and CEBRA adapters — complete

- Add fixed-artifact UMAP and CEBRA transforms behind the same method registry.
- Add the chosen common comparator and diagnostics.

Acceptance:

- both adapters pass the same interface tests;
- identical embeddings score zero or numerical tolerance;
- mismatched time grids or feature dimensions fail before scoring;
- switching the `method` config does not require changes to `init.py` or
  `optimization/batch_runner.py`.

### Phase E — Multi-objective batch and Pareto reporting — complete

- Enable the approved objective set, request MO-CMA with NSGA-II fallback, and
  add manifest validation.
- Implement Pareto export and plots.

Acceptance:

- BatchTK receives exactly the enabled metric names in stable order;
- disabling an objective removes it from both payload and study directions;
- an incompatible resume is rejected;
- Pareto CSV/JSON rows match Optuna `study.best_trials`;
- invalid/spike-blocked trials do not appear as useful Pareto solutions.

### Phase F — Documentation and smoke tests — local work complete; cluster run pending

- Update `README.md` with reference preparation, local run, Slurm/SGE run,
  resume, and Pareto export commands.
- Run unit tests under `NewBatchtk`, then a minimal local job and a short cluster smoke job before the full search.

## 15. Implemented tests

Focused coverage is in:

- `test_multi_objective.py`: rate parity/grouping, CSD configuration, activity
  loading/sampling/smoothing, pre-dynamics model capacity, compact reference
  preparation, provenance hashing, manifold comparison, sampler fallback,
  Pareto dominance, and equality with an actual Optuna journal front;
- `test_objective_pipeline.py`: JSON-safe diagnostics and the global invalid
  trial policy;
- `test_spike_guard.py`: existing detector and blockade behavior.

Run fast tests with:

```bash
conda run -n NewBatchtk python -m pytest -q tests
```

Keep full NEURON simulations out of the unit suite; use a separately documented smoke command.

## 16. Data and artifact policy

Suggested location:

```text
data/calibration/
├── csd/
│   └── experimental_csd.npz
└── manifolds/
    ├── experimental_activity.npz
    ├── reference_metadata.json
    ├── umap_reducer.joblib
    ├── umap_embedding.npy
    ├── cebra_model.pt
    └── cebra_embedding.npy
```

Small metadata and checksums should be version controlled. Large experimental arrays/models should use the project's approved data store or Git LFS; configuration must refer to an explicit path and checksum. Do not silently fall back to files in the ignored nested repository.

## 17. Implementation validation

- `NewBatchtk`: 36 unit/regression/structure tests passed. Structure tests
  enforce a declarative `cfg.py`, a short explicit `init.py`, and only the
  three standard Python entry files at the source root.
- OptunaHub MO-CMA: an in-memory two-objective, eight-trial study completed
  using the registered `mocma` sampler; the NSGA-II fallback is unit tested.
- CSD: the pinned backend/template loaded successfully with shape `(30, 210)`
  and event-relative times `[-10, 199]` ms; strict Git-revision validation also
  passed. With the actual upstream backend, identical small synthetic CSDs
  scored `0.0` and a time-shifted comparison scored `0.5303300858899106`. A
  production-scale Wasserstein run remains for cluster validation because
  upstream uses a costly fixed 30×200 transport grid.
- Reference preparation: the real configured CEBRA command produced compact
  `(124, 91)` activity and a stored `(124, 3)` embedding with provenance
  metadata. Runtime supports loading this embedding instead of recomputing it.
- Manifold storage: CEBRA and UMAP inputs were copied into
  `data/manifolds/`; active configuration no longer reads artifacts from the
  ignored prototype tree.
- CEBRA smoke: a connected 245-cell model (seven cells per population) produced
  matched `(124, 91)` activity matrices, `(124, 3)` embeddings, finite
  `manifold_loss=1.3350294219994112`, and `trial_valid=true`. The same full
  smoke was repeated successfully after the directory reorganization.

### Remaining production validation

1. Place the pinned `csd_quant` checkout at `external/csd_quant` (or set
   `M1_CSD_QUANT_PATH`) on the cluster filesystem.
2. Run a short non-optimized/full-network job with all three objectives and
   inspect contact order, depth direction, sign convention, and `[0, 200)` ms
   coverage in the CSD diagnostics.
3. Run a small multi-objective cluster study, reconstruct its Pareto outputs
   from the journal, and then authorize the full trial budget.

## 18. References used for this plan

- [NetPyNE BatchTools user documentation](https://doc.netpyne.org/user_documentation.html#running-a-batch-job-beta)
- [BatchTK 0.1.6 Optuna implementation](https://github.com/jchen6727/batchtk/blob/a86e6d92d7487471100ebc112cf1bdf521ca86ac/batchtk/algos/optuna_utils.py)
- [BatchTK CMA-ES implementation showing single-objective selection](https://github.com/jchen6727/batchtk/blob/7773dc0b2e9841bec8c1294a5265b295576bb491/batchtk/algos/cmaes_utils.py)
- [`csd_quant` template scoring function](https://github.com/smcelroy97/csd_quant/blob/676bf568bbcdd877e5a8d4f60c2a4b5788233020/wasserstein_dist.py)
- [`csd_quant` Wasserstein implementation](https://github.com/smcelroy97/csd_quant/blob/676bf568bbcdd877e5a8d4f60c2a4b5788233020/utils.py)
- [Optuna multi-objective and Pareto-front example](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html)
