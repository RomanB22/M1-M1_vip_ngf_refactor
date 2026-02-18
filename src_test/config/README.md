# `src_test/config` layout

- `cfg_defaults.yaml`: stable defaults loaded by `src_test/cfg.py`.
- `src_test/cfg.py`: active knobs/overrides plus derived values.

The intended workflow is:
1. Keep day-to-day tuning parameters in `src_test/cfg.py`.
2. Move rarely changed constants into `cfg_defaults.yaml`.
3. Keep derived values (e.g. `duration`, `timeRanges`, `ratesLong`) in `src_test/cfg.py` so dependencies remain explicit.
