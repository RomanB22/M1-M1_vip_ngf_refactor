from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types

import numpy as np

from m1_model.jaxley_data import cortical_population_specs, m1_defaults
from m1_model.jaxley_loss import build_target_arrays, rate_fitness_tranges_jax, windowed_rates_to_dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_defs_module():
    fake_netpyne = types.ModuleType("netpyne")
    fake_netpyne.specs = object()
    sys.modules.setdefault("netpyne", fake_netpyne)

    defs_path = PROJECT_ROOT / "src_test" / "defs.py"
    spec = spec_from_file_location("defs_for_loss_test", defs_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jaxley_loss_matches_existing_rate_loss_formula():
    specs = cortical_population_specs()
    defaults = m1_defaults()
    targets, widths, mins = build_target_arrays([spec.__dict__ for spec in specs])

    rates = np.zeros((len(specs), len(defaults.rate_windows)), dtype=float)
    for pop_idx, spec in enumerate(specs):
        base = spec.target["target"]
        rates[pop_idx] = np.asarray([base - 1.0, base, base + 1.0, base + 2.0], dtype=float)

    expected_pop_rates = windowed_rates_to_dict(
        [spec.name for spec in specs],
        rates,
        defaults.rate_windows,
    )
    defs = _load_defs_module()
    expected = defs.rateFitnessFuncTranges(
        {"popRates": expected_pop_rates},
        pops={spec.name: spec.target for spec in specs},
        maxFitness=1000.0,
        tranges=defaults.rate_windows,
    )

    actual = rate_fitness_tranges_jax(rates, targets, widths, mins, max_fitness=1000.0)
    assert np.isclose(actual, expected)
