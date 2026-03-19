from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from m1_model.jaxley_m1 import M1_DEFAULT_PARAMS, M1_PARAM_BOUNDS, M1_PARAM_ORDER, dict_to_vector, unconstrained_from_bounded, vector_to_dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_batch_params():
    batch_params_path = PROJECT_ROOT / "src_test" / "batch_params.py"
    spec = spec_from_file_location("batch_params_test", batch_params_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_batch_params(0.5, 1.5)


def test_param_names_and_bounds_match_batch_params():
    params = _load_batch_params()
    assert M1_PARAM_ORDER == tuple(params.keys())
    assert M1_PARAM_BOUNDS == params


def test_vector_dict_roundtrip_is_reversible():
    default_vector = dict_to_vector(M1_DEFAULT_PARAMS)
    reconstructed = vector_to_dict(default_vector)
    assert reconstructed == M1_DEFAULT_PARAMS


def test_unconstrained_roundtrip_respects_bounds():
    default_vector = dict_to_vector(M1_DEFAULT_PARAMS)
    unconstrained = unconstrained_from_bounded(default_vector)
    reconstructed = dict_to_vector(vector_to_dict(default_vector))
    assert np.allclose(default_vector, reconstructed)
    assert unconstrained.shape == default_vector.shape
