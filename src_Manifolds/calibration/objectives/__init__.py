"""Calibration objective registry."""

from .csd import score_csd_objective, validate_csd_setup
from .manifold import (
    score_manifold_objective,
    validate_manifold_capacity,
    validate_manifold_setup,
)
from .firing_rates import score_rate_objective, validate_rate_config


OBJECTIVE_SCORERS = {
    "population_rates": score_rate_objective,
    "csd_wasserstein": score_csd_objective,
    "manifold": score_manifold_objective,
}

OBJECTIVE_VALIDATORS = {
    "population_rates": validate_rate_config,
    "csd_wasserstein": validate_csd_setup,
    "manifold": validate_manifold_setup,
}

OBJECTIVE_NETWORK_VALIDATORS = {
    "manifold": validate_manifold_capacity,
}


__all__ = ["OBJECTIVE_NETWORK_VALIDATORS", "OBJECTIVE_SCORERS", "OBJECTIVE_VALIDATORS"]
