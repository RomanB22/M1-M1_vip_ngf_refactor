"""Apply BatchTK mappings and runtime-only configuration before network construction."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

from calibration.objective_config import get_active_objectives

from .analysis_config import configure_recording_and_analysis
from .mutation_presets import build_mutation_list
from . import network_helpers


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _configure_plot_output(cfg: Any) -> None:
    """Keep local plots on by default without plotting every BatchTK trial."""

    override = os.environ.get("M1_PLOT_RESULTS")
    if override is not None:
        cfg.plotSimResults = _parse_boolean(override)
    elif cfg._batchtk_path_pointer is not None:
        cfg.plotSimResults = False


def _enabled_objectives_from_environment(cfg: Any) -> None:
    raw_names = os.environ.get("M1_SMOKE_OBJECTIVES")
    if not raw_names:
        return
    enabled = {name.strip() for name in raw_names.split(",") if name.strip()}
    unknown = enabled.difference(cfg.objectives)
    if unknown:
        raise ValueError(f"Unknown M1_SMOKE_OBJECTIVES: {sorted(unknown)}")
    for name, spec in cfg.objectives.items():
        spec["enabled"] = name in enabled


def _apply_smoke_test_overrides(cfg: Any) -> None:
    raw_single_cell = os.environ.get("M1_SINGLE_CELL_POPS")
    if raw_single_cell is not None:
        cfg.singleCellPops = raw_single_cell.lower() in {"1", "true", "yes"}
    cfg.testCellsPerPop = int(os.environ.get("M1_TEST_CELLS_PER_POP", cfg.testCellsPerPop))
    if cfg.testCellsPerPop < 1:
        raise ValueError("M1_TEST_CELLS_PER_POP must be at least 1")
    if not cfg.singleCellPops:
        return

    cfg.dt = float(os.environ.get("M1_TEST_DT_MS", cfg.dt))
    cfg.coreneuron = False
    if cfg.pt5b_variant == "tim":
        cfg.cellModelLoadModeByLabel = dict(cfg.cellModelLoadModeByLabel)
        cfg.cellModelLoadModeByLabel["PT5B_full"] = "saved"
    _enabled_objectives_from_environment(cfg)


def _derive_dependent_values(cfg: Any) -> None:
    cfg.duration = 2 * float(cfg.preTone) + float(cfg.postTone)
    cfg.timeRanges = [cfg.duration - cfg.postTone - cfg.preTone, cfg.duration]
    cfg.printPopAvgRates = [
        [start, min(start + 250, cfg.timeRanges[1])]
        for start in range(int(cfg.timeRanges[0]), int(cfg.timeRanges[1]), 250)
    ]
    cfg.recordStep = cfg.dt
    cfg.dendNa = 0.3 if cfg.pt5b_variant == "standard" else 1.0
    cfg.ihGbar = 1.0 if cfg.SimulateBaseline else 0.25
    cfg.modifyMechs = dict(cfg.modifyMechs)
    cfg.modifyMechs["startTime"] = cfg.preTone
    cfg.modifyMechs["endTime"] = cfg.duration
    cfg.numCellsLong = int(1000 * cfg.scaleDensity)
    quiet_rates = [0, 2.5]
    tvl_rates = quiet_rates if cfg.SimulateBaseline else [0, 10]
    cfg.ratesLong = {
        "TPO": quiet_rates,
        "TVL": tvl_rates,
        "S1": quiet_rates,
        "S2": quiet_rates,
        "cM1": quiet_rates,
        "M2": quiet_rates,
        "OC": quiet_rates,
    }


def _load_in_vivo_inputs(cfg: Any) -> None:
    if not cfg.addInVivoThalamus:
        return
    baseline, movement = network_helpers.loadThalSpikes(str(PROJECT_ROOT), cfg, skipEmpty=False)
    sampled_cells, _session_names = network_helpers.m1SampledDepths(
        str(PROJECT_ROOT), layer_order=["1", "2", "4", "5A", "5B", "6"]
    )
    cfg.numSampledCellsPerLayer = network_helpers.average_dict_entries(sampled_cells)
    baseline = network_helpers.trimTVLSpikes(baseline, cfg)
    movement = network_helpers.trimTVLSpikes(movement, cfg)
    cfg.spikeTimesInVivo = baseline if cfg.SimulateBaseline else movement
    gc.collect()


def prepare_runtime_config(cfg: Any) -> None:
    """Finalize config after BatchTK mappings, without hiding values in cfg.py."""

    if getattr(cfg, "_runtime_configured", False):
        return
    cfg.update()
    if cfg._batchtk_path_pointer is not None:
        cfg.saveFolder = cfg._batchtk_path_pointer
        cfg.simLabel = cfg._batchtk_label_pointer

    _configure_plot_output(cfg)
    _apply_smoke_test_overrides(cfg)
    _derive_dependent_values(cfg)
    if cfg.loadmutantParams:
        raise ValueError("cfg.loadmutantParams is not implemented")
    cfg.mutations = build_mutation_list(cfg)

    active = get_active_objectives(cfg.objectives)
    if not cfg.plotSimResults and cfg.optimizationMode and not any(
        spec.get("kind") == "csd_wasserstein" for spec in active.values()
    ):
        cfg.recordLFP = []

    _load_in_vivo_inputs(cfg)
    configure_recording_and_analysis(cfg, PROJECT_ROOT)
    cfg._runtime_configured = True


__all__ = ["PROJECT_ROOT", "prepare_runtime_config"]
