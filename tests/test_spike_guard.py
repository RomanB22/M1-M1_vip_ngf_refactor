from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_TEST = PROJECT_ROOT / "src_test"
if str(SRC_TEST) not in sys.path:
    sys.path.insert(0, str(SRC_TEST))

import spike_guard


def _build_spike(amplitude: float = 30.0) -> list[float]:
    return [-65.0, -55.0, -35.0, -10.0, amplitude, -5.0, -40.0, -65.0]


def _build_wiggle() -> list[float]:
    return [-5.0, -1.0, 2.0, -2.0, -5.0]


def _default_exc_params() -> dict[str, float]:
    cfg = spike_guard.default_spike_guard_config()
    params = dict(cfg["families"]["exc"])
    params["candidateStartMv"] = cfg["candidateStartMv"]
    params["plateauMv"] = cfg["plateauMv"]
    return params


def test_reference_detector_accepts_normal_ap_train():
    params = _default_exc_params()
    trace = [-65.0] * 5 + _build_spike() + [-65.0] * 8 + _build_spike() + [-65.0] * 5
    result = spike_guard.run_reference_detector(trace, dt_ms=0.25, params=params)
    assert result["accepted_count"] == 2
    assert result["rejected_count"] == 0


def test_reference_detector_rejects_plateau_wiggles():
    params = _default_exc_params()
    trace = [-5.0] * 4 + _build_wiggle() * 8 + [-5.0] * 4
    result = spike_guard.run_reference_detector(trace, dt_ms=0.25, params=params)
    assert result["accepted_count"] == 0
    assert result["rejected_count"] >= 4
    assert result["plateau_max_ms"] > 0.0


def test_reference_detector_keeps_real_spikes_and_rejects_false_ones():
    params = _default_exc_params()
    trace = [-65.0] * 6 + _build_spike() + [-5.0] * 4 + _build_wiggle() * 4 + [-65.0] * 6 + _build_spike()
    result = spike_guard.run_reference_detector(trace, dt_ms=0.25, params=params)
    assert result["accepted_count"] == 2
    assert result["rejected_count"] >= 2


def test_blockade_classification_for_long_plateau():
    cfg = spike_guard.default_spike_guard_config()
    metrics = {
        "accepted_spikes": 0,
        "rejected_candidates": 8,
        "plateau_max_ms": 125.0,
    }
    assert spike_guard.cell_is_blocked(metrics, cfg)


def test_guard_injection_preserves_spikegenloc_and_skips_vecstim():
    cfg = spike_guard.default_spike_guard_config()
    it_rule = {
        "conds": {"cellType": "IT", "cellModel": "HH_reduced"},
        "secs": {
            "soma": {"geom": {}, "topol": {}, "spikeGenLoc": 0.7},
            "dend": {"geom": {}, "topol": {"parentSec": "soma", "parentX": 1.0, "childX": 0.0}},
        },
    }
    vecstim_rule = {"conds": {"cellType": "IT", "cellModel": "VecStim"}, "secs": {}}

    injected = spike_guard.inject_guard_into_rule(it_rule, cfg)
    skipped = spike_guard.inject_guard_into_rule(vecstim_rule, cfg)

    assert injected == {"secName": "soma", "loc": 0.7}
    assert it_rule["secs"]["soma"]["spikeGenLoc"] == 0.7
    assert "spike_guard" in it_rule["secs"]["soma"]["pointps"]
    assert "vref" in it_rule["secs"]["soma"]["pointps"]["spike_guard"]
    assert it_rule["secs"]["soma"]["threshold"] == 0.5
    assert skipped is None


def test_guard_pointp_detection_matches_name_or_mod():
    assert spike_guard.is_spike_guard_pointp("spike_guard", {"mod": "Other"})
    assert spike_guard.is_spike_guard_pointp("custom_name", {"mod": "SpikeGuard"})
    assert not spike_guard.is_spike_guard_pointp("custom_name", {"mod": "Izhi2007a"})


def test_collect_metrics_finds_guard_on_non_root_section():
    cfg = spike_guard.default_spike_guard_config()

    class MockHObj:
        accepted_count = 4.0
        rejected_count = 7.0
        plateau_max_ms = 123.5

    class MockCell:
        gid = 12
        tags = {"pop": "IT5A", "cellType": "IT", "cellModel": "HH_full"}
        secs = {
            "soma": {"topol": {}, "pointps": {}},
            "axon_0": {"topol": {"parentSec": "soma", "parentX": 1.0, "childX": 0.0}, "pointps": {"spike_guard": {"hObj": MockHObj()}}},
        }

    metrics = spike_guard.collect_local_guard_metrics([MockCell()], cfg)
    assert metrics["cell_12"]["sourceSec"] == "axon_0"
    assert metrics["cell_12"]["accepted_spikes"] == 4
    assert metrics["cell_12"]["rejected_candidates"] == 7
    assert metrics["cell_12"]["plateau_max_ms"] == 123.5


def test_collect_metrics_skips_cells_without_section_mapping():
    cfg = spike_guard.default_spike_guard_config()

    class MockPointCell:
        gid = 99
        tags = {"pop": "TVL", "cellType": "VecStim", "cellModel": "VecStim"}

        def secs(self):
            return {}

    metrics = spike_guard.collect_local_guard_metrics([MockPointCell()], cfg)
    assert metrics == {}


def test_blocked_pop_summary_and_penalty():
    cfg = spike_guard.default_spike_guard_config()
    all_metrics = {
        "cell_1": {"gid": 1, "pop": "IT5A", "accepted_spikes": 0, "rejected_candidates": 8, "plateau_max_ms": 130.0},
        "cell_2": {"gid": 2, "pop": "IT5A", "accepted_spikes": 1, "rejected_candidates": 9, "plateau_max_ms": 140.0},
        "cell_3": {"gid": 3, "pop": "IT5A", "accepted_spikes": 0, "rejected_candidates": 7, "plateau_max_ms": 120.0},
        "cell_4": {"gid": 4, "pop": "PV5A", "accepted_spikes": 5, "rejected_candidates": 0, "plateau_max_ms": 0.0},
    }

    summary = spike_guard.summarize_guard_metrics(all_metrics, cfg, single_cell_pops=False)
    results = spike_guard.guard_summary_for_results(summary)

    assert summary["blockedPops"] == ["IT5A"]
    assert summary["blockedPopCount"] == 1
    assert summary["pops"]["IT5A"]["blockedCells"] == 3
    assert summary["pops"]["IT5A"]["blocked"] is True
    assert spike_guard.blockade_penalty(summary, cfg) == 250.0
    assert results["blockedPops"] == ["IT5A"]
    assert "cells" not in results


def test_batch_param_space_is_unchanged():
    sys.path.insert(0, str(SRC_TEST))
    from batch_params import get_batch_params

    params = get_batch_params(0.5, 1.5)
    assert len(params) == 18
    assert sorted(params.keys()) == sorted(
        [
            "weightLong.TPO",
            "weightLong.TVL",
            "weightLong.S1",
            "weightLong.S2",
            "weightLong.cM1",
            "weightLong.M2",
            "weightLong.OC",
            "EEGain",
            "IEweights.0",
            "IEweights.1",
            "IEweights.2",
            "IIweights.0",
            "IIweights.1",
            "IIweights.2",
            "EICellTypeGain.PV",
            "EICellTypeGain.SOM",
            "EICellTypeGain.VIP",
            "EICellTypeGain.NGF",
        ]
    )
