from __future__ import annotations

from collections.abc import Mapping
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable


ELIGIBLE_CELL_TYPES = {"IT", "PT", "CT", "PV", "SOM", "VIP", "NGF"}
EXCITATORY_CELL_TYPES = {"IT", "PT", "CT"}
DEFAULT_POINTP_NAME = "spike_guard"


def default_spike_guard_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "pointpName": DEFAULT_POINTP_NAME,
        "mod": "SpikeGuard",
        "vref": "detector_v",
        "candidateStartMv": -20.0,
        "plateauMv": -40.0,
        "plateauMs": 100.0,
        "thresholdForDetectorV": 0.5,
        "lossPenaltyPerBlockedPop": 250.0,
        "blockedFractionThreshold": 0.10,
        "blockedMinCells": 3,
        "blockedNoSpikesMinRejected": 5,
        "blockedRejectedScale": 3,
        "blockedRejectedOffset": 5,
        "families": {
            "exc": {
                "minPeakMv": 10.0,
                "minProminenceMv": 20.0,
                "minDvdtMvPerMs": 10.0,
                "refractoryMs": 2.0,
            },
            "inh": {
                "minPeakMv": 0.0,
                "minProminenceMv": 15.0,
                "minDvdtMvPerMs": 8.0,
                "refractoryMs": 1.0,
            },
        },
        "cellTypeOverrides": {},
    }


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def normalize_spike_guard_config(raw_cfg: Any) -> dict[str, Any]:
    cfg = default_spike_guard_config()
    if isinstance(raw_cfg, dict):
        _deep_update(cfg, raw_cfg)
    return cfg


def guard_enabled(raw_cfg: Any) -> bool:
    return bool(normalize_spike_guard_config(raw_cfg).get("enabled", True))


def is_guard_eligible_rule(rule: dict[str, Any]) -> bool:
    conds = rule.get("conds", {}) or {}
    cell_type = conds.get("cellType")
    return isinstance(rule.get("secs"), dict) and bool(rule["secs"]) and cell_type in ELIGIBLE_CELL_TYPES


def guard_family_for_cell_type(cell_type: str) -> str:
    return "exc" if cell_type in EXCITATORY_CELL_TYPES else "inh"


def rule_guard_params(rule: dict[str, Any], raw_cfg: Any) -> dict[str, float]:
    cfg = normalize_spike_guard_config(raw_cfg)
    conds = rule.get("conds", {}) or {}
    cell_type = conds.get("cellType")
    family = guard_family_for_cell_type(cell_type)
    family_cfg = deepcopy(cfg["families"][family])
    override = cfg.get("cellTypeOverrides", {}).get(cell_type, {})
    family_cfg.update(override)
    family_cfg["candidateStartMv"] = float(cfg["candidateStartMv"])
    family_cfg["plateauMv"] = float(cfg["plateauMv"])
    return family_cfg


def resolve_source_sec_name_and_loc(secs: dict[str, Any]) -> tuple[str, float]:
    for sec_name, sec in secs.items():
        if "spikeGenLoc" in sec:
            return sec_name, float(sec["spikeGenLoc"])

    for sec_name, sec in secs.items():
        topol = sec.get("topol", {}) or {}
        if len(topol) == 0:
            return sec_name, 0.5

    first_name = next(iter(secs))
    return first_name, 0.5


def inject_guard_into_rule(rule: dict[str, Any], raw_cfg: Any) -> dict[str, Any] | None:
    if not is_guard_eligible_rule(rule):
        return None

    cfg = normalize_spike_guard_config(raw_cfg)
    pointp_name = str(cfg["pointpName"])
    sec_name, loc = resolve_source_sec_name_and_loc(rule["secs"])
    sec = rule["secs"][sec_name]
    sec.setdefault("pointps", {})

    guard_params = rule_guard_params(rule, cfg)
    sec["pointps"][pointp_name] = {
        "mod": cfg["mod"],
        "loc": loc,
        "vref": cfg["vref"],
        "candidateStartMv": float(guard_params["candidateStartMv"]),
        "minPeakMv": float(guard_params["minPeakMv"]),
        "minProminenceMv": float(guard_params["minProminenceMv"]),
        "minDvdtMvPerMs": float(guard_params["minDvdtMvPerMs"]),
        "refractoryMs": float(guard_params["refractoryMs"]),
        "plateauMv": float(guard_params["plateauMv"]),
    }
    sec["threshold"] = float(cfg["thresholdForDetectorV"])

    return {"secName": sec_name, "loc": loc}


def inject_guard_into_netparams(netParams: Any, raw_cfg: Any) -> dict[str, Any]:
    report = {"eligible": 0, "injected": 0, "labels": {}}
    if not guard_enabled(raw_cfg):
        return report

    for label, rule in getattr(netParams, "cellParams", {}).items():
        if not is_guard_eligible_rule(rule):
            continue
        report["eligible"] += 1
        injected = inject_guard_into_rule(rule, raw_cfg)
        if injected:
            report["injected"] += 1
            report["labels"][label] = injected
    return report


def install_netpyne_spike_source_patch() -> None:
    from netpyne import sim
    from netpyne.cell.compartCell import CompartCell

    if getattr(CompartCell._setConnPointP, "_spike_guard_patched", False):
        return

    def set_conn_pointp_ignoring_spike_guard(self: Any, params: dict[str, Any], secLabels: list[str], weightIndex: int):
        pointp = None
        if len(secLabels) == 1 and "pointps" in self.secs[secLabels[0]]:
            for pointp_name, pointp_params in self.secs[secLabels[0]]["pointps"].items():
                if pointp_name == DEFAULT_POINTP_NAME or pointp_params.get("mod") == "SpikeGuard":
                    continue
                if "vref" in pointp_params:
                    pointp = pointp_name
                    if "synList" in pointp_params:
                        if isinstance(params.get("synMech"), list):
                            weightIndex = [pointp_params["synList"].index(synMech) for synMech in params.get("synMech")]
                        else:
                            weightIndex = pointp_params["synList"].index(params.get("synMech"))

        if pointp and params["synsPerConn"] > 1:
            if sim.cfg.verbose:
                print(
                    "  Error: Multiple synapses per connection rule not allowed for cells where V is not in section (cell gid=%d) "
                    % (self.gid)
                )
            return -1, weightIndex

        return pointp, weightIndex

    set_conn_pointp_ignoring_spike_guard._spike_guard_patched = True  # type: ignore[attr-defined]
    CompartCell._setConnPointP = set_conn_pointp_ignoring_spike_guard


def _guard_pointp_from_cell(cell: Any, pointp_name: str) -> tuple[Any, Any] | tuple[None, None]:
    secs = getattr(cell, "secs", {}) or {}
    if not isinstance(secs, Mapping):
        return None, None
    for sec_name, sec in secs.items():
        if not isinstance(sec, Mapping):
            continue
        pointps = sec.get("pointps", {}) or {}
        if not isinstance(pointps, Mapping):
            continue
        pointp = pointps.get(pointp_name)
        if pointp:
            return sec_name, pointp.get("hObj")
    return None, None


def collect_local_guard_metrics(cells: Iterable[Any], raw_cfg: Any) -> dict[str, dict[str, Any]]:
    cfg = normalize_spike_guard_config(raw_cfg)
    pointp_name = str(cfg["pointpName"])
    metrics: dict[str, dict[str, Any]] = {}

    for cell in cells:
        sec_name, hobj = _guard_pointp_from_cell(cell, pointp_name)
        if hobj is None:
            continue
        key = f"cell_{int(cell.gid)}"
        metrics[key] = {
            "gid": int(cell.gid),
            "pop": cell.tags.get("pop"),
            "cellType": cell.tags.get("cellType"),
            "cellModel": cell.tags.get("cellModel"),
            "sourceSec": sec_name,
            "accepted_spikes": int(round(float(hobj.accepted_count))),
            "rejected_candidates": int(round(float(hobj.rejected_count))),
            "plateau_max_ms": float(hobj.plateau_max_ms),
        }
    return metrics


def cell_is_blocked(cell_metrics: dict[str, Any], raw_cfg: Any) -> bool:
    cfg = normalize_spike_guard_config(raw_cfg)
    accepted = int(cell_metrics.get("accepted_spikes", 0))
    rejected = int(cell_metrics.get("rejected_candidates", 0))
    plateau_max_ms = float(cell_metrics.get("plateau_max_ms", 0.0))

    if plateau_max_ms < float(cfg["plateauMs"]):
        return False
    if accepted == 0 and rejected >= int(cfg["blockedNoSpikesMinRejected"]):
        return True
    return rejected >= int(cfg["blockedRejectedScale"]) * accepted + int(cfg["blockedRejectedOffset"])


def summarize_guard_metrics(all_metrics: dict[str, dict[str, Any]], raw_cfg: Any, single_cell_pops: bool) -> dict[str, Any]:
    cfg = normalize_spike_guard_config(raw_cfg)
    cells: dict[str, dict[str, Any]] = {}
    pops: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "numCells": 0,
            "blockedCells": 0,
            "blockedFraction": 0.0,
            "acceptedSpikes": 0,
            "rejectedCandidates": 0,
            "maxPlateauMs": 0.0,
            "blocked": False,
        }
    )

    for key, metrics in sorted(all_metrics.items(), key=lambda item: item[1].get("gid", -1)):
        blocked = cell_is_blocked(metrics, cfg)
        entry = dict(metrics)
        entry["blocked"] = blocked
        cells[key] = entry

        pop_name = metrics.get("pop") or "UNKNOWN"
        pop = pops[pop_name]
        pop["numCells"] += 1
        pop["blockedCells"] += int(blocked)
        pop["acceptedSpikes"] += int(metrics.get("accepted_spikes", 0))
        pop["rejectedCandidates"] += int(metrics.get("rejected_candidates", 0))
        pop["maxPlateauMs"] = max(pop["maxPlateauMs"], float(metrics.get("plateau_max_ms", 0.0)))

    blocked_pops: list[str] = []
    for pop_name, pop in pops.items():
        if pop["numCells"] > 0:
            pop["blockedFraction"] = pop["blockedCells"] / float(pop["numCells"])
        if single_cell_pops:
            pop["blocked"] = pop["blockedCells"] > 0
        else:
            pop["blocked"] = pop["blockedCells"] >= int(cfg["blockedMinCells"]) and pop["blockedFraction"] >= float(
                cfg["blockedFractionThreshold"]
            )
        if pop["blocked"]:
            blocked_pops.append(pop_name)

    return {
        "cells": cells,
        "pops": dict(sorted(pops.items())),
        "blockedPops": sorted(blocked_pops),
        "blockedPopCount": len(blocked_pops),
    }


def blockade_penalty(summary: dict[str, Any], raw_cfg: Any) -> float:
    cfg = normalize_spike_guard_config(raw_cfg)
    return float(cfg["lossPenaltyPerBlockedPop"]) * float(summary.get("blockedPopCount", 0))


def guard_summary_for_results(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "blockedPopCount": summary.get("blockedPopCount", 0),
        "blockedPops": summary.get("blockedPops", []),
        "pops": summary.get("pops", {}),
    }


def run_reference_detector(trace_mv: Iterable[float], dt_ms: float, params: dict[str, float]) -> dict[str, Any]:
    trace = list(trace_mv)
    if not trace:
        return {
            "accepted_times_ms": [],
            "accepted_count": 0,
            "rejected_count": 0,
            "plateau_max_ms": 0.0,
        }

    accepted_times_ms: list[float] = []
    rejected_count = 0
    plateau_run_ms = 0.0
    plateau_max_ms = 0.0
    last_v = trace[0]
    last_dvdt = 0.0
    recent_trough = trace[0]
    candidate_trough = trace[0]
    candidate_peak = trace[0]
    candidate_max_dvdt = 0.0
    in_candidate = False
    last_accept_t = -1.0e9

    for i, v in enumerate(trace):
        t_ms = i * dt_ms
        dvdt = 0.0 if i == 0 else (v - last_v) / dt_ms

        if v >= params["plateauMv"]:
            plateau_run_ms += dt_ms
            plateau_max_ms = max(plateau_max_ms, plateau_run_ms)
        else:
            plateau_run_ms = 0.0

        if in_candidate:
            candidate_peak = max(candidate_peak, v)
            candidate_max_dvdt = max(candidate_max_dvdt, dvdt)
            if dvdt <= 0.0:
                accepted = (
                    (t_ms - last_accept_t) >= params["refractoryMs"]
                    and candidate_peak >= params["minPeakMv"]
                    and (candidate_peak - candidate_trough) >= params["minProminenceMv"]
                    and candidate_max_dvdt >= params["minDvdtMvPerMs"]
                )
                if accepted:
                    accepted_times_ms.append(t_ms)
                    last_accept_t = t_ms
                else:
                    rejected_count += 1
                in_candidate = False
                recent_trough = v
        else:
            if dvdt <= 0.0:
                recent_trough = v
            elif v < recent_trough:
                recent_trough = v

            crossed = dvdt > 0.0 and (
                (v >= params["candidateStartMv"] and last_v < params["candidateStartMv"])
                or (last_v >= params["candidateStartMv"] and last_dvdt <= 0.0)
            )
            if crossed:
                in_candidate = True
                candidate_trough = recent_trough
                candidate_peak = v
                candidate_max_dvdt = dvdt

        last_v = v
        last_dvdt = dvdt

    return {
        "accepted_times_ms": accepted_times_ms,
        "accepted_count": len(accepted_times_ms),
        "rejected_count": rejected_count,
        "plateau_max_ms": plateau_max_ms,
    }
