"""Derive recording and plotting settings from the declarative simulation cfg."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


ALL_CORTICAL_POPS = [
    "NGF1",
    "IT2", "PV2", "SOM2", "VIP2", "NGF2",
    "IT4", "PV4", "SOM4", "VIP4", "NGF4",
    "IT5A", "PV5A", "SOM5A", "VIP5A", "NGF5A",
    "IT5B", "PT5B", "PV5B", "SOM5B", "VIP5B", "NGF5B",
    "IT6", "CT6", "PV6", "SOM6", "VIP6", "NGF6",
]
DEFAULT_LFP_COLORS = [
    (0.1216, 0.4667, 0.7059),
    (1.0000, 0.4980, 0.0549),
    (0.1725, 0.6275, 0.1725),
    (0.8392, 0.1529, 0.1569),
    (0.5804, 0.4039, 0.7412),
    (0.5490, 0.3373, 0.2941),
]


def _recorded_cells(mode: int) -> list[Any]:
    if mode == 0:
        return ["all"]
    if mode == 1:
        return [(population, 0) for population in ALL_CORTICAL_POPS]
    if mode == 2:
        return [("IT2", 10), ("IT5A", 10), ("PT5B", 10), ("PV5B", 10), ("SOM5B", 10)]
    if mode == 3:
        return [(population, 50) for population in ("IT5A", "PT5B")] + [
            ("PT5B", index) for index in (393, 579, 19, 104)
        ]
    if mode == 4:
        return [(population, 50) for population in ("IT2", "IT4", "IT5A", "PT5B")] + [
            ("IT5A", index) for index in (393, 447, 579, 19, 104)
        ] + [("PT5B", index) for index in (393, 447, 579, 19, 104, 214, 1138, 979, 799)]
    if mode == 5:
        populations = ("PV2", "PV4", "PV5A", "PV5B", "PV6", "PT5B")
        return [(population, index) for population in populations for index in range(0, 100, 2)]
    raise ValueError(f"Unknown cfg.cellsrec mode: {mode}")


def _raster_has_selected_spikes(sim: Any, settings: dict[str, Any]) -> bool:
    """Return whether the configured populations spike in the plotted window."""

    included = set(settings.get("include", []))
    population_by_gid = {
        int(cell["gid"]): str(cell.get("tags", {}).get("pop", ""))
        for cell in getattr(sim.net, "allCells", [])
        if isinstance(cell, dict) and "gid" in cell
    }
    selected_gids = {
        gid for gid, population in population_by_gid.items() if population in included
    }
    start, stop = settings.get("timeRange", [float("-inf"), float("inf")])
    spike_ids = sim.allSimData.get("spkid", [])
    spike_times = sim.allSimData.get("spkt", [])
    return any(
        int(gid) in selected_gids and float(start) <= float(time) <= float(stop)
        for gid, time in zip(spike_ids, spike_times)
    )


def skip_unavailable_plots(sim: Any, cfg: Any) -> None:
    """Remove plots that NetPyNE cannot render for the gathered data."""

    raster = cfg.analysis.get("plotRaster")
    if raster and not _raster_has_selected_spikes(sim, raster):
        cfg.analysis.pop("plotRaster")
        print("Skipping plotRaster: selected cortical populations produced no spikes.")


def configure_recording_and_analysis(cfg: Any, project_root: Path) -> None:
    """Apply derived recording settings after BatchTK mappings are known."""

    cfg.recordCells = _recorded_cells(int(cfg.cellsrec))
    cfg.recordStep = float(cfg.dt)
    cfg.analysis = {}

    if not bool(cfg.plotSimResults):
        cfg.recordTraces = {}
        cfg.recordDipole = False
        return

    color_path = project_root / "cells" / "popColors.pkl"
    with color_path.open("rb") as handle:
        population_colors = pickle.load(handle)["popColors"]

    cfg.recordTraces = {
        "V_soma": {"sec": "soma", "loc": 0.5, "var": "v"},
        "V_apic_3": {"sec": "apic_3", "loc": 0.5, "var": "v", "conds": {"pop": "PT5B"}},
        "V_dend_1": {"sec": "dend_1", "loc": 0.5, "var": "v", "conds": {"pop": "PT5B"}},
    }
    cfg.recordDipole = True
    cfg.analysis["plotRaster"] = {
        "include": ALL_CORTICAL_POPS,
        "orderBy": ["pop", "y"],
        "timeRange": cfg.timeRanges,
        "saveFig": True,
        "showFig": cfg.showPlots,
        "popRates": True,
        "orderInverse": True,
        "popColors": population_colors,
        "figSize": (12, 18),
        "lw": 0.3,
        "markerSize": 3,
        "marker": ".",
        "dpi": 300,
    }
    cfg.analysis["plotTraces"] = {
        "include": cfg.recordCells,
        "timeRange": cfg.timeRanges,
        "overlay": True,
        "oneFigPer": "trace",
        "figSize": (10, 4),
        "saveFig": True,
        "showFig": cfg.showPlots,
        "subtitles": True,
        "legend": True,
    }
    cfg.analysis["plotLFP"] = {
        "plots": ["timeSeries", "PSD", "spectrogram"],
        "electrodes": list(range(len(cfg.recordLFP)))[::2],
        "colors": DEFAULT_LFP_COLORS,
        "timeRange": cfg.timeRanges,
        "minFreq": 1,
        "maxFreq": 80,
        "figSize": (8, 4),
        "saveData": False,
        "saveFig": True,
        "showFig": cfg.showPlots,
    }
    cfg.analysis["plotDipole"] = {
        "timeRange": cfg.timeRanges,
        "saveFig": True,
        "showFig": cfg.showPlots,
    }
    cfg.analysis["plotCSD"] = {
        "timeRange": cfg.timeRanges,
        "saveFig": True,
        "showFig": cfg.showPlots,
    }
    cfg.analysis["plotEEG"] = {
        "timeRange": cfg.timeRanges,
        "saveFig": True,
        "showFig": cfg.showPlots,
    }


__all__ = [
    "ALL_CORTICAL_POPS",
    "configure_recording_and_analysis",
    "skip_unavailable_plots",
]
