"""
cfg.py

Simulation configuration for M1 model (using NetPyNE)

Contributors: salvadordura@gmail.com
"""

from __future__ import annotations

import gc
import os
import pickle
from pathlib import Path

import yaml
from netpyne import specs

import defs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_TEST_DIR = Path(__file__).resolve().parent
DEFAULTS_PATH = SRC_TEST_DIR / "config" / "cfg_defaults.yaml"

cfg = specs.SimConfig()

# Batch pointer hooks expected by batchtk/netpyne.
cfg._batchtk_label_pointer = None
cfg._batchtk_path_pointer = None


def _load_cfg_defaults(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing cfg defaults file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    defaults = data.get("cfg_defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("cfg_defaults.yaml must define a mapping under 'cfg_defaults'.")

    for key, value in defaults.items():
        setattr(cfg, key, value)

    return data


def _record_cells_for_mode(mode: int, allpops: list[str], recpops: list[str]):
    if mode == 0:
        return ["all"]
    if mode == 1:
        return [(pop, 0) for pop in allpops]
    if mode == 2:
        return [("IT2", 10), ("IT5A", 10), ("PT5B", 10), ("PV5B", 10), ("SOM5B", 10)]
    if mode == 3:
        return [(pop, 50) for pop in ["IT5A", "PT5B"]] + [("PT5B", x) for x in [393, 579, 19, 104]]
    if mode == 4:
        return (
            [(pop, 50) for pop in ["IT2", "IT4", "IT5A", "PT5B"]]
            + [("IT5A", x) for x in [393, 447, 579, 19, 104]]
            + [("PT5B", x) for x in [393, 447, 579, 19, 104, 214, 1138, 979, 799]]
        )
    if mode == 5:
        return [(pop, i) for pop in recpops for i in range(0, 100, int(100 / 50))]
    raise ValueError(f"Unsupported cfg.cellsrec={mode}")


cfg_blob = _load_cfg_defaults(DEFAULTS_PATH)
allpops = list(cfg_blob.get("allpops", []))
recpops = list(cfg_blob.get("recpops", []))
long_rate_templates = dict(cfg_blob.get("long_rate_templates", {}))

#------------------------------------------------------------------------------
# Active knobs: keep this section short and explicit.
#------------------------------------------------------------------------------
cfg.preTone = 1500
cfg.postTone = 1500
cfg.SimulateBaseline = True
cfg.addInVivoThalamus = False

cfg.cellsrec = 1
cfg.simLabel = "v103_tune3"
cfg.saveFolder = "./batchData/v103_manualTune"

cfg.pt5b_variant = "standard"  # "tim" or "standard"
cfg.heterozygous = False
cfg.blockNa = False
cfg.drugTreatment = False

cfg.addConn = 1
cfg.addSubConn = 1
cfg.addLongConn = 1
cfg.scaleDensity = 1.0

cfg.EEGain = 1.0
cfg.EIGain = 1.0
cfg.IEGain = 1.0
cfg.IIGain = 1.0
cfg.EICellTypeGain = {"PV": 1.0, "SOM": 1.0, "VIP": 1.0, "NGF": 1.0}
cfg.IEweights = [1.0, 1.0, 1.0]  # L2/3+4, L5, L6
cfg.IIweights = [1.0, 1.0, 1.0]  # L2/3+4, L5, L6

cfg.weightLong = {
    "TPO": 0.5,
    "TVL": 0.5,
    "S1": 0.5,
    "S2": 0.5,
    "cM1": 0.5,
    "M2": 0.5,
    "OC": 0.5,
}

cfg.addPulses = False
cfg.addIClamp = 0
cfg.addNetStim = 0

def refresh_derived_cfg_values() -> None:
    #------------------------------------------------------------------------------
    # Derived configuration
    #------------------------------------------------------------------------------
    if cfg.addInVivoThalamus:
        cfg.preTone = 1500
        cfg.postTone = 1500

    cfg.duration = cfg.preTone + cfg.postTone
    cfg.timeRanges = [cfg.duration - cfg.postTone, cfg.duration]
    cfg.printPopAvgRates = cfg.timeRanges
    cfg.recordStep = cfg.dt

    cfg.recordCells = _record_cells_for_mode(cfg.cellsrec, allpops, recpops)
    cfg.dendNa = 0.3 if cfg.pt5b_variant == "standard" else 0.3

    if cfg.loadmutantParams:
        raise ValueError("cfg.loadmutantParams is not implemented yet")

    cfg.mutations = []
    if cfg.heterozygous:
        cfg.mutations.append(
            {
                "label": "PT5B_full",
                "mech": "na12mut",
                "param": "gbar",
                "op": "set",
                "value": 0.0,
                "sections": "ALL",
                "only_if_present": {"mech": "na12mut"},
            }
        )

    if cfg.blockNa:
        cfg.mutations.extend(
            [
                {
                    "label": "PT5B_full",
                    "mech": "na12",
                    "param": "gbar",
                    "op": "set",
                    "value": 0.0,
                    "sections": "ALL",
                    "only_if_present": {"mech": "na12"},
                },
                {
                    "label": "PT5B_full",
                    "mech": "na12mut",
                    "param": "gbar",
                    "op": "set",
                    "value": 0.0,
                    "sections": "ALL",
                    "only_if_present": {"mech": "na12mut"},
                },
                {
                    "label": "PT5B_full",
                    "mech": "nax",
                    "param": "gbar",
                    "op": "set",
                    "value": 0.0,
                    "sections": "ALL",
                    "only_if_present": {"mech": "nax"},
                },
            ]
        )

    ih_quiet = float(long_rate_templates.get("ihQuiet", 1.0))
    ih_movement = float(long_rate_templates.get("ihMovement", 0.25))
    cfg.ihGbar = ih_quiet if cfg.SimulateBaseline else ih_movement

    cfg.modifyMechs = {
        "startTime": cfg.preTone,
        "endTime": cfg.duration,
        "cellType": "PT",
        "mech": "hd",
        "property": "gbar",
        "newFactor": 1.00,
        "origFactor": 0.75,
    }

    cfg.numCellsLong = int(1000 * cfg.scaleDensity)
    long_range_quiet = list(long_rate_templates.get("LongRangeQuiet", [0, 2.5]))
    tvl_quiet = list(long_rate_templates.get("TVLquiet", [0, 2.5]))
    tvl_movement = list(long_rate_templates.get("TVLmovement", [0, 10]))
    tvl_rates = tvl_quiet if cfg.SimulateBaseline else tvl_movement
    cfg.ratesLong = {
        "TPO": [0, 5],
        "TVL": tvl_rates,
        "S1": [0, 5],
        "S2": [0, 5],
        "cM1": long_range_quiet,
        "M2": long_range_quiet,
        "OC": [0, 5],
    }


refresh_derived_cfg_values()

os.makedirs(cfg.saveFolder, exist_ok=True)

#------------------------------------------------------------------------------
# Analysis and plotting
#------------------------------------------------------------------------------
with (PROJECT_ROOT / "cells" / "popColors.pkl").open("rb") as file_obj:
    popColors = pickle.load(file_obj)["popColors"]

cfg.analysis["plotRaster"] = {
    "include": allpops,
    "orderBy": ["pop", "y"],
    "timeRange": cfg.timeRanges,
    "saveFig": True,
    "showFig": False,
    "popRates": True,
    "orderInverse": True,
    "popColors": popColors,
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
    "oneFigPer": "cell",
    "figSize": (10, 4),
    "saveFig": True,
    "subtitles": True,
}

#------------------------------------------------------------------------------
# In vivo M1 and thalamus sampled neurons & spikes
#------------------------------------------------------------------------------
if cfg.addInVivoThalamus:
    baselineSpks, movementAndPostSpks, M1sampledCells, foldersName = defs.loadThalSpikes(
        str(PROJECT_ROOT), cfg, skipEmpty=False
    )

    trimmedBaseline = defs.trimTVLSpikes(baselineSpks, cfg)
    trimmedMovement = defs.trimTVLSpikes(movementAndPostSpks, cfg)

    cfg.numSampledCellsPerLayer = defs.average_dict_entries(M1sampledCells)
    cfg.spikeTimesInVivo = trimmedBaseline if cfg.SimulateBaseline else trimmedMovement

    del baselineSpks, movementAndPostSpks, trimmedBaseline, trimmedMovement
    gc.collect()
