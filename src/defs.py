"""
defs.py

Definition of the cells and auxiliar functions used in the model

Contributors: romanbaravalle@gmail.com
"""
from netpyne import specs
import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Union
import random

def addLongConnections(cwd, netParams, cfg):
    #TODO: Check and rewrite to load in-vivo spikes
    import pickle, json
    ## load experimentally based parameters for long range inputs
    with open(cwd + '/conn/conn_long.pkl', 'rb') as fileObj:
        connLongData = pickle.load(fileObj)
    # ratesLong = connLongData['rates']

    numCells = cfg.numCellsLong
    noise = cfg.noiseLong
    start = cfg.startLong

    if cfg.addInVivoThalamus: 
        longPops = ['TPO', 'S1', 'S2', 'cM1', 'M2', 'OC']
    else:
        longPops = ['TPO', 'TVL', 'S1', 'S2', 'cM1', 'M2', 'OC']
    ## create populations with fixed
    for longPop in longPops:
        netParams.popParams[longPop] = {'cellModel': 'VecStim', 'numCells': numCells, 'rate': cfg.ratesLong[longPop],
                                        'noise': noise, 'start': start, 'pulses': [],
                                        'ynormRange': cfg.layer['long' + longPop]}
        if isinstance(cfg.ratesLong[longPop], str):  # filename to load spikes from
            spikesFile = cfg.ratesLong[longPop]
            with open(spikesFile, 'r') as f: spks = json.load(f)
            netParams.popParams[longPop].pop('rate')
            netParams.popParams[longPop]['spkTimes'] = spks

    if cfg.addInVivoThalamus:   
        netParams.popParams['TVL'] = {'cellModel': 'VecStim',
                                                 'numCells': len(cfg.spikeTimesInVivo),
                                                 'spkTimes': cfg.spikeTimesInVivo,
                                                 'ynormRange': cfg.layer['long' + 'TVL']}
    return connLongData

def SampleSpikes(spikeTimesList, cfg, preTone=-2., postTone=2, baselineEnd=-0.5, skipEmpty=False):
    # Guard rails for movement/post windows when not simulating baseline
    if (cfg.SimulateBaseline == False and cfg.preTone > 2000.):
        raise ValueError("cfg.preTone cannot be larger than 2000 ms")
    if (cfg.SimulateBaseline == False and cfg.postTone > 2000.):
        raise ValueError("cfg.postTone cannot be larger than 2000 ms")

    MovementTrials = []
    BaselineTrials = []
    for spkList in spikeTimesList:
        MovementTrialsAux = []
        BaselineTrialsAux = []
        for spkTimes in spkList:
            # Baseline window: [preTone, baselineEnd] in seconds
            if (preTone <= spkTimes <= baselineEnd):
                # store as ms, baseline window re-zeroed to 0 at preTone
                BaselineTrialsAux.append(1000 * (spkTimes + abs(preTone)))
            # Movement + post window re-zeroed to cfg.preTone
            if (-cfg.preTone/1000. <= spkTimes <= cfg.postTone/1000.):
                PositiveTimes = 1000 * spkTimes + cfg.preTone
                MovementTrialsAux.append(PositiveTimes)

        if skipEmpty:
            if len(MovementTrialsAux) > 0: MovementTrials.append(MovementTrialsAux)
            if len(BaselineTrialsAux) > 0: BaselineTrials.append(BaselineTrialsAux)
        else:
            MovementTrials.append(MovementTrialsAux)
            BaselineTrials.append(BaselineTrialsAux)

    # Sample spikes
    random.seed(cfg.seeds['tvl_sampling'])
    baselineSpks = random.choices(BaselineTrials, k=cfg.numCellsLong)
    baselineSpks = [list(i) for i in baselineSpks]

    movementAndPostSpks = random.choices(MovementTrials, k=cfg.numCellsLong)
    movementAndPostSpks = [list(i) for i in movementAndPostSpks]

    # --- New behavior for baseline simulation: mirror + original, exactly two copies ---
    if cfg.SimulateBaseline is True:
        # one baseline copy span in ms
        sampledSpikesSpan = int(round(1000 * (baselineEnd - preTone)))
        if sampledSpikesSpan <= 0:
            raise ValueError("Baseline window must have positive duration (baselineEnd must be > preTone).")

        twice_span = 2 * sampledSpikesSpan
        # Enforce: duration cannot exceed the two concatenated copies
        if cfg.duration > twice_span:
            raise ValueError(
                f"cfg.duration ({cfg.duration} ms) exceeds twice the baseline copy ({twice_span} ms). "
                "Mirror+original creates exactly two copies; reduce cfg.duration or widen the baseline window."
            )

        def mirror_then_original(lst):
            # keep only times that fall within a single copy window [0, span)
            base = [t for t in lst if 0 <= t < sampledSpikesSpan]

            # mirror around the right edge of the first copy: t' = span - t
            mirrored = [sampledSpikesSpan - t for t in base]
            # keep mirrored within (0, span] to avoid negatives; allow 'span' then clip later to duration
            mirrored = [t for t in mirrored if 0 < t <= sampledSpikesSpan]

            # original shifted to the second copy: [span, 2*span)
            shifted = [t + sampledSpikesSpan for t in base]

            # concatenate, order, dedupe
            out = sorted(set(mirrored + shifted))

            # hard-clip to cfg.duration (if duration < 2*span)
            return [t for t in out if 0 <= t <= cfg.duration]

        baselineSpks = [mirror_then_original(trial) for trial in baselineSpks]

    # Note: movementAndPostSpks unchanged
    return baselineSpks, movementAndPostSpks

def cellPerlayer(numbers):
    Layers = {'1': [0.0, 0.1*1350], '2': [0.1*1350,0.29*1350], '4': [0.29*1350,0.37*1350], '5A': [0.3*1350,0.47*1350], '5B': [0.47*1350,0.8*1350], '6': [0.8*1350, 1.0*1350]}

    from collections import defaultdict

    counts = defaultdict(int)

    for num in numbers:
        for layer, (low, high) in Layers.items():
            if low <= num < high:
                counts[layer] += 1
                break  # Assumes one number belongs to only one layer

    return counts

def loadThalSpikes(cwd, cfg, skipEmpty=False):
    import json
    with open(cwd+"/data/spikingData/ThRates.json", "r") as fileObj:
        data = json.loads(fileObj.read())

    spikeTimesList = []
    M1sampledCells = []
    foldersName = []

    for folder in data.keys():
        for i in range(len(data[folder].keys())-4): # exclude M1_cell_depths, Th_cell_depths, meanRate, stdRate
            spkid =  data[folder]['trial_%d' % i]['spkid']
            spkt = data[folder]['trial_%d' % i]['spkt']
            npre = int(np.max(spkid)) + 1
            spkTimes_by_cell = [[] for _ in range(npre)]
            for t, i in zip(spkt, spkid):
                spkTimes_by_cell[int(i)].append(float(t))
            spikeTimesList[len(spikeTimesList):] += spkTimes_by_cell
        cellDepths = data[folder]['M1_cell_depths']
        counts = cellPerlayer(cellDepths)
        M1sampledCells.append(counts)
        foldersName.append(folder)

    baselineSpks, movementAndPostSpks = SampleSpikes(spikeTimesList, cfg, skipEmpty=skipEmpty)

    return baselineSpks, movementAndPostSpks, M1sampledCells, foldersName

def average_dict_entries(dicts: List[Union[dict, defaultdict]]) -> Dict[str, float]:
    totals = defaultdict(int)
    counts = defaultdict(int)

    for entry in dicts:
        for key, value in entry.items():
            totals[key] += value
            counts[key] += 1

    averages = {key: int(totals[key] / counts[key]) for key in totals}
    return averages

def trimTVLSpikes(spikeList, cfg):
    trimmedList = []
    for i in spikeList:
        # We need to align the spike time to avoid numerical errors in the delivery of the vecStim (due torounding errors it could happen that the simulator find a negative delivery time, which stops the simulation)
        trimmedList.append(np.unique([round(np.round(j / cfg.dt) * cfg.dt, 2) for j in i if (0<j<cfg.duration)]).tolist())

    return trimmedList

def strip_range_like_globals(rule: dict) -> None:
    """
    Remove globals that look like RANGE vars (e.g., hinf_catcb, minf_catcb).
    These cannot be assigned via NetPyNE 'globals' because they require a section context.
    """
    globs = rule.get("globals", {})
    # Hard drop the known offenders
    for k in ("hinf_catcb", "minf_catcb"):
        globs.pop(k, None)
    # Generic safety: drop any <name>_<mech> where <name> is a common RANGE pattern
    range_like_prefixes = ("hinf", "minf")  # extend if needed
    for k in list(globs.keys()):
        parts = k.rsplit("_", 1)
        if len(parts) == 2 and parts[0] in range_like_prefixes:
            globs.pop(k, None)