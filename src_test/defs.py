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

from pathlib import Path
import yaml

def rateFitnessFuncTranges(simData, **kwargs):
        import numpy as np
        pops = kwargs['pops']
        maxFitness = kwargs['maxFitness']
        tranges = kwargs['tranges']

        popFitnessAll = []

        for trange in tranges:
            popFitnessAll.append([min(np.exp(abs(v['target'] - simData['popRates'][k]['%d_%d'%(trange[0], trange[1])])/v['width']), maxFitness) 
                if simData['popRates'][k]['%d_%d'%(trange[0], trange[1])] > v['min'] else maxFitness for k, v in pops.items()])
        
        popFitness = np.mean(np.array(popFitnessAll), axis=0)
        
        fitness = np.mean(popFitness)

        popInfo = '; '.join(['%s rate=%.1f fit=%1.f' % (p, np.mean(list(simData['popRates'][p].values())), popFitness[i]) for i,p in enumerate(pops)])
        print('  ' + popInfo)

        return fitness

#------------------------------------------------------------------------------
## Function to calculate the fitness according to required rate
def rateFitnessFunc(simData, extraConds=False, **kwargs):
    import numpy as np
    pops = kwargs['pops']
    maxFitness = kwargs['maxFitness']

    factor=1
    # Add extra conditions to the fitness. It 'breaks' the fitness function
    if extraConds:
        # check I > E in each layer
        condsIE_L23 = (simData['popRates']['PV2'] > simData['popRates']['IT2']) and (simData['popRates']['SOM2'] > simData['popRates']['IT2'])
        condsIE_L5A = (simData['popRates']['PV5A'] > simData['popRates']['IT5A']) and (simData['popRates']['SOM5A'] > simData['popRates']['IT5A'])
        condsIE_L5B = (simData['popRates']['PV5B'] > simData['popRates']['IT5B']) and (simData['popRates']['SOM5B'] > simData['popRates']['IT5B'])
        condsIE_L6 = (simData['popRates']['PV6'] > simData['popRates']['IT6']) and (simData['popRates']['SOM6'] > simData['popRates']['IT6'])
        # check E L5 > L6 > L2
        condEE562_0 = (simData['popRates']['IT5A']+simData['popRates']['IT5B']+simData['popRates']['PT5B'])/3 > (simData['popRates']['IT6']+simData['popRates']['CT6'])/2
        condEE562_1 = (simData['popRates']['IT6']+simData['popRates']['CT6'])/2 > simData['popRates']['IT2']
        # check PV > SOM in each layer
        condsPVSOM_L23 = (simData['popRates']['PV2'] > simData['popRates']['SOM2'])
        condsPVSOM_L5A = (simData['popRates']['PV5A'] > simData['popRates']['SOM5A'])
        condsPVSOM_L5B = (simData['popRates']['PV5B'] > simData['popRates']['SOM5B'])
        condsPVSOM_L6 = (simData['popRates']['PV6'] > simData['popRates']['SOM6'])

        conds = [condsIE_L23, condsIE_L5A, condsIE_L5B, condsIE_L6, condEE562_0, condEE562_1, condsPVSOM_L23, condsPVSOM_L5A, condsPVSOM_L5B, condsPVSOM_L6]

        if not all(conds): factor = 1.5
        
    popFitness = [min(np.exp(factor*abs(v['target'] - simData['popRates'][k])/v['width']), maxFitness) 
                if simData['popRates'][k] > v['min'] else maxFitness for k,v in pops.items()]
    fitness = np.mean(popFitness)

    popInfo = '; '.join(['%s rate=%.1f fit=%1.f'%(p, simData['popRates'][p], popFitness[i]) for i,p in enumerate(pops)])
    print('  '+popInfo)
    return fitness

#------------------------------------------------------------------------------
## Function to modify cell params during sim (e.g. modify PT ih)
def modifyMechsFunc(simTime, cfg):
    from netpyne import sim

    t = simTime

    cellType = cfg.modifyMechs['cellType']
    mech = cfg.modifyMechs['mech']
    prop = cfg.modifyMechs['property']
    newFactor = cfg.modifyMechs['newFactor']
    origFactor = cfg.modifyMechs['origFactor']
    factor = newFactor / origFactor
    change = False

    if cfg.modifyMechs['endTime']-1.0 <= t <= cfg.modifyMechs['endTime']+1.0:
        factor = origFactor / newFactor if abs(newFactor) > 0.0 else origFactor
        change = True

    elif t >= cfg.modifyMechs['startTime']-1.0 <= t <= cfg.modifyMechs['startTime']+1.0:
        factor = newFactor / origFactor if abs(origFactor) > 0.0 else newFactor
        change = True

    if change:
        print('   Modifying %s %s %s by a factor of %f' % (cellType, mech, prop, factor))
        for cell in sim.net.cells:
            if 'cellType' in cell.tags and cell.tags['cellType'] == cellType:
                for secName, sec in cell.secs.items():
                    if mech in sec['mechs'] and prop in sec['mechs'][mech]:
                        # modify python
                        sec['mechs'][mech][prop] = [g * factor for g in sec['mechs'][mech][prop]] if isinstance(sec['mechs'][mech][prop], list) else sec['mechs'][mech][prop] * factor

                        # modify neuron
                        for iseg, seg in enumerate(sec['hObj']):  # set mech params for each segment
                            if sim.cfg.verbose: print('   Modifying %s %s %s by a factor of %f' % (secName, mech, prop, factor))
                            setattr(getattr(seg, mech), prop, getattr(getattr(seg, mech), prop) * factor)
    return None

def addLongConnections(cwd, netParams, cfg):
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

# --- Sampling in-vivo thalamic spikes ---

def cellPerlayer(numbers):
    Layers = {'1': [0.0, 0.1*1350], '2': [0.1*1350,0.29*1350], '4': [0.29*1350,0.37*1350], '5A': [0.3*1350,0.47*1350], '5B': [0.47*1350,0.8*1350], '6': [0.8*1350, 1.0*1350]}

    from collections import defaultdict

    counts = defaultdict(int)
    for num in numbers:
        # Check if value is larger than layer 6's upper bound
        if num >= 1.0*1350:
            counts['6'] += 1
        else:
            for layer, (low, high) in Layers.items():
                if low <= num < high:
                    counts[layer] += 1
                    break  # Assumes one number belongs to only one layer

    return counts

def m1SampledDepths(cwd, layer_order = ["1", "23", "4", "5A", "5B", "6"]):
    import json

    data_file = Path(cwd) / "data/spikingData/m1_cell_depths_by_session.json"
    with open(data_file, "r") as file_obj:
        data = json.load(file_obj)

    if not isinstance(data, dict):
        raise ValueError(f"{data_file} must contain a JSON object mapping session -> depths list")

    M1sampledCells = {}
    sessionNames = []

    for session, cell_depths in data.items():
        if not isinstance(cell_depths, list):
            raise ValueError(f"Session '{session}' must contain a list of depths")

        counts = cellPerlayer(cell_depths)
        M1sampledCells[session] = {layer: int(counts.get(layer, 0)) for layer in layer_order}
        sessionNames.append(session)

    return M1sampledCells, sessionNames

def average_dict_entries(dicts: Union[List[Union[dict, defaultdict]], Dict[str, Union[dict, defaultdict]]]) -> Dict[str, float]:
    totals = defaultdict(int)
    counts = defaultdict(int)

    entries = dicts.values() if isinstance(dicts, dict) else dicts

    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"average_dict_entries expects dict entries, got {type(entry).__name__}")
        for key, value in entry.items():
            totals[key] += value
            counts[key] += 1

    averages = {key: int(totals[key] / counts[key]) for key in totals}
    return averages

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
    if bool(cfg.SimulateBaseline):
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

def loadThalSpikes(cwd, cfg, skipEmpty=False):
    import pandas

    data_file = Path(cwd) / "data/spikingData/thalamic_spikes_by_spkid_tone_onset_(-3.23, 1.44).csv"
    data = pandas.read_csv(data_file)

    required_cols = {"spkt", "spkid"}
    if not required_cols.issubset(set(data.columns)):
        raise ValueError(f"{data_file} must contain columns {required_cols}, found {list(data.columns)}")

    baseline_start = -3.0
    baseline_end = baseline_start + cfg.preTone / 1000.0
    if bool(cfg.SimulateBaseline) and baseline_end > -0.5:
        raise ValueError("Baseline end time exceeds -0.5 s, which may cause overlap with movement window. Decrease cfg.duration")

    if baseline_end <= baseline_start:
        raise ValueError(
            f"Invalid baseline period ({baseline_start}, {baseline_end}). "
            "Expected baseline_end > baseline_start."
        )

    pre_tone_s = cfg.preTone / 1000.0
    baseline_source_start = -pre_tone_s
    baseline_source_end = 0.0
    task_end = cfg.postTone / 1000.0

    spkid_col = data["spkid"].astype(int)
    spkt_col = data["spkt"].astype(float)
    available_spkids = spkid_col.dropna().unique().tolist()

    if len(available_spkids) == 0:
        raise ValueError(f"No spkid values found in {data_file}")

    random.seed(cfg.seeds["tvl_sampling"])
    k = cfg.numCellsLong
    if k <= len(available_spkids):
        sampled_spkids = random.sample(available_spkids, k=k)
    else:
        sampled_spkids = random.choices(available_spkids, k=k)

    spkt_by_spkid = defaultdict(list)
    for spkt, spkid in zip(spkt_col.tolist(), spkid_col.tolist()):
        spkt_by_spkid[int(spkid)].append(float(spkt))

    baselineSpks = []
    movementAndPostSpks = []
    for spkid in sampled_spkids:
        spkts = spkt_by_spkid[int(spkid)]

        # Use only preTone spikes as baseline source: [-preTone, 0] s.
        baseline_source_ms = [1000.0 * (t - baseline_source_start) for t in spkts if baseline_source_start <= t <= baseline_source_end]
        baseline_source_ms = sorted(baseline_source_ms)

        # Build (mirrored baseline, baseline) with contiguous windows:
        # mirrored in [0, preTone], baseline shifted in [preTone, 2*preTone].
        mirrored_ms = [cfg.preTone - t for t in baseline_source_ms]
        mirrored_ms = sorted(mirrored_ms)
        baseline_shifted_ms = [t + cfg.preTone for t in baseline_source_ms]
        baselineSpks_cell = mirrored_ms + baseline_shifted_ms

        # Optional task segment appended after mirrored+baseline.
        task_ms = []
        if cfg.postTone > 0:
            task_ms = [1000.0 * t + 2 * cfg.preTone for t in spkts if 0.0 <= t <= task_end]
            task_ms = sorted(task_ms)
        movement_ms = baselineSpks_cell + task_ms

        if skipEmpty:
            if len(baselineSpks_cell) > 0:
                baselineSpks.append(baselineSpks_cell)
            if len(movement_ms) > 0:
                movementAndPostSpks.append(movement_ms)
        else:
            baselineSpks.append(baselineSpks_cell)
            movementAndPostSpks.append(movement_ms)

    return baselineSpks, movementAndPostSpks

def trimTVLSpikes(spikeList, cfg):
    trimmedList = []
    for i in spikeList:
        # We need to align the spike time to avoid numerical errors in the delivery of the vecStim (due torounding errors it could happen that the simulator find a negative delivery time, which stops the simulation)
        trimmedList.append(np.unique([round(np.round(j / cfg.dt) * cfg.dt, 2) for j in i if (0<j<cfg.duration)]).tolist())

    return trimmedList

# -- Others --

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

# --- keep only properties for cell types whose LABELS are enabled in config/cells.yml ---

def _enabled_labels_from_yaml(p: str | Path) -> set[str]:
    with open(p, "r") as f:
        y = yaml.safe_load(f) or {}
    labels = y.get("enabled_cells", None)
    if not labels:
        return set()  # empty => allow all
    return {str(x) for x in labels}

def _labels_to_celltypes(netParams, labels: set[str]) -> set[str]:
    """
    Determine allowed cellTypes by looking up the loaded rules in netParams.cellParams
    whose labels are listed in `labels`. If labels is empty -> allow all discovered cellTypes.
    """
    allowed: set[str] = set()
    rules = getattr(netParams, "cellParams", {}) or {}
    if not labels:
        # No restriction: include all present types
        for rule in rules.values():
            ct = (rule.get("conds") or {}).get("cellType")
            if isinstance(ct, str):
                allowed.add(ct)
        return allowed

    for lbl in labels:
        rule = rules.get(lbl)
        if not isinstance(rule, dict):
            continue
        ct = (rule.get("conds") or {}).get("cellType")
        if isinstance(ct, str):
            allowed.add(ct)
    return allowed

def _filter_celltype_value(value, allowed: set[str]):
    if isinstance(value, list):
        kept = [ct for ct in value if ct in allowed]
        return kept if kept else None
    else:
        return value if (isinstance(value, str) and value in allowed) else None

def filter_by_enabled_cells_yaml(netParams, cells_yaml_path: str | Path, verbose: bool = True):
    """
    Prune netParams.* so ONLY cell types corresponding to labels listed in config/cells.yml
    remain referenced.

    - Reads `enabled_cells` (labels). If empty/missing -> keeps everything.
    - Maps those labels -> cellTypes via netParams.cellParams[*]['conds']['cellType'].
    - Removes pops whose pop['cellType'] not in allowed.
    - Trims or removes connParams/subConnParams when pre/post cellType(s) fall outside allowed.
    """
    enabled_labels = _enabled_labels_from_yaml(cells_yaml_path)
    allowed_types = _labels_to_celltypes(netParams, enabled_labels)

    report = {
        "enabled_labels": sorted(enabled_labels),
        "allowed_cellTypes": sorted(allowed_types),
        "removed_pops": [],
        "removed_conn_rules": [],
        "edited_conn_rules": 0,
        "removed_subconn_rules": [],
        "edited_subconn_rules": 0,
    }

    if verbose:
        if enabled_labels:
            print(f"[cells.yml] Enabled labels: {report['enabled_labels']}")
        else:
            print("[cells.yml] No labels specified -> no pruning.")
        print(f"[cells.yml] Allowed cellTypes resolved: {report['allowed_cellTypes']}")

    # Nothing to do if no restriction
    if not enabled_labels or not allowed_types:
        return report

    # 1) Populations
    for pop_label, spec in list((netParams.popParams or {}).items()):
        ct = spec.get("cellType")
        if isinstance(ct, str) and ct not in allowed_types:
            del netParams.popParams[pop_label]
            report["removed_pops"].append(pop_label)

    # Helper for conds filtering
    def _filter_conds(conds: dict) -> tuple[bool, dict]:
        if not isinstance(conds, dict) or "cellType" not in conds:
            return True, conds
        new_val = _filter_celltype_value(conds["cellType"], allowed_types)
        if new_val is None:
            return False, conds
        new = dict(conds)
        new["cellType"] = new_val
        return True, new

    # 2) Connection rules
    for rule_label, rule in list((netParams.connParams or {}).items()):
        pre_ok, pre_new = _filter_conds(rule.get("preConds", {}))
        post_ok, post_new = _filter_conds(rule.get("postConds", {}))
        if not pre_ok or not post_ok:
            del netParams.connParams[rule_label]
            report["removed_conn_rules"].append(rule_label)
        else:
            if pre_new is not rule.get("preConds") or post_new is not rule.get("postConds"):
                rule["preConds"] = pre_new
                rule["postConds"] = post_new
                report["edited_conn_rules"] += 1

    # 3) Subcellular rules
    for rule_label, rule in list((netParams.subConnParams or {}).items()):
        pre_ok, pre_new = _filter_conds(rule.get("preConds", {}))
        post_ok, post_new = _filter_conds(rule.get("postConds", {}))
        if not pre_ok or not post_ok:
            del netParams.subConnParams[rule_label]
            report["removed_subconn_rules"].append(rule_label)
        else:
            if pre_new is not rule.get("preConds") or post_new is not rule.get("postConds"):
                rule["preConds"] = pre_new
                rule["postConds"] = post_new
                report["edited_subconn_rules"] += 1

    if verbose:
        print(f"[cells.yml] Removed pops: {report['removed_pops']}")
        print(f"[cells.yml] Conn rules removed/edited: "
              f"{len(report['removed_conn_rules'])}/{report['edited_conn_rules']}")
        print(f"[cells.yml] SubConn rules removed/edited: "
              f"{len(report['removed_subconn_rules'])}/{report['edited_subconn_rules']}")

    return report
