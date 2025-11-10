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
        # Check if value is larger than layer 6's upper bound
        if num >= 1.0*1350:
            counts['6'] += 1
        else:
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

# ------- CEBRA functions --------

def sampleNeuronsFromModel(sim, cfg, plot=False):
    """
    Sample a given number of cells from each layer and plot in 3D.

    Parameters
    ----------
    sim : NetPyNE simulation object (with sim.net.cells populated)
    samples_to_pick : dict
        Keys are layer names (e.g., 'L1', 'L2/3', 'L4', 'L5', 'L6'),
        Values are number of cells to sample from each layer.

    Returns
    -------
    sampled_cells : dict
        Keys are layer names, values are list of sampled cell dicts
        with 'gid' and 'pos'.
    """
    # Sample neurons from the model following the same sampling of layers as in cfg.numSampledCellsPerLayer. Use same random seed to sample always the same neurons
    import random
    random.seed(cfg.seeds['m1_sampling'])
    
    # --- Helper to map yNorm -> layer ---
    def get_layer(y_norm):
        for layer, (ymin, ymax) in cfg.layer.items():
            if ymin <= y_norm < ymax:
                return layer
        return None
    
    # --- Collect cells by layer ---
    cells_by_layer = {layer: [] for layer in cfg.layer.keys()}
    for cell in sim.net.cells:
        y_norm = cell.tags.get('ynorm')
        if y_norm is None:
            continue
        layer = get_layer(y_norm)
        if layer:
            cells_by_layer[layer].append(cell.gid)

    # --- Sample from each layer ---
    sampled_cells = {}
    for layer, n in cfg.numSampledCellsPerLayer.items():
        if layer in cells_by_layer:
            sampled_cells[layer] = random.sample(
                cells_by_layer[layer],
                min(n, len(cells_by_layer[layer]))
            )
        else:
            sampled_cells[layer] = []

    if plot:
        import matplotlib.pyplot as plt
        # --- Plot sampled cells ---
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.tab10.colors  # distinct colors per layer
        for i, (layer, cells) in enumerate(sampled_cells.items()):
            xs = [c['pos'][0] for c in cells]
            ys = [c['pos'][1] for c in cells]
            zs = [c['pos'][2] for c in cells]
            ax.scatter(xs, ys, zs, label=f"Layer {layer}",
                    color=colors[i % len(colors)], s=40)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Sampled Cell Locations by Layer (using yNorm)")
        ax.legend()
        plt.show()

    return sampled_cells

def bin_spikes(spike_times, dt, wdw_start, wdw_end):
    # Function that puts spikes into bins
    edges = np.arange(wdw_start, wdw_end, dt)  # Get edges of time bins
    num_bins = edges.shape[0] - 1  # Number of bins
    num_neurons = spike_times.shape[0]  # Number of neurons
    neural_data = np.empty([num_bins, num_neurons])  # Initialize array for binned neural data
    # Count number of spikes in each bin for each neuron, and put in array
    for i in range(num_neurons):
        neural_data[:, i] = np.histogram(spike_times[i], edges)[0]
    return neural_data

def smoothen_spikes(neural_data, kernel_size=23, sigma=2.5): # kernel_size should be an odd number
    import scipy
    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # Normalize to preserve unit area
    
    # Repeat the kernel to match the number of neurons (M)
    kernel = np.tile(kernel, (neural_data.shape[0], 1))
    
    conv = scipy.signal.fftconvolve(neural_data, kernel, mode="same", axes=0)
    # conv = scipy.signal.fftconvolve(neural_data, kernel, mode="same") # wrong way of smoothing, but worked good - need to find why
    
    return conv

def binnedRaster(simData, cfg):
    fs = 1/cfg.recordStep
    bin_time = cfg.CEBRA_params['dt']
    spike_times = np.array(simData['spkt'].to_python())
    spike_ids = np.array(simData['spkid'].to_python())
    sampledCells = [j for i in cfg.sampled_cells.values() for j in i]
    sampledCells.sort()

    # Convert transient time to seconds for consistency
    transient_sec = cfg.transient / 1000.

    # print(sampledCells)
    spike_timesAux = []
    for i in sampledCells:
        cell_spikes = spike_times[spike_ids == i] / 1000.  # Convert to seconds
        # Filter out spikes during transient period
        cell_spikes_filtered = cell_spikes[cell_spikes >= transient_sec]
        spike_timesAux.append(cell_spikes_filtered)
    spike_times = np.array(spike_timesAux, dtype=object)

    # Start analysis from transient time, end at original duration
    start_time = transient_sec
    end_time = cfg.duration / 1000. + bin_time*fs/1000.
    Raster = bin_spikes(spike_times, bin_time*fs/1000., start_time, end_time).T
    neural_data_smooth = smoothen_spikes(Raster)

    # plot_data(neural_data, neural_data_smooth, all_aux)
    Raster = neural_data_smooth
    # Convert to rate
    Raster /= bin_time
    return Raster

def linear_fit(emb1, emb2):
    from sklearn.linear_model import LinearRegression
    fit = LinearRegression().fit(emb1, emb2)
    # coef, intercept = fit.coef_, fit.intercept_
    r_2 = fit.score(emb1, emb2)
    return r_2