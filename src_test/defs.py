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

from pathlib import Path
from cell_selection import load_cells_config, enabled_labels_from_config, resolve_enabled_populations

#------------------------------------------------------------------------------
## Function to calculate the fitness according to required rate
def rateFitnessFunc(simData, extraConds=False, **kwargs):
    import numpy as np
    pops = kwargs['pops']
    maxFitness = kwargs['maxFitness']
    pop_rates = simData.get('popRates', {})

    def _has(*pop_names):
        return all(pop_name in pop_rates for pop_name in pop_names)

    def _rate(pop_name):
        return pop_rates.get(pop_name)

    factor=1
    # Add extra conditions to the fitness. It 'breaks' the fitness function
    if extraConds:
        conds = []
        # check I > E in each layer
        if _has('PV2', 'SOM2', 'IT2'):
            conds.append((_rate('PV2') > _rate('IT2')) and (_rate('SOM2') > _rate('IT2')))
        if _has('PV5A', 'SOM5A', 'IT5A'):
            conds.append((_rate('PV5A') > _rate('IT5A')) and (_rate('SOM5A') > _rate('IT5A')))
        if _has('PV5B', 'SOM5B', 'IT5B'):
            conds.append((_rate('PV5B') > _rate('IT5B')) and (_rate('SOM5B') > _rate('IT5B')))
        if _has('PV6', 'SOM6', 'IT6'):
            conds.append((_rate('PV6') > _rate('IT6')) and (_rate('SOM6') > _rate('IT6')))

        # check E L5 > L6 > L2
        if _has('IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6'):
            conds.append(((_rate('IT5A') + _rate('IT5B') + _rate('PT5B')) / 3.0) > ((_rate('IT6') + _rate('CT6')) / 2.0))
        if _has('IT6', 'CT6', 'IT2'):
            conds.append(((_rate('IT6') + _rate('CT6')) / 2.0) > _rate('IT2'))

        # check PV > SOM in each layer
        if _has('PV2', 'SOM2'):
            conds.append(_rate('PV2') > _rate('SOM2'))
        if _has('PV5A', 'SOM5A'):
            conds.append(_rate('PV5A') > _rate('SOM5A'))
        if _has('PV5B', 'SOM5B'):
            conds.append(_rate('PV5B') > _rate('SOM5B'))
        if _has('PV6', 'SOM6'):
            conds.append(_rate('PV6') > _rate('SOM6'))

        if conds and not all(conds):
            factor = 1.5

    popFitness = []
    for pop_name, pop_cfg in pops.items():
        pop_rate = pop_rates.get(pop_name)
        if pop_rate is None or pop_rate <= pop_cfg['min']:
            popFitness.append(maxFitness)
            continue

        popFitness.append(min(np.exp(factor * abs(pop_cfg['target'] - pop_rate) / pop_cfg['width']), maxFitness))

    if not popFitness:
        return maxFitness

    fitness = np.mean(popFitness)

    popInfo = '; '.join(
        [
            '%s rate=%s fit=%1.f' % (pop_name, 'NA' if pop_rates.get(pop_name) is None else f'{pop_rates[pop_name]:.1f}', popFitness[i])
            for i, pop_name in enumerate(pops)
        ]
    )
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

# --- keep only properties for enabled populations and matching rule conditions ---

def _infer_cellmod_from_pop_params(netParams) -> dict[str, str]:
    inferred = {
        'IT2': 'HH_reduced',
        'IT4': 'HH_reduced',
        'IT5A': 'HH_full',
        'IT5B': 'HH_reduced',
        'PT5B': 'HH_full',
        'IT6': 'HH_reduced',
        'CT6': 'HH_reduced',
    }
    for pop_name in inferred:
        spec = (getattr(netParams, 'popParams', {}) or {}).get(pop_name, {})
        model = spec.get('cellModel') if isinstance(spec, dict) else None
        if isinstance(model, str):
            inferred[pop_name] = model
    return inferred


def _is_numeric(value) -> bool:
    return isinstance(value, (int, float))


def _is_numeric_range(value) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_numeric(v) for v in value)


def _ranges_overlap(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _ynorm_matches(pop_ynorm, cond_ynorm) -> bool:
    if cond_ynorm is None:
        return True
    if pop_ynorm is None:
        return True
    if _is_numeric(cond_ynorm):
        return pop_ynorm[0] <= cond_ynorm <= pop_ynorm[1]
    if _is_numeric_range(cond_ynorm):
        return _ranges_overlap(pop_ynorm, cond_ynorm)
    if isinstance(cond_ynorm, (list, tuple)):
        return any(_ynorm_matches(pop_ynorm, item) for item in cond_ynorm)
    return True


def _scalar_or_list_matches(value, condition) -> bool:
    if isinstance(condition, (list, tuple)) and not _is_numeric_range(condition):
        return value in condition
    return value == condition


def _pop_matches_conds(pop_name: str, pop_spec: dict, conds: dict) -> bool:
    if not isinstance(conds, dict):
        return True

    if 'pop' in conds and not _scalar_or_list_matches(pop_name, conds['pop']):
        return False

    if 'cellType' in conds:
        cell_type = pop_spec.get('cellType')
        if cell_type is None or not _scalar_or_list_matches(cell_type, conds['cellType']):
            return False

    if 'cellModel' in conds:
        cell_model = pop_spec.get('cellModel')
        if cell_model is None or not _scalar_or_list_matches(cell_model, conds['cellModel']):
            return False

    if 'ynorm' in conds:
        pop_ynorm = pop_spec.get('ynormRange')
        if not _ynorm_matches(pop_ynorm, conds['ynorm']):
            return False

    return True


def _tighten_value(condition, allowed_values):
    if isinstance(condition, list):
        kept = [value for value in condition if value in allowed_values]
        return (len(kept) > 0), kept
    if isinstance(condition, tuple):
        kept = [value for value in condition if value in allowed_values]
        return (len(kept) > 0), kept
    return (condition in allowed_values), condition


def _tighten_conds(conds: dict, matched_pop_names: list[str], active_pops: dict[str, dict]) -> tuple[bool, dict]:
    if not isinstance(conds, dict):
        return True, conds

    if not matched_pop_names:
        return False, conds

    matched_specs = [active_pops[pop_name] for pop_name in matched_pop_names]
    allowed_pop_names = set(matched_pop_names)
    allowed_cell_types = {spec.get('cellType') for spec in matched_specs if isinstance(spec.get('cellType'), str)}
    allowed_cell_models = {spec.get('cellModel') for spec in matched_specs if isinstance(spec.get('cellModel'), str)}

    new_conds = dict(conds)

    if 'pop' in new_conds:
        ok, tightened = _tighten_value(new_conds['pop'], allowed_pop_names)
        if not ok:
            return False, conds
        new_conds['pop'] = tightened

    if 'cellType' in new_conds:
        ok, tightened = _tighten_value(new_conds['cellType'], allowed_cell_types)
        if not ok:
            return False, conds
        new_conds['cellType'] = tightened

    if 'cellModel' in new_conds:
        ok, tightened = _tighten_value(new_conds['cellModel'], allowed_cell_models)
        if not ok:
            return False, conds
        new_conds['cellModel'] = tightened

    return True, new_conds


def _prune_rule_set(rules: dict, active_pops: dict[str, dict], report: dict, removed_key: str, edited_key: str):
    for rule_label, rule in list((rules or {}).items()):
        pre_conds = rule.get('preConds', {})
        post_conds = rule.get('postConds', {})

        pre_matches = [
            pop_name for pop_name, pop_spec in active_pops.items()
            if _pop_matches_conds(pop_name, pop_spec, pre_conds)
        ]
        post_matches = [
            pop_name for pop_name, pop_spec in active_pops.items()
            if _pop_matches_conds(pop_name, pop_spec, post_conds)
        ]

        if not pre_matches or not post_matches:
            del rules[rule_label]
            report[removed_key].append(rule_label)
            continue

        pre_ok, pre_new = _tighten_conds(pre_conds, pre_matches, active_pops)
        post_ok, post_new = _tighten_conds(post_conds, post_matches, active_pops)

        if not pre_ok or not post_ok:
            del rules[rule_label]
            report[removed_key].append(rule_label)
            continue

        if pre_new != pre_conds or post_new != post_conds:
            rule['preConds'] = pre_new
            rule['postConds'] = post_new
            report[edited_key] += 1


def filter_by_enabled_cells_yaml(netParams, cells_yaml_path: str | Path, cellmod: dict | None = None, verbose: bool = True):
    cell_cfg = load_cells_config(Path(cells_yaml_path))
    enabled_labels = enabled_labels_from_config(cell_cfg)

    report = {
        'enabled_labels': sorted(enabled_labels),
        'removed_pops': [],
        'removed_conn_rules': [],
        'edited_conn_rules': 0,
        'removed_subconn_rules': [],
        'edited_subconn_rules': 0,
    }

    if cellmod is None:
        cellmod = _infer_cellmod_from_pop_params(netParams)

    if enabled_labels:
        allowed_local_pops = set(resolve_enabled_populations(enabled_labels, cellmod))
        for pop_label, spec in list((netParams.popParams or {}).items()):
            if not isinstance(spec, dict):
                continue
            if spec.get('cellModel') == 'VecStim':
                continue
            if pop_label not in allowed_local_pops:
                del netParams.popParams[pop_label]
                report['removed_pops'].append(pop_label)

    active_pops = {k: v for k, v in (netParams.popParams or {}).items() if isinstance(v, dict)}

    _prune_rule_set(netParams.connParams, active_pops, report, 'removed_conn_rules', 'edited_conn_rules')
    _prune_rule_set(netParams.subConnParams, active_pops, report, 'removed_subconn_rules', 'edited_subconn_rules')

    if verbose:
        if enabled_labels:
            print(f"[cells.yml] Enabled labels: {report['enabled_labels']}")
        else:
            print('[cells.yml] No labels specified -> no explicit pop pruning.')
        print(f"[cells.yml] Removed pops: {report['removed_pops']}")
        print(f"[cells.yml] Conn rules removed/edited: {len(report['removed_conn_rules'])}/{report['edited_conn_rules']}")
        print(f"[cells.yml] SubConn rules removed/edited: {len(report['removed_subconn_rules'])}/{report['edited_subconn_rules']}")

    return report
