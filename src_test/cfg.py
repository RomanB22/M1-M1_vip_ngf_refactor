"""
cfg.py 

Simulation configuration for M1 model (using NetPyNE)

Contributors: salvadordura@gmail.com
"""

from netpyne import specs

try:
    from netpyne.batchtools.runners import Runner_SimConfig as BatchRunnerSimConfig
except Exception:
    BatchRunnerSimConfig = None

import pickle
from pathlib import Path
import defs
import gc

cwd = str(Path.cwd())
cfg = BatchRunnerSimConfig() if BatchRunnerSimConfig is not None else specs.SimConfig()

#------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
#------------------------------------------------------------------------------
cfg._batchtk_label_pointer = None
cfg._batchtk_path_pointer = None

#------------------------------------------------------------------------------
# Run parameters
#------------------------------------------------------------------------------
cfg.diversity = True
cfg.pasUniformFrac = 0.30   # +/-10% variation in somatic passive properties in the populations

cfg.preTone = 1000
cfg.postTone = 1500 # Movement part
cfg.SimulateBaseline = True
cfg.addInVivoThalamus = False # To add the sampled spike times from in-vivo recordings on TVL
if bool(cfg.SimulateBaseline) is True:
    cfg.postTone = 0
cfg.duration = 2 * cfg.preTone + cfg.postTone
cfg.dt = 0.025
cfg.seeds = {'conn': 4321, 'stim': 1234, 'loc': 4321, 'tvl_sampling': 1234, 'cell': 1234} 
cfg.hParams = {'celsius': 34, 'v_init': -80}  
cfg.verbose = False
cfg.createNEURONObj = True
cfg.createPyStruct = True
cfg.connRandomSecFromList = False  # set to false for reproducibility 
cfg.cvode_active = False
cfg.cvode_atol = 1e-6
cfg.cache_efficient = True
cfg.printRunTime = 0.1
cfg.printSynsAfterRule = False
cfg.pt3dRelativeToCellLocation = True
cfg.oneSynPerNetcon = True  # only affects conns not in subconnParams; produces identical results
cfg.validateNetParams = True
cfg.progressBar = 0

cfg.includeParamsLabel = False
cfg.timeRanges = [cfg.duration-cfg.postTone-cfg.preTone, cfg.duration]
step = 250

cfg.printPopAvgRates = [
    [t, min(t + step, cfg.timeRanges[1])]
    for t in range(cfg.timeRanges[0], cfg.timeRanges[1], step)
]

cfg.checkErrors = False
cfg.checkErrorsVerbose = False

cfg.rand123GlobalIndex = None
cfg.coreneuron = True
cfg.random123 = True
cfg.gpu = False

#------------------------------------------------------------------------------
# Recording 
#------------------------------------------------------------------------------
allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2',
           'IT4', 'PV4', 'SOM4', 'VIP4', 'NGF4',
           'IT5A', 'PV5A', 'SOM5A','VIP5A','NGF5A',
           'IT5B', 'PT5B', 'PV5B', 'SOM5B','VIP5B','NGF5B',
           'IT6','CT6','PV6','SOM6','VIP6','NGF6']
# allpops = ['PT5B']
recpops = ['PV2', 'PV4', 'PV5A', 'PV5B', 'PV6', 'PT5B']
cfg.cellsrec = 1
if cfg.cellsrec == 0:  cfg.recordCells = ['all'] # record all cells
elif cfg.cellsrec == 1: cfg.recordCells = [(pop,0) for pop in allpops] # record one cell of each pop
elif cfg.cellsrec == 2: cfg.recordCells = [('IT2',10), ('IT5A',10), ('PT5B',10), ('PV5B',10), ('SOM5B',10)] # record selected cells
elif cfg.cellsrec == 3: cfg.recordCells = [(pop,50) for pop in ['IT5A', 'PT5B']]+[('PT5B',x) for x in [393,579,19,104]] #,214,1138,799]] # record selected cells # record selected cells
elif cfg.cellsrec == 4: cfg.recordCells = [(pop,50) for pop in ['IT2', 'IT4', 'IT5A', 'PT5B']] \
										+ [('IT5A',x) for x in [393,447,579,19,104]] \
										+ [('PT5B',x) for x in [393,447,579,19,104,214,1138,979,799]] # record selected cells
elif cfg.cellsrec == 5: cfg.recordCells =  [(pop, i) for pop in recpops for i in range(0,100,int(100/50))] # record 5 cells of each selected pop

cfg.recordStim = False
cfg.recordTime = False  
cfg.recordStep = cfg.dt

#------------------------------------------------------------------------------
# Saving
#------------------------------------------------------------------------------
cfg.simLabel = 'v103_tune3'
cfg.saveFolder = './batchData/v103_manualTune'
cfg.savePickle = False
cfg.saveJson = False
cfg.saveDataInclude = ['simData', 'simConfig', 'netParams']#, 'net']
cfg.backupCfgFile = None #['cfg.py', 'backupcfg/'] 
cfg.gatherOnlySimData = False
cfg.saveCellSecs = False
cfg.saveCellConns = 0
cfg.compactConnFormat = 0

import os
os.makedirs(cfg.saveFolder, exist_ok=True)

#------------------------------------------------------------------------------
# Analysis and plotting 
#------------------------------------------------------------------------------
with open('./cells/popColors.pkl', 'rb') as fileObj: popColors = pickle.load(fileObj)['popColors']

# allpops = ['TVL']

cfg.analysis['plotRaster'] = {'include': allpops, 'orderBy': ['pop', 'y'], 'timeRange': cfg.timeRanges,
                             'saveFig': True, 'showFig': False, 'popRates': True, 
                             'orderInverse': True, 'popColors': popColors, 'figSize': (12,18), 'lw': 0.3,
                             'markerSize':3, 'marker': '.', 'dpi': 300} 

cfg.recordTraces = {'V_soma': {'sec':'soma', 'loc':0.5, 'var':'v'}, 
                    'V_apic_3': {'sec':'apic_3', 'loc':0.5, 'var':'v', 'conds':{'pop': 'PT5B'}},
                    'V_dend_1': {'sec':'dend_1', 'loc':0.5, 'var':'v', 'conds':{'pop': 'PT5B'}}}

cfg.analysis['plotTraces'] = {'include': cfg.recordCells, 'timeRange': cfg.timeRanges, 
								'overlay': True, 'oneFigPer': 'trace', 'figSize': (10,4), 
								'saveFig': True, 'subtitles': True, 'legend': True} 

# cfg.analysis['plotTraces'] = {'include': cfg.recordCells, 'timeRange': cfg.timeRanges, 
# 								'overlay': True, 'oneFigPer': 'cell', 'figSize': (10,4), 
# 								'saveFig': True, 'subtitles': True, 'legend': True} 

# Recording
# old sim/ version already used this same LFP layout
# center of the 300 x 1350 x 300 um column
cfg.recordLFP = [[150, y, 150] for y in range(200, 1300, 100)]

# EEG in NetPyNE comes from dipole recording, not from recordLFP
cfg.recordDipole = True

# optional: cheaper for LFP/EEG if runtime or memory gets heavy
# cfg.recordStep = 0.1

# Analysis
cfg.analysis['plotLFP'] = {
    'plots': ['timeSeries', 'PSD', 'spectrogram', 'locations'],   # add 'PSD', 'spectrogram', 'locations' if needed
    'electrodes': [10],        # with the range above, 10 is the deepest electrode
    'timeRange': cfg.timeRanges, 'maxFreq': 80, 'figSize': (8, 4), 'saveData': False, 'saveFig': True, 'showFig': False}

cfg.analysis['plotDipole'] = {'timeRange': cfg.timeRanges, 'saveFig': True, 'showFig': False}

cfg.analysis['plotEEG'] = {'timeRange': cfg.timeRanges, 'saveFig': True, 'showFig': False}

#------------------------------------------------------------------------------
# Cells
#------------------------------------------------------------------------------
cfg.cao_secs = 1.2

cfg.pt5b_variant = "tim"        # "tim" or "standard"
cfg.cellModelLoadMode = "saved" # default mode: "saved" or "source"
# Per-cell-rule override (keys are cell rule labels used in netParams, not population names).
# Set each entry to "saved" (load pre-saved if available) or "source" (force import from source).
# cfg.cellModelLoadModeByLabel = {
#     "IT2_reduced": "saved",
#     "IT4_reduced": "saved",
#     "IT5A_reduced": "saved",
#     "IT5B_reduced": "saved",
#     "PT5B_reduced": "saved",
#     "IT6_reduced": "saved",
#     "CT6_reduced": "saved",
#     "SOM_reduced": "saved",
#     "IT5A_full": "saved",
#     "PV_reduced": "saved",
#     "VIP_reduced": "saved",
#     "NGF_reduced": "saved",
#     "PT5B_full": "saved",
# }

cfg.dendNa = 0.3 if cfg.pt5b_variant=="standard" else 1.0 # 0.3 for "standard", 1.0 for "tim"

cfg.spikeGuard = {
    'enabled': True,
    'pointpName': 'spike_guard',
    'mod': 'SpikeGuard',
    'vref': 'detector_v',
    'candidateStartMv': -20.0,
    'plateauMv': -40.0,
    'plateauMs': 100.0,
    'thresholdForDetectorV': 0.5,
    'lossPenaltyPerBlockedPop': 250.0,
    'blockedFractionThreshold': 0.10,
    'blockedMinCells': 3,
    'blockedNoSpikesMinRejected': 5,
    'blockedRejectedScale': 3,
    'blockedRejectedOffset': 5,
    'families': {
        'exc': {
            'minPeakMv': 10.0,
            'minProminenceMv': 20.0,
            'minDvdtMvPerMs': 10.0,
            'refractoryMs': 2.0,
        },
        'inh': {
            'minPeakMv': 0.0,
            'minProminenceMv': 15.0,
            'minDvdtMvPerMs': 8.0,
            'refractoryMs': 1.0,
        },
    },
    'cellTypeOverrides': {},
}

cfg.loadmutantParams = False # MUTANT LOADING IS CHANGING SOMETHING
cfg.variant = 'WT' # L1666F, E1211K, D195G, R853Q, K1422E, M1879T, WT

if cfg.loadmutantParams: ValueError("cfg.loadmutantParams is not implemented yet")

# --- toggles ---
cfg.heterozygous = False   # zero out na12mut
cfg.blockNa      = False   # zero out na12 & na12mut
cfg.KCNT1        = False  # apply KCNT1 mutation effects

# --- mutation engine toggles (optional) ---
cfg.mutations_enabled = False
cfg.mutations_dry_run = False
# They serve different needs: one controls whether we mutate at all; the other controls how (apply vs. preview). You can combine them:
# enabled=True, dry_run=True → show planned changes, change nothing.
# enabled=True, dry_run=False → apply changes.
# enabled=False → ignore mutations entirely (dry_run is irrelevant).

# --- base mutation list (you can add more here) ---
cfg.mutations = []

# HETEROZYGOUS: zero na12mut.gbar everywhere it exists
if cfg.heterozygous:
    cfg.mutations += [
        {
            "label": "PT5B_full",
            "mech": "na12mut",
            "param": "gbar",
            "op": "set",
            "value": 0.0,
            "sections": "ALL",
            "only_if_present": {"mech": "na12mut"},
        }
    ]

# BLOCK Na: zero both na12.gbar and na12mut.gbar everywhere they exist
if cfg.blockNa:
    cfg.mutations += [
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

cfg.drugTreatment = False
cfg.verbose_drug_changes = False
cfg.drug_dry_run = False  # If True, will only print planned changes without applying them
# Example “intrinsic” sheet rows
cfg.cell_drugs = [
    # Block sodium in PV and SOM (e.g., TTX): set gbar/gmax to 0 for all sodium mechs you use
    {"cell_types": ["PV","SOM"], "mech": "nax",   "param": "gbar", "op": "set",   "value": 0.0, "sections": "ALL"},
    {"cell_types": ["PT","IT"],  "mech": "nax",  "param": "gbar", "op": "set",   "value": 0.0, "sections": "ALL"},
    {"cell_types": ["PT","IT"],  "mech": "na12",  "param": "gbar", "op": "set",   "value": 0.0, "sections": "ALL"},
    {"cell_types": ["PT","IT"],  "mech": "na12mut","param":"gbar", "op": "set",   "value": 0.0, "sections": "ALL"},

    # Ih blocker (ZD7288-like) on VIP only: scale to 20%
    {"cell_types": ["VIP"], "mech": "Ih", "param": "gIhbar", "op": "scale", "value": 0.2, "sections": "ALL"},

    # Boost HCN on NGF dendrite only
    {"cell_types": ["NGF"], "mech": "hd", "param": "gbar", "op": "scale", "value": 1.5, "sections": ["dend"]},

    # Reduce L-type Ca on SOM soma (catcb example would be 'gcatbar' etc.)
    {"cell_types": ["SOM"], "mech": "catcb", "param": "gcatbar", "op": "scale", "value": 0.5, "sections": ["soma"]},
]

# Example “synaptic” sheet rows
cfg.syn_drugs = [
    # Ketamine-like NMDA reduction: halve tau2NMDA and reduce gain (example)
    {"syn_mechs": ["NMDA"], "param": "tau2NMDA", "op": "scale", "value": 0.5},
    # AMPA potentiation: faster decay and slightly larger conductance (if you model it via tau2 only)
    {"syn_mechs": ["AMPA"], "param": "tau2", "op": "scale", "value": 0.8},
    # GABA-A depolarizing shift (rare, example): Erev from -80 to -70
    {"syn_mechs": ["GABAA","GABAA_VIP"], "param": "e", "op": "set", "value": -70},
    # GABA-B slowing
    {"syn_mechs": ["GABAB"], "param": "tau2", "op": "scale", "value": 1.25},
]

cfg.cellmod =  {'IT2': 'HH_reduced',
				'IT4': 'HH_reduced',
				'IT5A': 'HH_full',
				'IT5B': 'HH_reduced',
				'PT5B': 'HH_full',
				'IT6': 'HH_reduced',
				'CT6': 'HH_reduced'}

ihQuiet = 1.0 # Factor for ih gbar in PT cells at quiet state
# if cfg.pt5b_variant == "tim": ihQuiet = 0.5
ihMovement = 0.25 # Factor for ih gbar in PT cells at movement state
cfg.ihModel = 'migliore'  # ih model
cfg.ihGbar = ihQuiet if cfg.SimulateBaseline else ihMovement # multiplicative factor for ih gbar in PT cells
cfg.ihGbarZD = None # multiplicative factor for ih gbar in PT cells
cfg.ihGbarBasal = 1.0 # 0.1 # multiplicative factor for ih gbar in PT cells
cfg.ihlkc = 0.2 # ih leak param (used in Migliore)
cfg.ihlkcBasal = 1.0
cfg.ihlkcBelowSoma = 0.01
cfg.ihlke = -86  # ih leak param (used in Migliore)
cfg.ihSlope = 14*2

cfg.somaNa = 5
cfg.axonNa = 7
cfg.axonRa = 0.005

cfg.gpas = 0.5  # multiplicative factor for pas g in PT cells
cfg.epas = 0.9  # multiplicative factor for pas e in PT cells

cfg.modifyMechs = {'startTime': cfg.preTone, 'endTime': cfg.duration, 
                   'cellType':'PT', 'mech': 'hd', 'property': 'gbar', 'newFactor': 1.00, 'origFactor': 0.75}

#------------------------------------------------------------------------------
# Synapses
#------------------------------------------------------------------------------
cfg.synWeightFractionEE = [0.5, 0.5] # E->E AMPA to NMDA ratio
cfg.synWeightFractionEI = [0.5, 0.5] # E->I AMPA to NMDA ratio
cfg.synWeightFractionSOME = [0.9, 0.1] # SOM -> E GABAASlow to GABAB ratio
cfg.synWeightFractionNGF = [0.5, 0.5] # NGF GABAA to GABAB ratio

cfg.synsperconn = {'HH_full': 5, 'HH_reduced': 1, 'HH_simple': 1}
cfg.AMPATau2Factor = 1.0

cfg.addSynMechs = True
cfg.distributeSynsUniformly = True

#------------------------------------------------------------------------------
# Network 
#------------------------------------------------------------------------------
cfg.layer = {'1':[0.0, 0.1], '2': [0.1,0.29], '4': [0.29,0.37], '5A': [0.37,0.47], '24':[0.1,0.37], '5B': [0.47,0.8], '6': [0.8,1.0], 
'longTPO': [2.0,2.1], 'longTVL': [2.1,2.2], 'longS1': [2.2,2.3], 'longS2': [2.3,2.4], 'longcM1': [2.4,2.5], 'longM2': [2.5,2.6], 'longOC': [2.6,2.7]}  # normalized layer boundaries

cfg.singleCellPops = False  # Create pops with 1 single cell (to debug)
cfg.weightNorm = 1  # use weight normalization
cfg.weightNormThreshold = 4.0  # weight normalization factor threshold

cfg.addConn = 1
cfg.allowConnsWithWeight0 = True
cfg.allowSelfConns = False
cfg.scale = 1.0
cfg.sizeY = 1350.0
cfg.sizeX = 300.0
cfg.sizeZ = 300.0
cfg.scaleDensity = 1.0
cfg.correctBorderThreshold = 150.0

cfg.L5BrecurrentFactor = 1.0
cfg.ITinterFactor = 1.0
cfg.strengthFactor = 1.0

cfg.EEGain = 1.0
cfg.EIGain = 1.0
cfg.IEGain = 1.0
cfg.IIGain = 1.0

## E->I by target cell type
cfg.EICellTypeGain= {'PV': 1.0, 'SOM': 1.0, 'VIP': 1.0, 'NGF': 1.0}

cfg.IEdisynapticBias = None  # increase prob of I->Ey conns if Ex->I and Ex->Ey exist 

#------------------------------------------------------------------------------
## (deprecated) E->I gains 
cfg.EPVGain = 1.0
cfg.ESOMGain = 1.0

#------------------------------------------------------------------------------
## (deprecated) I->E gains
cfg.PVEGain = 1.0
cfg.SOMEGain = 1.0

#------------------------------------------------------------------------------
## (deprecated) I->I gains
cfg.PVSOMGain = None #0.25
cfg.SOMPVGain = None #0.25
cfg.PVPVGain = None # 0.75
cfg.SOMSOMGain = None #0.75

#------------------------------------------------------------------------------
## I->E/I layer weights (L2/3+4, L5, L6)
cfg.IEweights = [1.0, 1.0, 1.0]
cfg.IIweights = [1.0, 1.0, 1.0]

cfg.IPTGain = 1.0
cfg.IFullGain = 1.0  # deprecated
#------------------------------------------------------------------------------
# Subcellular distribution
#------------------------------------------------------------------------------
cfg.addSubConn = True
#------------------------------------------------------------------------------
# Long range inputs
#------------------------------------------------------------------------------
cfg.addLongConn = True
cfg.numCellsLong = int(1000 * cfg.scaleDensity) # num of cells per population
cfg.noiseLong = 1.0  # firing rate random noise
cfg.delayLong = 5.0  # (ms)
factor = 1
cfg.weightLong = {'TPO': 0.5*factor, 'TVL': 0.5*factor, 'S1': 0.5*factor, 'S2': 0.5*factor, 'cM1': 0.5*factor, 'M2': 0.5*factor, 'OC': 0.5*factor}  # corresponds to unitary connection somatic EPSP (mV)
cfg.startLong = 0  # start at 0 ms
cfg.numSampledCellsPerLayer = None

LongRangeQuiet = [0, 2.5]
TVLquiet = [0, 2.5] 
TVLmovement = [0, 10]  # TVL firing rate (Hz)

TVLRates = TVLquiet if cfg.SimulateBaseline else TVLmovement

cfg.ratesLong = {'TPO': LongRangeQuiet, 'TVL': TVLRates, 'S1': LongRangeQuiet, 'S2': LongRangeQuiet, 'cM1': LongRangeQuiet, 'M2': LongRangeQuiet, 'OC': LongRangeQuiet}

## input pulses
cfg.addPulses = False
cfg.pulse = {'pop': 'None', 'start': 1000, 'end': 1100, 'rate': 20, 'noise': 0.8}
cfg.pulse2 = {'pop': 'None', 'start': 1000, 'end': 1200, 'rate': 20, 'noise': 0.5, 'duration': None}

#------------------------------------------------------------------------------
# Current inputs 
#------------------------------------------------------------------------------
cfg.addIClamp = 0

cfg.IClamp1 = {'pop': 'IT5B', 'sec': 'soma', 'loc': 0.5, 'start': 0, 'dur': 1000, 'amp': 0.50}

#------------------------------------------------------------------------------
# NetStim inputs 
#------------------------------------------------------------------------------
cfg.addNetStim = 0

 			   ## pop, sec, loc, synMech, start, interval, noise, number, weight, delay 
# cfg.NetStim1 = {'pop': 'IT2', 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA','NMDA'], 'synMechWeightFactor': cfg.synWeightFractionEE,
# 				'start': 500, 'interval': 50.0, 'noise': 0.2, 'number': 1000.0/50.0, 'weight': 10.0, 'delay': 1}
cfg.NetStim1 = {'pop': 'IT2', 'ynorm':[0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0],
				'start': 500, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 'weight': 30.0, 'delay': 0}

#------------------------------------------------------------------------------
# In Vivo m1 and thalamus sampled neurons & spikes
#------------------------------------------------------------------------------
if cfg.addInVivoThalamus:
    baselineSpks, movementAndPostSpks = defs.loadThalSpikes(cwd, cfg, skipEmpty=False)
    M1sampledCells, sessionNames = defs.m1SampledDepths(cwd, layer_order = ["1", "2", "4", "5A", "5B", "6"])

    trimmedBaseline = defs.trimTVLSpikes(baselineSpks, cfg)
    trimmedMovement = defs.trimTVLSpikes(movementAndPostSpks, cfg)
    
    cfg.numSampledCellsPerLayer = defs.average_dict_entries(M1sampledCells)
    cfg.spikeTimesInVivo = trimmedBaseline if cfg.SimulateBaseline else trimmedMovement

    del baselineSpks, movementAndPostSpks, trimmedBaseline, trimmedMovement
    gc.collect()
