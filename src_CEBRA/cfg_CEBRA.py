"""
cfg.py 

Simulation configuration for M1 model (using NetPyNE)

Contributors: salvadordura@gmail.com
"""

from netpyne import specs
import numpy as np
import pickle
from pathlib import Path
import defs
import gc

cwd = str(Path.cwd())
cfg = specs.SimConfig()  

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
###
# FIRST THING IS TO LOAD THE TRIAL NEURAL_DATA (WE KNOW THAT WE HAVE 1500 MS PRE-TONE AND 1500 MS POST-TONE)
########

cfg.selected_sessions = '230517_2759_1606VAL' # 230712_3745_1356VAL 230517_2759_1606VAL 230711_3747_933VAL 230713_3745_1053VAL
cfg.selected_trial = 20

from utils import MetaParams
pmp = MetaParams('neural_data', {
        'region': ('m1', 'region to process as neural data', False),
        'rec_type': ('spikes', 'type of data to process (spikes or lfp)', False),
        'smooth': (True, 'smooth spikes'),
        # To use with training mode 'thal':
        # 'use_thal_as': ('th-cebra', 'preprocessing method for thalamus as auxiliary data', False), # e.g. 'th-cebra' or 'th-pca3' (PCA to 3 dimensions)
})
preprocess_label = pmp.full_label()
prepr_data_fname = Path('src_CEBRA/CEBRA_data') / 'all_data.pkl'

from prepr import load_preprocessed_joint_data

if prepr_data_fname.exists():
    neural_data, auxiliary, session_labels = load_preprocessed_joint_data(prepr_data_fname)
    print(f"Loaded data from {prepr_data_fname}")
else:
	print("Error: Preprocessed data file not found. Please run the preprocessing script first.")

index_session = [i for i, s in enumerate(session_labels) if s == cfg.selected_sessions][0]

import json

# Opening and reading the JSON file
with open(f'src_CEBRA/CEBRA_params/{cfg.selected_sessions}_CEBRAparams.txt', 'r') as f:
    # Parsing the JSON file into a Python dictionary
    cebraParams = json.load(f)

with open(f'src_CEBRA/CEBRA_params/{cfg.selected_sessions}_trial_idx.txt', 'r') as f:
    # Parsing the JSON file into a Python dictionary
    trial_idx = json.load(f)



# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(cfg.neural_data.T, aspect='auto', cmap='viridis')
# plt.legend()
# plt.savefig("NeuralData.png")
# print(np.shape(cfg.neural_data)) 
# quit()

validCellsDepth = cebraParams['cell_depths']

cfg.CEBRA_params = cebraParams

cfg.transient = 500.0  # transient period to skip at the beginning of the simulation and experiment (ms)

maskTrial = np.array(trial_idx) == cfg.selected_trial

cfg.dt = 0.025 # For GPU increase the dt to not get precision errors
cfg.singleCellPops = False  # Create pops with 1 single cell (to debug)
cfg.recordStep = cfg.dt

transientBins = int(cfg.transient/cebraParams['dt']) # number of bins to skip as transient

cfg.stageId = auxiliary[index_session][maskTrial,0]
cfg.stageId = cfg.stageId[transientBins-1:]  # remove transient
cfg.neural_data = neural_data[index_session][maskTrial,:] # time x cells

if len(validCellsDepth) > np.shape(cfg.neural_data)[1]:
    validCellsDepth = np.sort(validCellsDepth)[:np.shape(cfg.neural_data)[1]]
    validCellsDepth.tolist()
elif len(validCellsDepth) < np.shape(cfg.neural_data)[1]:
      ValueError('Number of valid cells in depth file less than number of cells in neural data')

cfg.neural_data = cfg.neural_data[transientBins-1:,:]  # remove transient
# cfg.neural_data = cfg.neural_data.T # transpose

cfg.period = 'full_trial' # 'scaled_tone', 'scaled_prep', 'full_unlock' full_trial

ihQuiet = 1.0 # Factor for ih gbar in PT cells at quiet state
ihMovement = 0.25
cfg.ihGbar = ihQuiet # multiplicative factor for ih gbar in PT cells

if cfg.period == 'full_trial':
	# Raw Data is concatenation of both
	durationPre = 1500.
	durationPost = 1500.
	cfg.SimulateBaseline = False
	cfg.preTone = durationPre
	cfg.postTone = durationPost
else:
	raise ValueError('cfg.period not recognized')
	
cfg.addInVivoThalamus = True # To add the sampled spike times from in-vivo recordings on TVL
cfg.duration = cfg.preTone + cfg.postTone
cfg.seeds = {'conn': 4321, 'stim': 1234, 'loc': 4321, 'tvl_sampling': 1234, 'm1_sampling': 4321}  # seeds for randomizers (connectivity, input stimulation, cell locations)
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
cfg.timeRanges = [cfg.transient, cfg.duration]
cfg.printPopAvgRates = cfg.timeRanges

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
                    'V_dend_5': {'sec':'dend_5', 'loc':0.5, 'var':'v', 'conds':{'pop': 'PT5B'}}}

cfg.analysis['plotTraces'] = {'include': cfg.recordCells, 'timeRange': cfg.timeRanges, 
								'overlay': True, 'oneFigPer': 'trace', 'figSize': (10,4), 
								'saveFig': True, 'subtitles': True} 
#------------------------------------------------------------------------------
# Cells
#------------------------------------------------------------------------------
cfg.pt5b_variant = "standard"        # "tim" or "standard"

cfg.dendNa = 0.3 if cfg.pt5b_variant=="standard" else 1.0 # 0.3 for "standard", 1.0 for "tim"

cfg.loadmutantParams = False
cfg.variant = 'WT' # L1666F, E1211K, D195G, R853Q, K1422E, M1879T, WT

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

cfg.weightNorm = 1  # use weight normalization
cfg.weightNormThreshold = 4.0  # weight normalization factor threshold

cfg.addConn = 1
cfg.allowConnsWithWeight0 = True
cfg.allowSelfConns = False
cfg.scale = 1.0
cfg.sizeY = 1350.0
cfg.sizeX = 300.0
cfg.sizeZ = 300.0
cfg.scaleDensity = 1.0 # 1.0
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
cfg.addSubConn = 1
#------------------------------------------------------------------------------
# Long range inputs
#------------------------------------------------------------------------------
cfg.addLongConn = 1
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

cfg.ratesLong = {'TPO': [0, 5], 'TVL': TVLRates, 'S1': [0, 5], 'S2': [0, 5], 'cM1': LongRangeQuiet, 'M2': LongRangeQuiet, 'OC': [0,5]}

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
	baselineSpks, movementAndPostSpks, M1sampledCells, foldersName = defs.loadThalSpikes(cwd, cfg, skipEmpty=False)

	trimmedBaseline = defs.trimTVLSpikes(baselineSpks, cfg)
	trimmedMovement = defs.trimTVLSpikes(movementAndPostSpks, cfg)

	# cfg.numSampledCellsPerLayer is an average of sampled cells per layer across all trials
	cfg.numSampledCellsPerLayer = defs.cellPerlayer(validCellsDepth)
      
	cfg.spikeTimesInVivo = trimmedBaseline if cfg.SimulateBaseline else trimmedMovement
	del baselineSpks, movementAndPostSpks, trimmedBaseline, trimmedMovement
	gc.collect()