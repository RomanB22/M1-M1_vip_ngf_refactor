"""Declarative NetPyNE simulation configuration.

This file intentionally contains parameter definitions only. BatchTK mappings,
environment-based smoke-test overrides, derived values, data loading, and plot
configuration are applied by ``simulation.runtime_config`` before
``netParams.py`` constructs the network.
"""

from netpyne.batchtools.runners import Runner_SimConfig

from calibration.objective_config import get_objectives


cfg = Runner_SimConfig()

# BatchTK communication pointers
cfg._batchtk_label_pointer = None
cfg._batchtk_path_pointer = None
cfg._runtime_configured = False

# Simulation timing and execution
cfg.preTone = 1000
cfg.postTone = 1500
cfg.SimulateBaseline = False
cfg.duration = 3500
cfg.dt = 0.025
cfg.hParams = {"celsius": 34, "v_init": -80}
cfg.seeds = {
    "conn": 4321,
    "stim": 1234,
    "loc": 4321,
    "tvl_sampling": 1234,
    "m1_sampling": 4321,
    "cell": 1234,
}
cfg.verbose = False
cfg.createNEURONObj = True
cfg.createPyStruct = True
cfg.connRandomSecFromList = False
cfg.cvode_active = False
cfg.cvode_atol = 1e-6
cfg.cache_efficient = True
cfg.printRunTime = 0.1
cfg.printSynsAfterRule = False
cfg.pt3dRelativeToCellLocation = True
cfg.oneSynPerNetcon = True
cfg.validateNetParams = True
cfg.progressBar = 0
cfg.checkErrors = False
cfg.checkErrorsVerbose = False
cfg.rand123GlobalIndex = None
cfg.coreneuron = True
cfg.random123 = True
cfg.gpu = False

# Calibration and rate-reporting windows
cfg.objectives = get_objectives()
cfg.optimizationMode = True
cfg.includeParamsLabel = False
cfg.timeRanges = [1000, 3500]
cfg.printPopAvgRates = [
    [1000, 1250],
    [1250, 1500],
    [1500, 1750],
    [1750, 2000],
    [2000, 2250],
    [2250, 2500],
    [2500, 2750],
    [2750, 3000],
    [3000, 3250],
    [3250, 3500],
]

# Recording and output
cfg.cellsrec = 1
cfg.plotSimResults = True
cfg.showPlots = False
cfg.recordCells = []
cfg.recordStim = False
cfg.recordTime = False
cfg.recordStep = 0.025
cfg.recordTraces = {}
cfg.recordLFP = [
    [150, 200, 150],
    [150, 400, 150],
    [150, 600, 150],
    [150, 800, 150],
    [150, 1000, 150],
    [150, 1200, 150],
]
cfg.recordDipole = False
cfg.analysis = {}
cfg.simLabel = "v103_tune3"
cfg.saveFolder = "./batchData/v103_manualTune"
cfg.savePickle = False
cfg.saveJson = True
cfg.saveDataInclude = ["simData", "simConfig", "netParams"]
cfg.backupCfgFile = None
cfg.gatherOnlySimData = False
cfg.saveCellSecs = False
cfg.saveCellConns = 0
cfg.compactConnFormat = 0

# Cell models and intrinsic parameters
cfg.diversity = True
cfg.pasUniformFrac = 0.30
cfg.cao_secs = 1.2
cfg.pt5b_variant = "tim"
cfg.cellModelLoadMode = "saved"
cfg.cellModelLoadModeByLabel = {
    "IT2_reduced": "saved",
    "IT4_reduced": "saved",
    "IT5A_reduced": "saved",
    "IT5B_reduced": "saved",
    "PT5B_reduced": "saved",
    "IT6_reduced": "saved",
    "CT6_reduced": "saved",
    "SOM_reduced": "saved",
    "IT5A_full": "saved",
    "PV_reduced": "saved",
    "VIP_reduced": "saved",
    "NGF_reduced": "saved",
    "PT5B_full": "source",
}
cfg.dendNa = 1.0
cfg.cellmod = {
    "IT2": "HH_reduced",
    "IT4": "HH_reduced",
    "IT5A": "HH_full",
    "IT5B": "HH_reduced",
    "PT5B": "HH_full",
    "IT6": "HH_reduced",
    "CT6": "HH_reduced",
}
cfg.ihModel = "migliore"
cfg.ihGbar = 0.25
cfg.ihGbarZD = None
cfg.ihGbarBasal = 1.0
cfg.ihlkc = 0.2
cfg.ihlkcBasal = 1.0
cfg.ihlkcBelowSoma = 0.01
cfg.ihlke = -86
cfg.ihSlope = 28
cfg.somaNa = 5
cfg.axonNa = 7
cfg.axonRa = 0.005
cfg.gpas = 0.5
cfg.epas = 0.9
cfg.modifyMechs = {
    "startTime": 1000,
    "endTime": 3500,
    "cellType": "PT",
    "mech": "hd",
    "property": "gbar",
    "newFactor": 1.0,
    "origFactor": 0.75,
}

# Spike blockade detector
cfg.spikeGuard = {
    "enabled": True,
    "pointpName": "spike_guard",
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

# Mutation and drug-treatment switches
cfg.loadmutantParams = False
cfg.variant = "WT"
cfg.heterozygous = False
cfg.blockNa = False
cfg.KCNT1 = False
cfg.mutations_enabled = False
cfg.mutations_dry_run = False
cfg.mutations = []
cfg.drugTreatment = False
cfg.verbose_drug_changes = False
cfg.drug_dry_run = False
cfg.cell_drugs = [
    {"cell_types": ["PV", "SOM"], "mech": "nax", "param": "gbar", "op": "set", "value": 0.0, "sections": "ALL"},
    {"cell_types": ["PT", "IT"], "mech": "nax", "param": "gbar", "op": "set", "value": 0.0, "sections": "ALL"},
    {"cell_types": ["PT", "IT"], "mech": "na12", "param": "gbar", "op": "set", "value": 0.0, "sections": "ALL"},
    {"cell_types": ["PT", "IT"], "mech": "na12mut", "param": "gbar", "op": "set", "value": 0.0, "sections": "ALL"},
    {"cell_types": ["VIP"], "mech": "Ih", "param": "gIhbar", "op": "scale", "value": 0.2, "sections": "ALL"},
    {"cell_types": ["NGF"], "mech": "hd", "param": "gbar", "op": "scale", "value": 1.5, "sections": ["dend"]},
    {"cell_types": ["SOM"], "mech": "catcb", "param": "gcatbar", "op": "scale", "value": 0.5, "sections": ["soma"]},
]
cfg.syn_drugs = [
    {"syn_mechs": ["NMDA"], "param": "tau2NMDA", "op": "scale", "value": 0.5},
    {"syn_mechs": ["AMPA"], "param": "tau2", "op": "scale", "value": 0.8},
    {"syn_mechs": ["GABAA", "GABAA_VIP"], "param": "e", "op": "set", "value": -70},
    {"syn_mechs": ["GABAB"], "param": "tau2", "op": "scale", "value": 1.25},
]

# Synapses and connection gains
cfg.synWeightFractionEE = [0.5, 0.5]
cfg.synWeightFractionEI = [0.5, 0.5]
cfg.synWeightFractionSOME = [0.9, 0.1]
cfg.synWeightFractionNGF = [0.5, 0.5]
cfg.synsperconn = {"HH_full": 5, "HH_reduced": 1, "HH_simple": 1}
cfg.AMPATau2Factor = 1.0
cfg.addSynMechs = True
cfg.distributeSynsUniformly = True
cfg.addConn = 1
cfg.allowConnsWithWeight0 = True
cfg.allowSelfConns = False
cfg.weightNorm = 1
cfg.weightNormThreshold = 4.0
cfg.L5BrecurrentFactor = 1.0
cfg.ITinterFactor = 1.0
cfg.strengthFactor = 1.0
cfg.EEGain = 1.0
cfg.EIGain = 1.0
cfg.IEGain = 1.0
cfg.IIGain = 1.0
cfg.EICellTypeGain = {"PV": 1.0, "SOM": 1.0, "VIP": 1.0, "NGF": 1.0}
cfg.IEdisynapticBias = None
cfg.EPVGain = 1.0
cfg.ESOMGain = 1.0
cfg.PVEGain = 1.0
cfg.SOMEGain = 1.0
cfg.PVSOMGain = None
cfg.SOMPVGain = None
cfg.PVPVGain = None
cfg.SOMSOMGain = None
cfg.IEweights = [1.0, 1.0, 1.0]
cfg.IIweights = [1.0, 1.0, 1.0]
cfg.IPTGain = 1.0
cfg.IFullGain = 1.0
cfg.addSubConn = True

# Network geometry and layer boundaries
cfg.scale = 1.0
cfg.scaleDensity = 1.0
cfg.sizeX = 300.0
cfg.sizeY = 1350.0
cfg.sizeZ = 300.0
cfg.correctBorderThreshold = 150.0
cfg.layer = {
    "1": [0.0, 0.1],
    "2": [0.1, 0.29],
    "4": [0.29, 0.37],
    "5A": [0.37, 0.47],
    "24": [0.1, 0.37],
    "5B": [0.47, 0.8],
    "6": [0.8, 1.0],
    "longTPO": [2.0, 2.1],
    "longTVL": [2.1, 2.2],
    "longS1": [2.2, 2.3],
    "longS2": [2.3, 2.4],
    "longcM1": [2.4, 2.5],
    "longM2": [2.5, 2.6],
    "longOC": [2.6, 2.7],
}

# Cheap developer smoke-test defaults; environment overrides are applied later
cfg.singleCellPops = False
cfg.testCellsPerPop = 7

# Long-range and experimental inputs
cfg.addLongConn = True
cfg.numCellsLong = 1000
cfg.noiseLong = 1.0
cfg.delayLong = 5.0
cfg.weightLong = {"TPO": 0.5, "TVL": 0.5, "S1": 0.5, "S2": 0.5, "cM1": 0.5, "M2": 0.5, "OC": 0.5}
cfg.startLong = 0
cfg.ratesLong = {
    "TPO": [0, 2.5],
    "TVL": [0, 10],
    "S1": [0, 2.5],
    "S2": [0, 2.5],
    "cM1": [0, 2.5],
    "M2": [0, 2.5],
    "OC": [0, 2.5],
}
cfg.addInVivoThalamus = False
cfg.numSampledCellsPerLayer = None
cfg.spikeTimesInVivo = []

# Optional stimulation
cfg.addPulses = False
cfg.pulse = {"pop": "None", "start": 1000, "end": 1100, "rate": 20, "noise": 0.8}
cfg.pulse2 = {"pop": "None", "start": 1000, "end": 1200, "rate": 20, "noise": 0.5, "duration": None}
cfg.addIClamp = 0
cfg.IClamp1 = {"pop": "IT5B", "sec": "soma", "loc": 0.5, "start": 0, "dur": 1000, "amp": 0.50}
cfg.addNetStim = 0
cfg.NetStim1 = {
    "pop": "IT2",
    "ynorm": [0, 1],
    "sec": "soma",
    "loc": 0.5,
    "synMech": ["AMPA"],
    "synMechWeightFactor": [1.0],
    "start": 500,
    "interval": 1000.0 / 60.0,
    "noise": 0.0,
    "number": 60.0,
    "weight": 30.0,
    "delay": 0,
}
