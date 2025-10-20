"""
cfg.py 

Simulation configuration for M1 model (using NetPyNE)

Contributors: salvadordura@gmail.com
"""

from netpyne import specs
import pickle

cfg = specs.SimConfig()  


#------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
#------------------------------------------------------------------------------
	
cfg.simLabel = 'v53_tune3'

#------------------------------------------------------------------------------
# Run parameters
#------------------------------------------------------------------------------
cfg.duration = 1.5*1e3 
cfg.dt = 0.05
cfg.seeds = {'conn': 4321, 'stim': 1234, 'loc': 4321} 
cfg.hParams = {'celsius': 34, 'v_init': -80}  
cfg.verbose = 0
cfg.createNEURONObj = True
cfg.createPyStruct = True 
cfg.connRandomSecFromList = False  # set to false for reproducibility  
cfg.cvode_active = False
cfg.cvode_atol = 1e-6
cfg.cache_efficient = True
cfg.printRunTime = 0.1

cfg.includeParamsLabel = False
cfg.printPopAvgRates = True

cfg.checkErrors = 0


#------------------------------------------------------------------------------
# Recording 
#------------------------------------------------------------------------------
allpops = ['IT2','PV2','SOM2','IT4','IT5A','PV5A','SOM5A','IT5B','PT5B','PV5B','SOM5B','IT6','CT6','PV6','SOM6']
cfg.cellsrec = 4
if cfg.cellsrec == 0:  # record all cells
	cfg.recordCells = ['all']
elif cfg.cellsrec == 1:  # record one cell of each pop
	cfg.recordCells = 	[(pop,0) for pop in allpops]
elif cfg.cellsrec == 2:  # record selected cells
	cfg.recordCells = [('IT2',10), ('IT5A',10), ('PT5B',10), ('PV5B',10), ('SOM5B',10)]
elif cfg.cellsrec == 3:  # record selected cells
	cfg.recordCells = [('IT5A',0), ('PT5B',0)]
elif cfg.cellsrec == 4:  # record selected cells
	cfg.recordCells = [] #[('PT5B_1',0), ('PT5B_ZD',0)]

cfg.recordTraces = {'V_soma': {'sec':'soma', 'loc':0.5, 'var':'v'}}
					# 'i_gabab': {'sec':'soma', 'loc':0.5, 'synMech': 'GABAB', 'var':'i'},
					# 'i_gabaa': {'sec':'soma', 'loc':0.5, 'synMech': 'GABAASlow', 'var':'i'}}
					# 'V_Apic23': {'sec':'apic_25', 'loc':0.20901261732739618, 'var':'v'},
					# 'V_Apic30': {'sec':'apic_30', 'loc':0.43547676639865263, 'var':'v'},
					# 'V_Apic81': {'sec':'apic_81', 'loc':0.36259169916613432, 'var':'v'}}

#cfg.recordLFP = [[150, y, 150] for y in range(200,1300,100)]

cfg.recordStim = False  
cfg.recordTime = False
cfg.recordStep = 0.1


#------------------------------------------------------------------------------
# Saving
#------------------------------------------------------------------------------

cfg.saveFolder = '../data/v53_manualTune'
cfg.savePickle = False
cfg.saveJson = True
cfg.saveDataInclude = ['simData', 'simConfig', 'netParams']#, 'net']
cfg.backupCfgFile = None #['cfg.py', 'backupcfg/'] 
cfg.gatherOnlySimData = False
cfg.saveCellSecs = 1
cfg.saveCellConns = 1

#------------------------------------------------------------------------------
# Analysis and plotting 
#------------------------------------------------------------------------------
# cfg.analysis['plotRaster'] = {'include':['all'], 'timeRange':[0,1000], 'saveFig': True, 'showFig': False, 'labels': 'overlay', 'popRates': True, 'orderInverse': True, 'figSize': (12,10),'lw': 0.3, 'markerSize':3, 'marker': '.', 'dpi': 300} 
cfg.analysis['plotTraces'] = {'include': [], 'oneFigPer': 'trace', 'overlay': 1, 'figSize': (12,8), 'timeRange': [0,1000], 'saveFig': True, 'showFig': False,
  							 'colors': [[0,0,1],[0,0.502,0]]}#, 'ylim': [-90, 15]} 
# cfg.analysis['plotLFP'] = {'includeAxon': False, 'figSize': (6,10), 'NFFT': 256*20, 'noverlap': 128*20, 'nperseg': 132*20, 'saveFig': True, 'showFig':True} 

# with open('cells/popColors.pkl', 'r') as fileObj: popColors = pickle.load(fileObj)['popColors']

# cfg.analysis['plotRaster'] = {'include': ['allCells'], 'saveFig': True, 'showFig': False, 'labels': 'overlay', 'popRates': True, 'orderInverse': True, 
#   							'popColors': popColors, 'figSize': (12,10), 'lw': 0.6, 'marker': '|'} 

# cfg.analysis['plotConn'] = {'includePre': ['allCells'], 'includePost': ['IT5A_1'], 'feature': 'strength', 'figSize': (10,10), 'groupBy': 'pop', \
#  						'graphType': 'bar', 'synOrConn': 'conn', 'synMech': None, 'saveData': None, 'saveFig': 1, 'showFig': 0}

#cfg.analysis['plotEPSPAmp'] = {'include': [0,1], 'trace':'V_soma', 'start': 700, 'interval': 50, 'number': 5, 'amp': 'ratio', 'saveFig': True, 'showFig': False}

#------------------------------------------------------------------------------
# Cells
#------------------------------------------------------------------------------
cfg.cellmod =  {'IT2': 'HH_reduced',
				'IT4': 'HH_reduced',
				'IT5A': 'HH_full',
				'IT5B': 'HH_reduced',
				'PT5B': 'HH_full',
				'IT6': 'HH_reduced',
				'CT6': 'HH_reduced'}

cfg.ihModel = 'migliore'  # ih model
cfg.ihGbar = 1.0  # multiplicative factor for ih gbar in PT cells
cfg.ihGbarZD = 0.25 # multiplicative factor for ih gbar in PT cells
cfg.ihGbarBasal = 1.0 # 0.1 # multiplicative factor for ih gbar in PT cells
cfg.ihlkc = 0.2 # ih leak param (used in Migliore)
cfg.ihlkcBasal = 1.0
cfg.ihlkcBelowSoma = 0.01
cfg.ihlke = -86  # ih leak param (used in Migliore)
cfg.ihSlope = 14*2

cfg.removeNa = False  # simulate TTX; set gnabar=0s
cfg.somaNa = 5
cfg.dendNa = 0.3
cfg.axonNa = 7
cfg.axonRa = 0.005

cfg.gpas = 0.5  # multiplicative factor for pas g in PT cells
cfg.epas = 0.9  # multiplicative factor for pas e in PT cells

#------------------------------------------------------------------------------
# Synapses
#------------------------------------------------------------------------------
cfg.synWeightFractionEE = [0.5, 0.5] # E->E AMPA to NMDA ratio
cfg.synWeightFractionEI = [0.5, 0.5] # E->I AMPA to NMDA ratio
cfg.synWeightFractionSOME = [0.9, 0.1] # SOM -> E GABAASlow to GABAB ratio

cfg.synsperconn = {'HH_full': 5, 'HH_reduced': 1, 'HH_simple': 1}

cfg.excTau2Factor = 1.0

#------------------------------------------------------------------------------
# Network 
#------------------------------------------------------------------------------
cfg.singleCellPops = 1 # Create pops with 1 single cell (to debug)
cfg.weightNorm = 1  # use weight normalization
cfg.weightNormThreshold = 4.0  # weight normalization factor threshold

cfg.addConn = 0
cfg.scale = 1.0
cfg.sizeY = 1350.0
cfg.sizeX = 60.0
cfg.sizeZ = 60.0

cfg.EEGain = 1.0
cfg.EIGain = 1.0
cfg.IEGain = 1.0
cfg.IIGain = 1.0

cfg.IEdisynapticBias = None  # increase prob of I->Ey conns if Ex->I and Ex->Ey exist 

#------------------------------------------------------------------------------
## E->I gains
cfg.EPVGain = 1.0
cfg.ESOMGain = 1.0

#------------------------------------------------------------------------------
## I->E gains
cfg.PVEGain = 1.0
cfg.SOMEGain = 1.0

#------------------------------------------------------------------------------
## I->I gains
cfg.PVIGain = 1.0
cfg.SOMIGain = 1.0

#------------------------------------------------------------------------------
## I->E/I layer weights (L2/3+4, L5, L6)
cfg.IEweights = [1.0, 1.0, 1.0]
cfg.IIweights = [1.0, 1.0, 1.0]

cfg.IPTGain = 1.0

#------------------------------------------------------------------------------
# Subcellular distribution
#------------------------------------------------------------------------------
cfg.addSubConn = 0

cfg.ihSubConn = 0
cfg.ihX = 4
cfg.ihY = 14


#------------------------------------------------------------------------------
# Long range inputs
#------------------------------------------------------------------------------
cfg.addLongConn = 0
cfg.numCellsLong = 1000  # num of cells per population
cfg.noiseLong = 1.0  # firing rate random noise
cfg.delayLong = 5.0  # (ms)
cfg.weightLong = 0.5  # corresponds to unitary connection somatic EPSP (mV)
cfg.startLong = 0  # start at 0 ms
cfg.ratesLong = {'TPO': [0,2], 'TVL': [0,2], 'S1': [0,2], 'S2': [0,2], 'cM1': [0,2], 'M2': [0,2], 'OC': [0,2]}

#cfg.ratesLong = {'TPO': [0,2], 'TVL': [0,2], 'S1': [0,2], 'S2': 'cells/ssc-3_lowrate_spikes.json', 'cM1': [0,2], 'M2': [0,2], 'OC': [0,2]}
#cfg.ratesLong = {'TPO': [0,0.1], 'TVL': [0,0.1], 'S1': [0,0.1], 'S2': 'cells/ssc-3_lowrate_spikes.json', 'cM1': [0,0.1], 'M2': [0,0.1], 'OC': [0,0.1]}
#cfg.ratesLong = {'TPO': [0,0.1], 'TVL': [0,0.1], 'S1': 'cells/ssc-3_spikes.json', 'S2': [0,0.1], 'cM1': [0,0.1], 'M2': [0,0.1], 'OC': [0,0.1]}


## input pulses
cfg.addPulses = 0
cfg.pulse = {'pop': 'None', 'start': 2000, 'end': 2200, 'rate': 30, 'noise': 0.5}


#------------------------------------------------------------------------------
# Current inputs 
#------------------------------------------------------------------------------
cfg.addIClamp = 0

			  ## pop,   sec,   loc, start,dur, amp (nA)
cfg.IClamp1 = {'pop': 'CT6', 'sec': 'soma', 'loc': 0.5, 'start': 100, 'dur': 400, 'amp': 0.4}
# cfg.IClamp1 = {'pop': 'PT5B_fI30', 'sec': 'soma', 'loc': 0.5, 'start': 500, 'dur': 1000, 'amp': 0.30}
# cfg.IClamp2 = {'pop': 'PT5B_fI60', 'sec': 'soma', 'loc': 0.5, 'start': 500, 'dur': 1000, 'amp': 0.6}

# cfg.IClamp3 = {'pop': 'PT5B_fI10', 'sec': 'soma', 'loc': 0.5, 'start': 500, 'dur': 1000, 'amp': 0.10}
# cfg.IClamp4 = {'pop': 'PT5B_fI40', 'sec': 'soma', 'loc': 0.5, 'start': 500, 'dur': 1000, 'amp': 0.40}


#------------------------------------------------------------------------------
# NetStim inputs 
#------------------------------------------------------------------------------
cfg.addNetStim = 1

cfg.stimSubConn = 0

# pops = ['IT2','IT4','IT5A','IT5B','IT6','CT6']
# sec = 'soma'
# for pop in pops:
# 	tmp = {'pop': pop, 'ynorm':[0.0, 1.0], 'sec': sec, 'loc': 0.5, 'synMech': ['AMPA','NMDA'], 'synMechWeightFactor': [0.5,0.5], 'start': 300, \
# 			'interval': 1000, 'noise': 0.0, 'number': 1, 'weight': 0.5, 'delay': 1}

#  	setattr(cfg, 'NetStim_'+pop+'_'+sec, tmp) 

cfg.NetStim1 = {'pop': 'PT5B', 'ynorm':[0.0, 1.0], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA','NMDA'], 'synMechWeightFactor': [0.5,0.5], 'start': 300, \
			'interval': 1000, 'noise': 0.0, 'number': 1, 'weight': 0.5, 'delay': 1}

# cfg.NetStim1 = {'pop': 'IT2', 'ynorm':[0.15, 0.18], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0],
# 				'start': 300, 'interval': 1000.0/32.0, 'noise': 0.01, 'number': 32, 'weight': 30.0, 'delay': 0}

# cfg.NetStim2 = {'pop': 'IT5A', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA','NMDA'], 'synMechWeightFactor': [0.5,0.5],
# 				'start': 300, 'interval': 1, 'noise': 0.0, 'number': 1, 'weight': 0.5, 'delay': 1}

# cfg.GroupNetStimEPT = {'nstype': 'pop', 'ynorm': [0,1], 'numStims': 1000, 'pop': 'IT2', 'cellRule': 'IT2_reduced', 'secList': 'alldend', 'allSegs': True, \
#  						'synMech': ['AMPA','NMDA'], 'start': 300, 'interval': 1000/4.0, 'noise': 0.0, 'number': 4, 'weight': 0.1, 'delay': 0}



# IT2 freq stim
# cfg.NetStim1 = {'pop': 'IT2', 'ynorm': [0,0.15], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0],
# 				'start': 500, 'interval': 1000.0/60.0, 'noise': 0.2, 'number': 60.0, 'weight': 30.0, 'delay': 1}


# #  			   # pop, sec, loc, synMech, start, interval, noise, number, weight, delay 
# cfg.NetStim1 = {'pop': 'IT5A', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA','NMDA'], 'synMechWeightFactor': [0.5,0.5], 'start': 700, \
# 				'interval': 1000, 'noise': 0.0, 'number': 1, 'weight': 0.5, 'delay': 1}

# cfg.NetStim1 = {'pop': 'PT5B', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['GABAASlow','GABAB'], 'synMechWeightFactor': [0.9,0.1], 'start': 700, \
# 				'interval': 1000, 'noise': 0.0, 'number': 1, 'weight': 0.5, 'delay': 1}

cfg.groupWeight = 0.05
cfg.groupRate = 20

# sheets fig 8
# cfg.GroupNetStimEPT = {'nstype': 'pop', 'ynorm': [0,1], 'numStims': 100, 'pop': ['PT5B','PT5B_ZD'], 'ynorm': [0,1], 'cellRule': 'PT5B_full', 'secList': 'alldend', 'allSegs': True, \
#  						'synMech': ['exc'], 'start': 300, 'interval': 1000/cfg.groupRate, 'noise': 0.5, 'number': 3, 'weight': 0.20, 'delay': 0}

# # sheets fig 9
# cfg.GroupNetStimEPT1 = {'nstype': 'pop', 'numStims': 100, 'pop': ['PT5B','PT5B_ZD'], 'ynorm': [0,1], 'cellRule': 'PT5B_full', 'secList': 'alldend', 'allSegs': True, \
#  						'synMech': ['exc'], 'start': 300, 'interval': 1000/cfg.groupRate, 'noise': 0.1, 'number': 1, 'weight': 0.0, 'delay': 0}

# cfg.GroupNetStimEPT2 = {'nstype': 'pop', 'numStims': 100, 'pop': ['PT5B','PT5B_ZD'], 'ynorm': [0,1], 'cellRule': 'PT5B_full', 'secList': 'alldend', 'allSegs': True, \
#  						'synMech': ['exc'], 'start': 300, 'interval': 1000/cfg.groupRate, 'noise': 0.1, 'number': 1, 'weight': 0.0, 'delay': 25}

# george 2009 fig 2; migliore 2012
# cfg.GroupNetStimW1 = {'nstype': 'pop', 'numStims': 75, 'pop': 'PT5B', 'ynorm': [0,1], 'cellRule': 'PT5B_full', 'secList': 'alldend', 'allSegs': True, \
# 						'synMech': ['AMPA'], 'start': 300, 'interval': 1000/cfg.groupRate, 'noise': 0.6, 'number': 1, 'weight': 0.1 , 'delay': 1}
# cfg.GroupNetStimW2 = {'nstype': 'pop', 'numStims': 75, 'pop': 'PT5B_ZD', 'cellRule': 'PT5B_full2', 'secList': 'alldend', 'allSegs': True, \
# 						'synMech': ['AMPA'], 'start': 300, 'interval': 1000/cfg.groupRate, 'noise': 0.6, 'number': 1, 'weight': 0.1 , 'delay': 1}


# sheets fig 4
#cfg.GroupNetStimEPT = {'nstype': 'pop', 'numStims': 20, 'pop': 'PT5B', 'ynorm': [0,1], 'cellRule': 'PT5B_full', 'secList': ['apic_0'], 'allSegs': True, \
# 						'synMech': ['exc'], 'start': 700, 'interval': 1000/cfg.groupRate, 'noise': 0.0, 'number': 1, 'weight': 0.0, 'delay': 1}

# sheets fig 11  # num inputs = 8*16*100*0.07 = 896 (grid size * num neurons / grid * prob of exciting)
# cfg.GroupNetStimEPT = {'nstype': 'pop', 'numStims': int(8*16*100*0.07*0.16), 'pop': 'PT5B', 'ynorm': [0,1], 'cellRule': 'PT5B_full', 'secList': 'alldend', 'allSegs': True, \
#  						'synMech': ['AMPA','NMDA'], 'synMechWeightFactor': [0.5,0.5], 'start': 500, 'interval': 1000/20.0, 'noise': 1.0, 'number': cfg.groupRate*4.0, 'weight': 0.1, 'delay': 0}

# cfg.GroupNetStimEPT2 = {'nstype': 'pop', 'numStims': int(8*16*100*0.07*0.16), 'pop': 'PT5B_ZD', 'ynorm': [0,1], 'cellRule': 'PT5B_full', 'secList': 'alldend', 'allSegs': True, \
# 						'synMech': ['AMPA','NMDA'], 'synMechWeightFactor': [0.5,0.5], 'start': 500, 'interval': 1000/20.0, 'noise': 1.0, 'number': cfg.groupRate*4.0, 'weight': 0.1, 'delay': 0}


#------------------------------------------------------------------------------
# Network clamp
#------------------------------------------------------------------------------
cfg.netClamp = 0

cfg.netClampConnsFile = '../data/v47_batch9/v47_batch9_0.json'
cfg.netClampSpikesFile = '../data/v47_batch5/v47_batch5_1_1_2_0_1_1_1_0.json'
cfg.netClampPop = ['PT5B_1']
cfg.netClampGid = 5198
cfg.netClampCreateAllConns = 1



