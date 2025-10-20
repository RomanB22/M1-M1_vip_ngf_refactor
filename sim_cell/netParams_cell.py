
"""
netParams.py 

High-level specifications for M1 network model using NetPyNE

Contributors: salvadordura@gmail.com
"""

from netpyne import specs
import pickle, json

netParams = specs.NetParams()   # object of class NetParams to store the network parameters

netParams.version = 55

try:
    from __main__ import cfg  # import SimConfig object with params from parent module
except:
    from cfg_cell import cfg

#------------------------------------------------------------------------------
#
# NETWORK PARAMETERS
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# General network parameters
#------------------------------------------------------------------------------
netParams.scale = cfg.scale # Scale factor for number of cells
netParams.sizeX = cfg.sizeX # x-dimension (horizontal length) size in um
netParams.sizeY = cfg.sizeY # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = cfg.sizeZ # z-dimension (horizontal depth) size in um
netParams.shape = 'cylinder' # cylindrical (column-like) volume

#------------------------------------------------------------------------------
# General connectivity parameters
#------------------------------------------------------------------------------
netParams.scaleConnWeight = 1.0 # Connection weight scale factor (default if no model specified)
netParams.scaleConnWeightModels = {'HH_simple': 1.0, 'HH_reduced': 1.0, 'HH_full': 1.0} #scale conn weight factor for each cell model
netParams.scaleConnWeightNetStims = 1.0 #0.5  # scale conn weight factor for NetStims
netParams.defaultThreshold = 0.0 # spike threshold, 10 mV is NetCon default, lower it for all cells
netParams.defaultDelay = 2.0 # default conn delay (ms)
netParams.propVelocity = 500.0 # propagation velocity (um/ms)
netParams.probLambda = 100.0  # length constant (lambda) for connection probability decay (um)
netParams.defineCellShapes = True  # convert stylized geoms to 3d points

#------------------------------------------------------------------------------
# Cell parameters
#------------------------------------------------------------------------------
cellModels = ['HH_simple', 'HH_reduced', 'HH_full']
excTypes = ['IT', 'CT', 'PT']
inhTypes = ['PV', 'SOM', 'VIP', 'NGF']

layer = {'1':[0.0, 0.1], '2': [0.1,0.29], '4': [0.29,0.37], '5A': [0.37,0.47], '24':[0.1,0.37], '5B': [0.47,0.8], '6': [0.8,1.0], 
'longTPO': [2.0,2.1], 'longTVL': [2.1,2.2], 'longS1': [2.2,2.3], 'longS2': [2.3,2.4], 'longcM1': [2.4,2.5], 'longM2': [2.5,2.6], 'longOC': [2.6,2.7]}  # normalized layer boundaries

#netParams.correctBorder = {'threshold': [cfg.correctBorderThreshold, cfg.correctBorderThreshold, cfg.correctBorderThreshold], 
#                        'yborders': [layer['2'][0], layer['5A'][0], layer['6'][0], layer['6'][1]]}  # correct conn border effect

#------------------------------------------------------------------------------
## Load cell rules previously saved using netpyne format
cellParamLabels = ['IT2_reduced', 'IT4_reduced', 'IT5A_reduced', 'IT5B_reduced', 'PT5B_reduced',
    'IT6_reduced', 'CT6_reduced', 'PV_simple', 'SOM_simple', 'IT5A_full']# 'VIP_reduced', 'NGF_simple','PT5B_full'] #  # list of cell rules to load from file
loadCellParams = cellParamLabels
saveCellParams = False #True

for ruleLabel in loadCellParams:
    netParams.loadCellParamsRule(label=ruleLabel, fileName='cells/'+ruleLabel+'_cellParams.pkl')

#------------------------------------------------------------------------------
# Specification of cell rules not previously loaded
# Includes importing from hoc template or python class, and setting additional params

#------------------------------------------------------------------------------
# Reduced cell model params (6-comp) 
reducedCells = {  # layer and cell type for reduced cell models
    'IT2_reduced':  {'layer': '2',  'cname': 'CSTR6', 'carg': 'BS1578'}, 
    'IT4_reduced':  {'layer': '4',  'cname': 'CSTR6', 'carg': 'BS1578'},
    'IT5A_reduced': {'layer': '5A', 'cname': 'CSTR6', 'carg': 'BS1579'},
    'IT5B_reduced': {'layer': '5B', 'cname': 'CSTR6', 'carg': 'BS1579'},
    'PT5B_reduced': {'layer': '5B', 'cname': 'SPI6',  'carg':  None},
    'IT6_reduced':  {'layer': '6',  'cname': 'CSTR6', 'carg': 'BS1579'},
    'CT6_reduced':  {'layer': '6',  'cname': 'CSTR6', 'carg': 'BS1578'}}

reducedSecList = {  # section Lists for reduced cell model
    'alldend':  ['Adend1', 'Adend2', 'Adend3', 'Bdend'],
    'spiny':    ['Adend1', 'Adend2', 'Adend3', 'Bdend'],
    'apicdend': ['Adend1', 'Adend2', 'Adend3'],
    'perisom':  ['soma']}

for label, p in reducedCells.items():  # create cell rules that were not loaded 
    if label not in loadCellParams:
        cellRule = netParams.importCellParams(label=label, conds={'cellType': label[0:2], 'cellModel': 'HH_reduced', 'ynorm': layer[p['layer']]},
        fileName='cells/'+p['cname']+'.py', cellName=p['cname'], cellArgs={'params': p['carg']} if p['carg'] else None)
        dendL = (layer[p['layer']][0]+(layer[p['layer']][1]-layer[p['layer']][0])/2.0) * cfg.sizeY  # adapt dend L based on layer
        for secName in ['Adend1', 'Adend2', 'Adend3', 'Bdend']: cellRule['secs'][secName]['geom']['L'] = dendL / 3.0  # update dend L
        for k,v in reducedSecList.items(): cellRule['secLists'][k] = v  # add secLists
        #netParams.addCellParamsWeightNorm(label, 'conn/'+label+'_weightNorm.pkl', threshold=cfg.weightNormThreshold)  # add weightNorm

        # set 3d points
        offset, prevL = 0, 0
        somaL = netParams.cellParams[label]['secs']['soma']['geom']['L']
        for secName, sec in netParams.cellParams[label]['secs'].items():
            sec['geom']['pt3d'] = []
            if secName in ['soma', 'Adend1', 'Adend2', 'Adend3']:  # set 3d geom of soma and Adends
                sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
                prevL = float(prevL + sec['geom']['L'])
                sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
            if secName in ['Bdend']:  # set 3d geom of Bdend
                sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
                sec['geom']['pt3d'].append([offset+sec['geom']['L'], somaL, 0, sec['geom']['diam']])        
            if secName in ['axon']:  # set 3d geom of axon
                sec['geom']['pt3d'].append([offset+0, 0, 0, sec['geom']['diam']])
                sec['geom']['pt3d'].append([offset+0, -sec['geom']['L'], 0, sec['geom']['diam']])   

        if saveCellParams: netParams.saveCellParamsRule(label=label, fileName='cells/'+label+'_cellParams.pkl')


#------------------------------------------------------------------------------
## PT5B full cell model params (700+ comps)
if 'PT5B_full' not in loadCellParams:
    ihMod2str = {'harnett': 1, 'kole': 2, 'migliore': 3}
    cellRule = netParams.importCellParams(label='PT5B_full', conds={'cellType': 'PT', 'cellModel': 'HH_full'},
      fileName='cells/PTcell.hoc', cellName='PTcell', cellArgs=[ihMod2str[cfg.ihModel], cfg.ihSlope], somaAtOrigin=True)
    nonSpiny = ['apic_0', 'apic_1']
    netParams.addCellParamsSecList(label='PT5B_full', secListName='perisom', somaDist=[0, 50])  # sections within 50 um of soma
    netParams.addCellParamsSecList(label='PT5B_full', secListName='below_soma', somaDistY=[-600, 0])  # sections within 0-300 um of soma
    for sec in nonSpiny: cellRule['secLists']['perisom'].remove(sec)
    cellRule['secLists']['alldend'] = [sec for sec in cellRule.secs if ('dend' in sec or 'apic' in sec)] # basal+apical
    cellRule['secLists']['apicdend'] = [sec for sec in cellRule.secs if ('apic' in sec)] # apical
    cellRule['secLists']['spiny'] = [sec for sec in cellRule['secLists']['alldend'] if sec not in nonSpiny]
    # Adapt ih params based on cfg param
    for secName in cellRule['secs']:
        for mechName,mech in cellRule['secs'][secName]['mechs'].items():
            if mechName in ['ih','h','h15', 'hd']: 
                mech['gbar'] = [g*cfg.ihGbar for g in mech['gbar']] if isinstance(mech['gbar'],list) else mech['gbar']*cfg.ihGbar
                if cfg.ihModel == 'migliore':   
                    mech['clk'] = cfg.ihlkc  # migliore's shunt current factor
                    mech['elk'] = cfg.ihlke  # migliore's shunt current reversal potential
                if secName.startswith('dend'): 
                    mech['gbar'] *= cfg.ihGbarBasal  # modify ih conductance in soma+basal dendrites
                    mech['clk'] *= cfg.ihlkcBasal  # modify ih conductance in soma+basal dendrites
                if secName in cellRule['secLists']['below_soma']: #secName.startswith('dend'): 
                    mech['clk'] *= cfg.ihlkcBelowSoma  # modify ih conductance in soma+basal dendrites
    # Reduce dend Na to avoid dend spikes (compensate properties by modifying axon params)
    for secName in cellRule['secLists']['alldend']:
        cellRule['secs'][secName]['mechs']['nax']['gbar'] = 0.0153130368342 * cfg.dendNa # 0.25 
    cellRule['secs']['soma']['mechs']['nax']['gbar'] = 0.0153130368342  * cfg.somaNa
    cellRule['secs']['axon']['mechs']['nax']['gbar'] = 0.0153130368342  * cfg.axonNa # 11  
    cellRule['secs']['axon']['geom']['Ra'] = 137.494564931 * cfg.axonRa # 0.005
    # Remove Na (TTX)
    if cfg.removeNa:
        for secName in cellRule['secs']: cellRule['secs'][secName]['mechs']['nax']['gbar'] = 0.0
    #netParams.addCellParamsWeightNorm('PT5B_full', 'conn/PT5B_full_weightNorm.pkl', threshold=cfg.weightNormThreshold)  # load weight norm
    if saveCellParams: netParams.saveCellParamsRule(label='PT5B_full', fileName='cells/PT5B_full_cellParams.pkl')


#------------------------------------------------------------------------------
## IT5A full cell model params (700+ comps)
if 'IT5A_full' not in loadCellParams:
    cellRule = netParams.importCellParams(label='IT5A_full', conds={'cellType': 'IT', 'cellModel': 'HH_full', 'ynorm': layer['5A']},
      fileName='cells/ITcell.py', cellName='ITcell', cellArgs={'params': 'BS1579'}, somaAtOrigin=True)
    netParams.renameCellParamsSec(label='IT5A_full', oldSec='soma_0', newSec='soma')
    #netParams.addCellParamsWeightNorm('IT5A_full', 'conn/IT_full_BS1579_weightNorm.pkl', threshold=cfg.weightNormThreshold) # add weightNorm before renaming soma_0
    netParams.addCellParamsSecList(label='IT5A_full', secListName='perisom', somaDist=[0, 50])  # sections within 50 um of soma
    cellRule['secLists']['alldend'] = [sec for sec in cellRule.secs if ('dend' in sec or 'apic' in sec)] # basal+apical
    cellRule['secLists']['apicdend'] = [sec for sec in cellRule.secs if ('apic' in sec)] # basal+apical
    cellRule['secLists']['spiny'] = [sec for sec in cellRule['secLists']['alldend'] if sec not in ['apic_0', 'apic_1']]
    if saveCellParams: netParams.saveCellParamsRule(label='IT5A_full', fileName='cells/IT5A_full_cellParams.pkl')


#------------------------------------------------------------------------------
## IT5B full cell model params (700+ comps) - not used
# if 'IT5B_full' not in loadCellParams:
#   cellRule = netParams.importCellParams(label='IT5B_full', conds={'cellType': 'IT', 'cellModel': 'HH_full', 'ynorm': layer['5B']},
#     fileName='cells/ITcell.py', cellName='ITcell', cellArgs={'params': 'BS1579'}, somaAtOrigin=True)
#   netParams.addCellParamsSecList(label='IT5B_full', secListName='perisom', somaDist=[0, 50])  # sections within 50 um of soma
#   cellRule['secLists']['alldend'] = [sec for sec in cellRule.secs if ('dend' in sec or 'apic' in sec)] # basal+apical
#   cellRule['secLists']['apicdend'] = [sec for sec in cellRule.secs if ('apic' in sec)] # basal+apical
#   cellRule['secLists']['spiny'] = [sec for sec in cellRule['secLists']['alldend'] if sec not in ['apic_0', 'apic_1']]
#   netParams.addCellParamsWeightNorm('IT5B_full', 'conn/IT_full_BS1579_weightNorm.pkl')
#   netParams.saveCellParamsRule(label='IT5B_full', fileName='cells/IT5B_full_cellParams.pkl')


#------------------------------------------------------------------------------
## PV cell params (3-comp)
if 'PV_simple' not in loadCellParams:
    cellRule = netParams.importCellParams(label='PV_simple', conds={'cellType':'PV', 'cellModel':'HH_simple'}, 
      fileName='cells/FS3.hoc', cellName='FScell1', cellInstance = True)
    cellRule['secLists']['spiny'] = ['soma', 'dend']
    #netParams.addCellParamsWeightNorm('PV_simple', 'conn/PV_simple_weightNorm.pkl', threshold=cfg.weightNormThreshold)
    if saveCellParams: netParams.saveCellParamsRule(label='PV_simple', fileName='cells/PV_simple_cellParams.pkl')


#------------------------------------------------------------------------------
## SOM cell params (3-comp)
if 'SOM_simple' not in loadCellParams:
    cellRule = netParams.importCellParams(label='SOM_simple', conds={'cellType':'SOM', 'cellModel':'HH_simple'}, 
      fileName='cells/LTS3.hoc', cellName='LTScell1', cellInstance = True)
    cellRule['secLists']['spiny'] = ['soma', 'dend']
    #netParams.addCellParamsWeightNorm('SOM_simple', 'conn/SOM_simple_weightNorm.pkl', threshold=cfg.weightNormThreshold)
    if saveCellParams: netParams.saveCellParamsRule(label='SOM_simple', fileName='cells/SOM_simple_cellParams.pkl')
    

#------------------------------------------------------------------------------
## VIP cell params (5-comp)
if 'VIP_reduced' not in loadCellParams:
    cellRule = netParams.importCellParams(label='VIP_reduced', conds={'cellType': 'VIP', 'cellModel': 'HH_reduced'}, fileName='cells/vipcr_cell.hoc',         cellName='VIPCRCell_EDITED', importSynMechs = True)
    cellRule['secLists']['spiny'] = ['soma', 'rad1', 'rad2', 'ori1', 'ori2']
    #netParams.addCellParamsWeightNorm('VIP_simple', 'conn/SOM_simple_weightNorm.pkl', threshold=cfg.weightNormThreshold)
    if saveCellParams: netParams.saveCellParamsRule(label='VIP_simple', fileName='cells/VIP_reduced_cellParams.pkl')


#------------------------------------------------------------------------------
## NGF cell params (1-comp)
if 'NGF_simple' not in loadCellParams:
    netParams.importCellParams(label='NGF_simple', conds={'cellType': 'NGF', 'cellModel': 'HH_simple'}, fileName='cells/ngf_cell.hoc', cellName='ngfcell', importSynMechs = True)
    #cellRule['secLists']['spiny'] = ['soma', 'rad1', 'rad2', 'ori1', 'ori2']
    #netParams.addCellParamsWeightNorm('VIP_simple', 'conn/SOM_simple_weightNorm.pkl', threshold=cfg.weightNormThreshold)
    if saveCellParams: netParams.saveCellParamsRule(label='NGF_simple', fileName='cells/NGF_simple_cellParams.pkl')

#------------------------------------------------------------------------------
# Population parameters
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
## load densities
with open('cells/cellDensity.pkl', 'rb') as fileObj: density = pickle.load(fileObj)['density']

## Local populations
### Layer 1:
netParams.popParams['NGF1']  =   {'cellModel': 'HH_simple',         'cellType': 'NGF', 'ynormRange': layer['1'], 'density': density[('M1','nonVIP')][0]}

### Layer 2/3:
netParams.popParams['IT2']  =   {'cellModel': cfg.cellmod['IT2'],  'cellType': 'IT', 'ynormRange': layer['2'], 'density': density[('M1','E')][1]}
netParams.popParams['SOM2'] =   {'cellModel': 'HH_simple',         'cellType': 'SOM','ynormRange': layer['2'], 'density': density[('M1','SOM')][1]}
netParams.popParams['PV2']  =   {'cellModel': 'HH_simple',         'cellType': 'PV', 'ynormRange': layer['2'], 'density': density[('M1','PV')][1]}
netParams.popParams['VIP2']  =  {'cellModel': 'HH_reduced',        'cellType': 'VIP', 'ynormRange': layer['2'], 'density': density[('M1','VIP')][1]}
netParams.popParams['NGF2']  =  {'cellModel': 'HH_simple',         'cellType': 'NGF', 'ynormRange': layer['2'], 'density': density[('M1','nonVIP')][1]}

### Layer 4:
netParams.popParams['IT4']  =   {'cellModel': cfg.cellmod['IT4'],  'cellType': 'IT', 'ynormRange': layer['4'], 'density': density[('M1','E')][2]}
netParams.popParams['SOM4'] =   {'cellModel': 'HH_simple',         'cellType': 'SOM','ynormRange': layer['4'], 'density': density[('M1','SOM')][2]}
netParams.popParams['PV4']  =   {'cellModel': 'HH_simple',         'cellType': 'PV', 'ynormRange': layer['4'], 'density': density[('M1','PV')][2]}
netParams.popParams['VIP4']  =  {'cellModel': 'HH_reduced',        'cellType': 'VIP', 'ynormRange': layer['4'], 'density': density[('M1','VIP')][2]}
netParams.popParams['NGF4']  =  {'cellModel': 'HH_simple',         'cellType': 'NGF', 'ynormRange': layer['4'], 'density': density[('M1','nonVIP')][2]}

### Layer 5A:
netParams.popParams['IT5A'] =   {'cellModel': cfg.cellmod['IT5A'], 'cellType': 'IT', 'ynormRange': layer['5A'], 'density': density[('M1','E')][3]}
netParams.popParams['SOM5A'] =  {'cellModel': 'HH_simple',         'cellType': 'SOM','ynormRange': layer['5A'], 'density': density[('M1','SOM')][3]}
netParams.popParams['PV5A']  =  {'cellModel': 'HH_simple',         'cellType': 'PV', 'ynormRange': layer['5A'], 'density': density[('M1','PV')][3]}
netParams.popParams['VIP5A']  = {'cellModel': 'HH_simple',         'cellType': 'VIP', 'ynormRange': layer['5A'], 'density': density[('M1','VIP')][3]}
netParams.popParams['NGF5A']  = {'cellModel': 'HH_simple',         'cellType': 'NGF', 'ynormRange': layer['5A'], 'density': density[('M1','nonVIP')][3]}

### Layer 5B:
netParams.popParams['IT5B'] =   {'cellModel': cfg.cellmod['IT5B'], 'cellType': 'IT', 'ynormRange': layer['5B'], 'density': 0.5*density[('M1','E')][4]}
netParams.popParams['PT5B'] =   {'cellModel': cfg.cellmod['PT5B'], 'cellType': 'PT', 'ynormRange': layer['5B'], 'density': 0.5*density[('M1','E')][4]}
netParams.popParams['SOM5B'] =  {'cellModel': 'HH_simple',         'cellType': 'SOM','ynormRange': layer['5B'], 'density': density[('M1','SOM')][4]}
netParams.popParams['PV5B']  =  {'cellModel': 'HH_simple',         'cellType': 'PV', 'ynormRange': layer['5B'], 'density': density[('M1','PV')][4]}
netParams.popParams['VIP5B']  = {'cellModel': 'HH_reduced',        'cellType': 'VIP', 'ynormRange': layer['5B'], 'density': density[('M1','VIP')][4]}
netParams.popParams['NGF5B']  = {'cellModel': 'HH_simple',         'cellType': 'NGF', 'ynormRange': layer['5B'], 'density': density[('M1','nonVIP')][4]}

### Layer 6:
netParams.popParams['IT6']  =   {'cellModel': cfg.cellmod['IT6'],  'cellType': 'IT', 'ynormRange': layer['6'],  'density': 0.5*density[('M1','E')][5]}
netParams.popParams['CT6']  =   {'cellModel': cfg.cellmod['CT6'],  'cellType': 'CT', 'ynormRange': layer['6'],  'density': 0.5*density[('M1','E')][5]}
netParams.popParams['SOM6'] =   {'cellModel': 'HH_simple',         'cellType': 'SOM','ynormRange': layer['6'],  'density': density[('M1','SOM')][5]}
netParams.popParams['PV6']  =   {'cellModel': 'HH_simple',         'cellType': 'PV', 'ynormRange': layer['6'],  'density': density[('M1','PV')][5]}
netParams.popParams['VIP6']  =  {'cellModel': 'HH_reduced',        'cellType': 'VIP', 'ynormRange': layer['6'], 'density': density[('M1','VIP')][1]}
netParams.popParams['NGF6']  =  {'cellModel': 'HH_simple',         'cellType': 'NGF', 'ynormRange': layer['6'], 'density': density[('M1','nonVIP')][1]}

if cfg.singleCellPops:
    for popName,pop in netParams.popParams.items():
        if cfg.NetStim1['pop'] == popName:
            pop['numCells'] = 1
        else:
            pop['numCells'] = 0

#------------------------------------------------------------------------------
## Long-range input populations (VecStims)
if cfg.addLongConn:
    ## load experimentally based parameters for long range inputs
    with open('conn/conn_long.pkl', 'rb') as fileObj: connLongData = pickle.load(fileObj)
    #ratesLong = connLongData['rates']

    numCells = cfg.numCellsLong
    noise = cfg.noiseLong
    start = cfg.startLong

    longPops = ['TPO', 'TVL', 'S1', 'S2', 'cM1', 'M2', 'OC']
    ## create populations with fixed 
    for longPop in longPops:
        netParams.popParams[longPop] = {'cellModel': 'VecStim', 'numCells': numCells, 'rate': cfg.ratesLong[longPop], 
                                        'noise': noise, 'start': start, 'pulses': [], 'ynormRange': layer['long'+longPop]}
        if isinstance(cfg.ratesLong[longPop], str): # filename to load spikes from
            spikesFile = cfg.ratesLong[longPop]
            with open(spikesFile, 'r') as f: spks = json.load(f)
            netParams.popParams[longPop].pop('rate')
            netParams.popParams[longPop]['spkTimes'] = spks


#------------------------------------------------------------------------------
# Synaptic mechanism parameters
#------------------------------------------------------------------------------
netParams.synMechParams['NMDA'] = {'mod': 'MyExp2SynNMDABB', 'tau1NMDA': 15, 'tau2NMDA': 150, 'e': 0}
netParams.synMechParams['AMPA'] = {'mod':'MyExp2SynBB', 'tau1': 0.05, 'tau2': 5.3, 'e': 0}
netParams.synMechParams['exc'] = {'mod':'Exp2Syn', 'tau1': 0.05, 'tau2': 5.3*cfg.excTau2Factor, 'e': 0}
netParams.synMechParams['GABAB'] = {'mod':'MyExp2SynBB', 'tau1': 3.5, 'tau2': 260.9, 'e': -93} 
netParams.synMechParams['GABAA'] = {'mod':'MyExp2SynBB', 'tau1': 0.07, 'tau2': 18.2, 'e': -80}
netParams.synMechParams['GABAASlow'] = {'mod': 'MyExp2SynBB','tau1': 2, 'tau2': 100, 'e': -80}
netParams.synMechParams['GABAASlowSlow'] = {'mod': 'MyExp2SynBB', 'tau1': 200, 'tau2': 400, 'e': -80}

ESynMech = ['AMPA', 'NMDA']
SOMESynMech = ['GABAASlow','GABAB']
SOMISynMech = ['GABAASlow']
PVSynMech = ['GABAA']


#------------------------------------------------------------------------------
# Long range input pulses
#------------------------------------------------------------------------------
if cfg.addPulses:
    for key in [k for k in dir(cfg) if k.startswith('pulse')]:
        params = getattr(cfg, key, None)
        [pop, start, end, rate, noise] = [params[s] for s in ['pop', 'start', 'end', 'rate', 'noise']]
        if pop in netParams.popParams:
            if 'pulses' not in netParams.popParams[pop]: netParams.popParams[pop]['pulses'] = {}    
            netParams.popParams[pop]['pulses'].append({'start': start, 'end': end, 'rate': rate, 'noise': noise})



#------------------------------------------------------------------------------
# Current inputs (IClamp)
#------------------------------------------------------------------------------
if cfg.addIClamp:
    for key in [k for k in dir(cfg) if k.startswith('IClamp')]:
        params = getattr(cfg, key, None)
        [pop,sec,loc,start,dur,amp] = [params[s] for s in ['pop','sec','loc','start','dur','amp']]

        cfg.analysis['plotTraces']['include'].append((pop,0))  # record that pop

        # add stim source
        netParams.stimSourceParams[key] = {'type': 'IClamp', 'delay': start, 'dur': dur, 'amp': amp}
        
        # connect stim source to target
        netParams.stimTargetParams[key+'_'+pop] =  {
            'source': key, 
            'conds': {'pop': pop},
            'sec': sec, 
            'loc': loc}

#------------------------------------------------------------------------------
# NetStim inputs
#------------------------------------------------------------------------------
if cfg.addNetStim:
    for key in [k for k in dir(cfg) if k.startswith('NetStim')]:
        params = getattr(cfg, key, None)
        [pop, ynorm, sec, loc, synMech, synMechWeightFactor, start, interval, noise, number, weight, delay] = \
        [params[s] for s in ['pop', 'ynorm', 'sec', 'loc', 'synMech', 'synMechWeightFactor', 'start', 'interval', 'noise', 'number', 'weight', 'delay']] 

        cfg.analysis['plotTraces']['include'].append((pop,0))

        if synMech == ESynMech:
            wfrac = cfg.synWeightFractionEE
        elif synMech == SOMESynMech:
            wfrac = cfg.synWeightFractionSOME
        else:
            wfrac = [1.0]

        # add stim source
        netParams.stimSourceParams[key] = {'type': 'NetStim', 'start': start, 'interval': interval, 'noise': noise, 'number': number}

        # connect stim source to target
        # for i, syn in enumerate(synMech):
        netParams.stimTargetParams[key+'_'+pop] =  {
            'source': key, 
            'conds': {'pop': pop, 'ynorm': ynorm},
            'sec': sec, 
            'loc': loc,
            'synMech': synMech,
            'weight': weight,
            'synMechWeightFactor': synMechWeightFactor,
            'delay': delay}

    # -------------------------------------------------------------------------
    # Group of NetStims
    for key in [k for k in dir(cfg) if k.startswith('GroupNetStim')]:
        params = getattr(cfg, key, None)
        nstype, numStims, pop, ynorm, cellRule, secList, allSegs, synMech, start, interval, noise, number, weight, delay = [params[s] for s in ['nstype', 'numStims', 'pop', 'ynorm', 'cellRule', 'secList', 'allSegs', 'synMech', 'start', 'interval', 'noise', 'number', 'weight', 'delay']]

        cfg.analysis['plotTraces']['include'].append((pop,0))

        if not isinstance(secList, list):
            secList = list(netParams.cellParams[cellRule]['secLists'][secList])

        istim = 0
        segs = []

        if synMech == ESynMech:
            wfrac = cfg.synWeightFractionEE
        elif synMech == SOMESynMech:
            wfrac = cfg.synWeightFractionSOME
        else:
            wfrac = [1.0]

        if nstype == 'stim':  # implement as a stim
            while istim < numStims:
                for secName,sec in netParams.cellParams[cellRule]['secs'].items():
                    if secName in secList:
                        if allSegs:
                            nseg = sec['geom']['nseg']
                            for iseg in range(nseg):
                                segs.append([secName, (iseg+1)*(1.0/(nseg+1))])
                                istim += 1
                                if istim >= numStims: break
                        else:
                            segs.append([secName, 0.5])
                            istim += 1
                        
                        if istim >= numStims: break

            for istim, seg in enumerate(segs):

                # add stim source
                netParams.stimSourceParams[key+'_'+str(istim)] = {'type': 'NetStim', 'start': start, 'interval': interval, 'noise': noise, 'number': number}

                # connect stim source to target
                for i, syn in enumerate(synMech):
                    netParams.stimTargetParams[key+'_'+pop+'_'+syn+'_'+str(istim)] =  {
                        'source': key+'_'+str(istim), 
                        'conds': {'pop': pop, 'ynorm': ynorm},
                        'sec': sec, 
                        'loc': loc,
                        'synMech': syn,
                        'weight': weight*wfrac[i],
                        'delay': delay}

        elif nstype == 'pop':  # implement as a pop
            netParams.popParams[key] = {'cellModel': 'NetStim', 'numCells': numStims, 'rate': cfg.groupRate if cfg.groupRate else 1000/interval,
                                         'noise': noise, 'start': start, 'number': number}
            
            netParams.connParams[key] = { 
                        'preConds': {'pop': key}, 
                        'postConds': {'pop': pop, 'ynorm': ynorm},
                        'synMech': synMech,
                        'weight': cfg.groupWeight if cfg.groupWeight is not None else weight, 
                        'synMechWeightFactor': wfrac,
                        'delay': delay,
                        'synsPerConn': 1,
                        'sec': secList}
            
            netParams.subConnParams[key] = {
                        'preConds': {'pop': key}, 
                        'postConds': {'pop': pop, 'ynorm': ynorm},  
                        'sec': secList, 
                        'groupSynMechs': ESynMech, 
                        'density': 'uniform'} 


#------------------------------------------------------------------------------
# Network clamp (integrate into netpyne?)
#------------------------------------------------------------------------------
if cfg.netClamp:
    # load full files and save reduced file
    import os.path
    from netpyne.sim import ijsonLoad

    tagsFile = cfg.netClampConnsFile[:-5]+'_netClampTags_%d.json'%(cfg.netClampGid)
    connsFile = cfg.netClampConnsFile[:-5]+'_netClampConns_%d.json'%(cfg.netClampGid)

    if not os.path.isfile(tagsFile) and not os.path.isfile(connsFile): # load both tags+conns
        ijsonLoad(cfg.netClampConnsFile, tagsGidRange=None, connsGidRange=[cfg.netClampGid], loadTags=True, loadConns=True, tagFormat=['pop'], connFormat=None, 
                        saveTags=tagsFile, saveConns=connsFile)
    elif not os.path.isfile(tagsFile): # load tags
        ijsonLoad(cfg.netClampConnsFile, tagsGidRange=None, loadTags=True, loadConns=False, tagFormat=['pop'], saveTags=tagsFile)
    elif not os.path.isfile(connsFile): # load conns
        ijsonLoad(cfg.netClampConnsFile, connsGidRange=[cfg.netClampGid], loadTags=False, loadConns=True, connFormat=None, saveConns=connsFile)

    # load reduced files
    with open(tagsFile, 'r') as fileObj: tags = json.load(fileObj)['tags']
    tags = {int(k) if k != 'format' else k:v  for k,v in tags.items()}
    with open(connsFile, 'r') as fileObj: conns = json.load(fileObj)['conns']#[cfg.netClampGid]
    conns = {int(k) if k != 'format' else k:v  for k,v in conns.items()}
    conns = conns[cfg.netClampGid]

    # load list of preGid cell
    preGids = list(set([conn['preGid'] for conn in conns]))

    # load spk times of preGid cells
    with open(cfg.netClampSpikesFile, 'r') as fileObj: data = json.load(fileObj)
    allSpkids,allSpkts = data['simData']['spkid'], data['simData']['spkt']
    # Note tags is imported using 'format', so is list of tags, in this case just 'pop'
    popIndex = tags['format'].index('pop')
    spkids,spkts,spkpops = zip(*[(spkid,spkt,tags[int(spkid)][popIndex]) for spkid,spkt in zip(allSpkids,allSpkts) if spkid in preGids])

    # group by prepops
    prePops = list(set(spkpops))

    # create all conns (even if no input spikes)    
    if cfg.netClampCreateAllConns:
        prePops = list(set([tag[popIndex] for tag in tags.values()]))
        for prePop in prePops:
            print('Finding conns from pop %s ...'%(str(prePop)))
            key = str(prePop)
            popGids = [i for i,tag in enumerate(tags.values()) if tag[popIndex]==prePop]
            # create 1 vectstim pop per conn (cellLabel=cell gid -- so can reference later)
            cellsList = []
            netParams.popParams[key] = {'cellModel': 'VecStim', 'cellsList': []}
            for popGid in popGids:
                cellsList.append({'cellLabel': int(popGid), 'spkTimes': [1]})
                netParams.popParams[key]['cellsList'] = list(cellsList)

                # calculate conns for this preGid
                preConns = [conn for conn in conns if conn['preGid'] == popGid]
                for i,preConn in enumerate(preConns):
                    netParams.connParams[key+'_'+str(popGid)+'_'+str(i)] = { 
                            'preConds': {'cellLabel': preConn['preGid']},  # cellLabel corresponds to gid 
                            'postConds': {'pop': cfg.netClampPop},
                            'synMech': str(preConn['synMech']),
                            'weight': float(preConn['weight']), 
                            'delay': float(preConn['delay']),
                            'sec': str(preConn['sec']), 
                            'loc': str(preConn['loc'])}

    else:
        # create one vectsim pop reproducing spk times per preGid; and multiple conns 
        for prePop in prePops:
            # get spkTimes for preGid
            spkGids = [spkid for spkid,spkpop in zip(spkids,spkpops) if spkpop == prePop]

            if len(spkGids) > 0 or cfg.netClampCreateAllConns:
                # get list of cell gids in this pop 
                popGids = list(set(spkGids))

                # set key/label of pop
                key = '' + str(prePop)

                # create 1 vectstim pop per conn (cellLabel=cell gid -- so can reference later)
                cellsList = []
                for popGid in popGids: 
                    spkTimes = [spkt for spkid,spkt in zip(spkids,spkts) if spkid == popGid]
                    if len(spkTimes) > 0:
                        cellsList.append({'cellLabel': int(popGid), 'spkTimes': spkTimes})
                        netParams.popParams[key] = {'cellModel': 'VecStim', 'cellsList': cellsList}

                        # calculate conns for this preGid
                        preConns = [conn for conn in conns if conn['preGid'] == popGid]

                        for i,preConn in enumerate(preConns):
                            netParams.connParams[key+'_'+str(popGid)+'_'+str(i)] = { 
                                    'preConds': {'cellLabel': preConn['preGid']},  # cellLabel corresponds to gid 
                                    'postConds': {'pop': cfg.netClampPop},
                                    'synMech': str(preConn['synMech']),
                                    'weight': float(preConn['weight']), 
                                    'delay': float(preConn['delay']),
                                    'sec': str(preConn['sec']), 
                                    'loc': str(preConn['loc'])}

        # longPops = ['TPO', 'TVL', 'S1', 'S2', 'cM1', 'M2', 'OC']

        # #Modify netClamp pops
        # rate = 50

        # for cell in netParams.popParams['PV5B']['cellsList']:
        #     cell['spkTimes'] = []# list(random.uniform(0,cfg.duration,(rate*cfg.duration/1000)))
        

#------------------------------------------------------------------------------
# Local connectivity parameters
#------------------------------------------------------------------------------
try:
    with open('conn/conn.pkl', 'r') as fileObj: connData = pickle.load(fileObj)
    bins = connData['bins']
    pmat = connData['pmat']
    wmat = connData['wmat']
except:
    pass
#------------------------------------------------------------------------------
## E -> E
if cfg.addConn and cfg.EEGain > 0.0:
    labelsConns = [('W+AS_norm', 'IT', 'L2/3,4'), ('W+AS_norm', 'IT', 'L5A,5B'), 
                   ('W+AS_norm', 'PT', 'L5B'), ('W+AS_norm', 'IT', 'L6'), ('W+AS_norm', 'CT', 'L6')]
    labelPostBins = [('W+AS', 'IT', 'L2/3,4'), ('W+AS', 'IT', 'L5A,5B'), ('W+AS', 'PT', 'L5B'), 
                    ('W+AS', 'IT', 'L6'), ('W+AS', 'CT', 'L6')]
    labelPreBins = ['W', 'AS', 'AS', 'W', 'W']
    preTypes = [['IT'], ['IT'], ['IT', 'PT'], ['IT','CT'], ['IT','CT']] 
    postTypes = ['IT', 'IT', 'PT', 'IT','CT']
    ESynMech = ['AMPA','NMDA']

    for i,(label, preBinLabel, postBinLabel) in enumerate(zip(labelsConns,labelPreBins, labelPostBins)):
        for ipre, preBin in enumerate(bins[preBinLabel]):
            for ipost, postBin in enumerate(bins[postBinLabel]):
                for cellModel in cellModels:
                    ruleLabel = 'EE_'+cellModel+'_'+str(i)+'_'+str(ipre)+'_'+str(ipost)
                    netParams.connParams[ruleLabel] = { 
                        'preConds': {'cellType': preTypes[i], 'ynorm': list(preBin)}, 
                        'postConds': {'cellModel': cellModel, 'cellType': postTypes[i], 'ynorm': list(postBin)},
                        'synMech': ESynMech,
                        'probability': pmat[label][ipost,ipre],
                        'weight': wmat[label][ipost,ipre] * cfg.EEGain / cfg.synsperconn[cellModel], 
                        'synMechWeightFactor': cfg.synWeightFractionEE,
                        'delay': 'defaultDelay+dist_3D/propVelocity',
                        'synsPerConn': cfg.synsperconn[cellModel],
                        'sec': 'spiny'}
            

#------------------------------------------------------------------------------
## E -> I
if cfg.EIGain: # Use IEGain if value set
    cfg.EPVGain = cfg.EIGain
    cfg.ESOMGain = cfg.EIGain
else: 
    cfg.EIGain = (cfg.EPVGain+cfg.ESOMGain)/2.0

if cfg.addConn and (cfg.EPVGain > 0.0 or cfg.ESOMGain > 0.0):
    labelsConns = ['FS', 'LTS']
    labelPostBins = ['FS/LTS', 'FS/LTS']
    labelPreBins = ['FS/LTS', 'FS/LTS']
    preTypes = ['IT', 'PT', 'CT']
    postTypes = ['PV', 'SOM']
    ESynMech = ['AMPA','NMDA']
    lGain = [cfg.EPVGain, cfg.ESOMGain] # E -> PV or E -> SOM
    for i,(label, preBinLabel, postBinLabel) in enumerate(zip(labelsConns,labelPreBins, labelPostBins)):
        for ipre, preBin in enumerate(bins[preBinLabel]):
            for ipost, postBin in enumerate(bins[postBinLabel]):
                ruleLabel = 'EI_'+str(i)+'_'+str(ipre)+'_'+str(ipost)
                netParams.connParams[ruleLabel] = {
                    'preConds': {'cellType': preTypes, 'ynorm': list(preBin)},
                    'postConds': {'cellType': postTypes[i], 'ynorm': list(postBin)},
                    'synMech': ESynMech,
                    'probability': pmat[label][ipost,ipre],
                    'weight': wmat[label][ipost,ipre] * lGain[i],
                    'synMechWeightFactor': cfg.synWeightFractionEI,
                    'delay': 'defaultDelay+dist_3D/propVelocity',
                    'sec': 'soma'} # simple I cells used right now only have soma


#------------------------------------------------------------------------------
## I -> all
if cfg.IEGain: # Use IEGain if value set
    cfg.PVEGain = cfg.IEGain
    cfg.SOMEGain = cfg.IEGain
else: 
    cfg.IEGain = (cfg.PVEGain+cfg.SOMEGain)/2.0

if cfg.IIGain:  # Use IIGain if value set
    cfg.SOMPVGain = cfg.IIGain
    cfg.PVSOMGain = cfg.IIGain
    cfg.SOMSOMGain = cfg.IIGain
    cfg.PVPVGain = cfg.IIGain
else:
    cfg.IIGain = (cfg.PVSOMGain+cfg.SOMPVGain+cfg.SOMSOMGain+cfg.PVPVGain)/4.0

if cfg.addConn and (cfg.IEGain > 0.0 or cfg.IIGain > 0.0):
    # Local, intralaminar only; all-to-all but distance-based; high weights; L5A/B->L5A/B
    preCellTypes = ['SOM', 'SOM', 'SOM', 'PV', 'PV', 'PV']
    ynorms = [(layer['2'][0],layer['4'][1]), (layer['5A'][0],layer['5B'][1]), (layer['6'][0],layer['6'][1])]*2
    IEweights = cfg.IEweights * 2  # [I->E2/3+4, I->E5, I->E6] weights (Note * 2 is repeat list operator)
    IIweights = cfg.IIweights * 2  # [I->I2/3+4, I->I5, I->I6] weights (Note * 2 is repeat list operator)
    postCellTypes = ['PT', ['IT','CT'], 'PV', 'SOM']
    IEdisynBiases = [None, cfg.IEdisynapticBias, cfg.IEdisynapticBias, None, cfg.IEdisynapticBias, cfg.IEdisynapticBias]
    disynapticBias = None  # default, used for I->I

    for i,(preCellType, ynorm, IEweight, IIweight, IEdisynBias) in enumerate(zip(preCellTypes, ynorms, IEweights, IIweights, IEdisynBiases)):
        for ipost, postCellType in enumerate(postCellTypes):
            for cellModel in cellModels:
                if postCellType == 'PV':    # postsynaptic I cell
                    sec = 'soma'
                    synWeightFraction = [1]
                    if preCellType == 'PV':             # PV->PV
                        weight = IIweight * cfg.PVPVGain
                        synMech = PVSynMech
                    else:                           # SOM->PV
                        weight = IIweight * cfg.SOMPVGain
                        synMech = SOMISynMech
                elif postCellType == 'SOM': # postsynaptic I cell
                    sec = 'soma'
                    synWeightFraction = [1]
                    if preCellType == 'PV':             # PV->SOM
                        weight = IIweight * cfg.PVSOMGain
                        synMech = PVSynMech
                    else:                           # SOM->SOM
                        weight = IIweight * cfg.SOMSOMGain
                        synMech = SOMISynMech
                elif postCellType == ['IT','CT']: # postsynaptic IT,CT cell
                    disynapticBias = IEdisynBias
                    if preCellType == 'PV':             # PV->E
                        weight = IEweight * cfg.PVEGain
                        synMech = PVSynMech
                        sec = 'perisom'
                    else:                           # SOM->E
                        weight = IEweight * cfg.SOMEGain
                        synMech = SOMESynMech
                        sec = 'spiny'
                        synWeightFraction = cfg.synWeightFractionSOME
                elif postCellType == 'PT': # postsynaptic PT cell
                    disynapticBias = IEdisynBias
                    if preCellType == 'PV':             # PV->E
                        weight = IEweight * cfg.IPTGain * cfg.PVEGain
                        synMech = PVSynMech
                        sec = 'perisom'
                    else:                           # SOM->E
                        weight = IEweight * cfg.IPTGain * cfg.SOMEGain
                        synMech = SOMESynMech
                        sec = 'spiny'
                        synWeightFraction = cfg.synWeightFractionSOME
                if cellModel == 'HH_full':
                    weight = weight * cfg.IFullGain


                ruleLabel = 'I_'+cellModel+'_'+str(i)+'_'+str(ipost)
                netParams.connParams[ruleLabel] = {
                    'preConds': {'cellType': preCellType, 'ynorm': ynorm},
                    'postConds': {'cellModel': cellModel, 'cellType': postCellType, 'ynorm': ynorm},
                    'synMech': synMech,
                    'probability': '1.0 * exp(-dist_3D/probLambda)',
                    'weight': weight / cfg.synsperconn[cellModel],
                    'delay': 'defaultDelay+dist_3D/propVelocity',
                    'synsPerConn': cfg.synsperconn[cellModel],
                    'synMechWeightFactor': synWeightFraction,
                    'sec': sec,
                    'disynapticBias': disynapticBias}


#------------------------------------------------------------------------------
# Long-range  connectivity parameters
#------------------------------------------------------------------------------
if cfg.addLongConn:

    # load load experimentally based parameters for long range inputs
    cmatLong = connLongData['cmat']
    binsLong = connLongData['bins']

    longPops = ['TPO', 'TVL', 'S1', 'S2', 'cM1', 'M2', 'OC']
    cellTypes = ['IT', 'PT', 'CT', 'PV', 'SOM']
    EorI = ['exc', 'inh']
    syns = {'exc': ESynMech, 'inh': 'GABAA'}
    synFracs = {'exc': cfg.synWeightFractionEE, 'inh': [1.0]}

    for longPop in longPops:
        for ct in cellTypes:
            for EorI in ['exc', 'inh']:
                for i, (binRange, convergence) in enumerate(zip(binsLong[(longPop, ct)], cmatLong[(longPop, ct, EorI)])):
                    for cellModel in cellModels:
                        ruleLabel = longPop+'_'+ct+'_'+EorI+'_'+cellModel+'_'+str(i)
                        netParams.connParams[ruleLabel] = { 
                            'preConds': {'pop': longPop}, 
                            'postConds': {'cellModel': cellModel, 'cellType': ct, 'ynorm': list(binRange)},
                            'synMech': syns[EorI],
                            'convergence': convergence,
                            'weight': cfg.weightLong / cfg.synsperconn[cellModel], 
                            'synMechWeightFactor': cfg.synWeightFractionEE,
                            'delay': 'defaultDelay+dist_3D/propVelocity',
                            'synsPerConn': cfg.synsperconn[cellModel],
                            'sec': 'spiny'}


#------------------------------------------------------------------------------
# Subcellular connectivity (synaptic distributions)
#------------------------------------------------------------------------------         

#------------------------------------------------------------------------------
# NetStim (LSPS pixel) -> PT (Sheets fig4)
if cfg.ihSubConn:
    lenX = 8
    lenY = 24 
    spacing = 50
    fixedSomaY = -735
    gridX = range(0, spacing*lenX, spacing)
    gridY = range(0, -spacing*lenY, -spacing)

    map2d = [[0 for _ in range(lenY)] for _ in range(lenX)]
    map2d[cfg.ihX][cfg.ihY] = 1

    netParams.subConnParams['ih'] = {
        'preConds': {'pop': ['GroupNetStimEPT','GroupNetStimEPT2']}, 
        'postConds': {'cellType': ['PT','PT2']},  
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': {'type': '2Dmap', 'gridX': gridX, 'gridY': gridY, 'gridValues': map2d, 'fixedSomaY': fixedSomaY}} 

#------------------------------------------------------------------------------
# NetStim (L2/3 stand-in) -> PT (Sheets fig11)
if cfg.stimSubConn:
    with open('conn/conn_dend_PT.json', 'r') as fileObj: connDendPTData = json.load(fileObj)
    with open('conn/conn_dend_IT.json', 'r') as fileObj: connDendITData = json.load(fileObj)
    

    lenY = 30 
    spacing = 50
    gridY = range(0, -spacing*lenY, -spacing)
    synDens, _, fixedSomaY = connDendPTData['synDens'], connDendPTData['gridY'], connDendPTData['fixedSomaY']
    for k in synDens.keys():
        prePop,postType = k.split('_')  # eg. split 'M2_PT'
        if prePop == 'L2': prePop = ['GroupNetStimEPT', 'GroupNetStimEPT2']  # include conns from layer 2/3 and 4
        netParams.subConnParams[k] = {
        'preConds': {'pop': prePop}, 
        'postConds': {'cellType': ['PT', 'PT2']},  
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': {'type': '1Dmap', 'gridX': None, 'gridY': gridY, 'gridValues': synDens[k], 'fixedSomaY': fixedSomaY}} 


#------------------------------------------------------------------------------
# Network model
if cfg.addSubConn:
    with open('conn/conn_dend_PT.json', 'r') as fileObj: connDendPTData = json.load(fileObj)
    with open('conn/conn_dend_IT.json', 'r') as fileObj: connDendITData = json.load(fileObj)
    
    #------------------------------------------------------------------------------
    # L2/3,TVL,S2,cM1,M2 -> PT (Suter, 2015)
    lenY = 30 
    spacing = 50
    gridY = range(0, -spacing*lenY, -spacing)
    synDens, _, fixedSomaY = connDendPTData['synDens'], connDendPTData['gridY'], connDendPTData['fixedSomaY']
    for k in synDens.keys():
        prePop,postType = k.split('_')  # eg. split 'M2_PT'
        if prePop == 'L2': prePop = 'IT2'  # include conns from layer 2/3 and 4
        netParams.subConnParams[k] = {
        'preConds': {'pop': prePop}, 
        'postConds': {'cellType': postType},  
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': {'type': '1Dmap', 'gridX': None, 'gridY': gridY, 'gridValues': synDens[k], 'fixedSomaY': fixedSomaY}} 


    #------------------------------------------------------------------------------
    # TPO, TVL, M2, OC  -> E (L2/3, L5A, L5B, L6) (Hooks 2013)
    lenY = 26
    spacing = 50
    gridY = range(0, -spacing*lenY, -spacing)
    synDens, _, fixedSomaY = connDendITData['synDens'], connDendITData['gridY'], connDendITData['fixedSomaY']
    for k in synDens.keys():
        prePop,post = k.split('_')  # eg. split 'M2_L2'
        postCellTypes = ['IT','PT','CT'] if prePop in ['OC','TPO'] else ['IT','CT']  # only OC,TPO include PT cells
        postyRange = list(layer[post.split('L')[1]]) # get layer yfrac range 
        if post == 'L2': postyRange[1] = layer['4'][1]  # apply L2 rule also to L4 
        netParams.subConnParams[k] = {
        'preConds': {'pop': prePop}, 
        'postConds': {'ynorm': postyRange , 'cellType': postCellTypes},  
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': {'type': '1Dmap', 'gridX': None, 'gridY': gridY, 'gridValues': synDens[k], 'fixedSomaY': fixedSomaY}} 


    #------------------------------------------------------------------------------
    # S1, S2, cM1 -> E IT/CT; no data, assume uniform over spiny
    netParams.subConnParams['S1,S2,cM1->IT,CT'] = {
        'preConds': {'pop': ['S1','S2','cM1']}, 
        'postConds': {'cellType': ['IT','CT']},
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': 'uniform'} 


    #------------------------------------------------------------------------------
    # rest of local E->E (exclude IT2->PT); uniform distribution over spiny
    netParams.subConnParams['IT2->non-PT'] = {
        'preConds': {'pop': ['IT2']}, 
        'postConds': {'cellType': ['IT','CT']},
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': 'uniform'} 
        
    netParams.subConnParams['non-IT2->E'] = {
        'preConds': {'pop': ['IT4','IT5A','IT5B','PT5B','IT6','CT6']}, 
        'postConds': {'cellType': ['IT','PT','CT']},
        'sec': 'spiny',
        'groupSynMechs': ESynMech, 
        'density': 'uniform'} 


    #------------------------------------------------------------------------------
    # PV->E; perisomatic (no sCRACM)
    netParams.subConnParams['PV->E'] = {
        'preConds': {'cellType': 'PV'}, 
        'postConds': {'cellType': ['IT', 'CT', 'PT']},  
        'sec': 'perisom', 
        'density': 'uniform'} 


    #------------------------------------------------------------------------------
    # SOM->E; apical dendrites (no sCRACM)
    netParams.subConnParams['SOM->E'] = {
        'preConds': {'cellType': 'SOM'}, 
        'postConds': {'cellType': ['IT', 'CT', 'PT']},  
        'sec': 'apicdend',
        'groupSynMechs': SOMESynMech,
        'density': 'uniform'} 


    #------------------------------------------------------------------------------
    # All->I; apical dendrites (no sCRACM)
    netParams.subConnParams['All->I'] = {
        'preConds': {'cellType': ['IT', 'CT', 'PT', 'SOM', 'PV']+longPops}, 
        'postConds': {'cellType': ['SOM', 'PV']},  
        'sec': 'spiny',
        'groupSynMechs': ESynMech,
        'density': 'uniform'} 


#------------------------------------------------------------------------------
# Description
#------------------------------------------------------------------------------
netParams.description = """ 
- M1 net, 6 layers, 7 cell types 
- NCD-based connectivity from  Weiler et al. 2008; Anderson et al. 2010; Kiritani et al. 2012; 
  Yamawaki & Shepherd 2015; Apicella et al. 2012
- Parametrized version based on Sam's code
- Updated cell models and mod files
- Added parametrized current inputs
- Fixed bug: prev was using cell models in /usr/site/nrniv/local/python/ instead of cells 
- Use 5 synsperconn for 5-comp cells (HH_reduced); and 1 for 1-comp cells (HH_simple)
- Fixed bug: made global h params separate for each cell model
- Fixed v_init for different cell models
- New IT cell with same geom as PT
- Cleaned cfg and moved background inputs here
- Set EIGain and IEGain for each inh cell type
- Added secLists for PT full
- Fixed reduced CT (wrong vinit and file)
- Added subcellular conn rules to distribute synapses
- PT full model soma centered at 0,0,0 
- Set cfg seeds here to ensure they get updated
- Added PVSOMGain and SOMPVGain
- PT subcellular distribution as a cfg param
- Cylindrical volume
- DefaultDelay (for local conns) = 2ms
- Added long range connections based on Yamawaki 2015a,b; Suter 2015; Hooks 2013; Meyer 2011
- Updated cell densities based on Tsai 2009; Lefort 2009; Katz 2011; Wall 2016; 
- Separated PV and SOM of L5A vs L5B
- Fixed bugs in local conn (PT, PV5, SOM5, L6)
- Added perisom secList including all sections 50um from soma
- Added subcellular conn rules (for both full and reduced models)
- Improved cell models, including PV and SOM fI curves
- Improved subcell conn rules based on data from Suter15, Hooks13 and others
- Adapted Bdend L of reduced cell models
- Made long pop rates a cfg param
- Set threshold to 0.0 mV
- Parametrized I->E/I layer weights
- Added missing subconn rules (IT6->PT; S1,S2,cM1->IT/CT; long->SOM/PV)
- Added threshold to weightNorm (PT threshold=10x)
- weightNorm threshold as a cfg parameter
- Separate PV->SOM, SOM->PV, SOM->SOM, PV->PV gains 
- Conn changes: reduced IT2->IT4, IT5B->CT6, IT5B,6->IT2,4,5A, IT2,4,5A,6->IT5B; increased CT->PV6+SOM6
- Parametrized PT ih gbar 
- Added IFullGain parameter: I->E gain for full detailed cell models
- Replace PT ih with Migliore 2012
- Parametrized ihGbar, ihGbarBasal, dendNa, axonNa, axonRa, removeNa, excTauFactor 
- Replaced cfg list params with dicts
- Parametrized ihLkcBasal
- Fixed synMechWeightFactor
- Parametrized PT ih slope and ihlkcBelowSoma
- Added disynapticBias to I->E (Yamawaki&Shepherd,2015)
- Fixed E->CT bin 0.9-1.0
- Replaced GABAB with exp2syn and adapted synMech ratios
- Parametrized somaNa
- Added ynorm condition to NetStims
- Added option to play back recorded spikes into long-range inputs
- Fixed Bdend pt3d y location
- Added netParams.convertCellShapes = True to convert stylized geoms to 3d points
- New layer boundaries, cell densities, conn, FS+SOM L4 grouped with L2/3, low cortical input to L4
- Increased exc->L4 based on Yamawaki 2015 fig 5
"""