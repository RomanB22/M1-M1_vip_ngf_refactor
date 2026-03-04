"""
init.py

Starting script to run NetPyNE-based M1 model.

Usage:
    python init.py # Run simulation, optionally plot a raster

MPI usage:
    mpiexec -n 4 nrniv -python -mpi init.py

Contributors: salvadordura@gmail.com
"""

import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers
from netpyne import sim
import json
from netParams import netParams, cfg
from pathlib import Path
import defs


sim.initialize(
    simConfig = cfg, 	
    netParams = netParams)  				# create network object and set cfg and net params
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations

# print("Checking weightNorm vs nseg for all cells...")
# for cell in sim.net.cells:
#     for sec_name, sec in cell.secs.items():
#         wn = sec.get('weightNorm', None)
#         if wn is None:
#             continue
#         try:
#             nseg = int(sec['hObj'].nseg)
#         except Exception:
#             continue
#         if len(wn) != nseg:
#             print(f"Cell GID {cell.gid}, sec {sec_name}: len(weightNorm)={len(wn)}, nseg={nseg}")
# quit()


sim.net.connectCells()            			# create connections between cells based on params
sim.net.addStims() 							# add network stimulation
sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
#------------------------------------------------------------------------------
# Simulation option 1: standard
sim.runSim()                              # run parallel Neuron simulation (calling func to modify mechs)
# # Simulation option 2: interval function to modify mechanism params
#TODO: Check that it works properly on CoreNEURON
# print(cfg.modifyMechs)
# sim.runSimWithIntervalFunc(cfg.transient+cfg.preTone, defs.modifyMechsFunc, funcArgs={'cfg': cfg})       # run parallel Neuron simulation (calling func to modify mechs)

sim.gatherData()                  			# gather spiking data and cell info from each node
# Gather/save data option 2: distributed saving across nodes
# sim.saveDataInNodes()
# sim.gatherDataFromFiles()

sim.simData.numSampledCellsPerLayer = cfg.numSampledCellsPerLayer
sim.simData.norm_layers = cfg.layer

sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#
sim.analysis.plotData()         			# plot spike raster etc

print('completed simulation...')

if sim.rank == 0:
    # netParams.save("{}/{}_params.json".format(cfg.saveFolder, cfg.simLabel))
    print('transmitting data...')
    inputs = cfg.get_mappings()
    # print(json.dumps({**inputs}))
    results = sim.analysis.popAvgRates(tranges=cfg.printPopAvgRates, show=False) #TODO: Avoid printing firing rates

    sim.simData['popRates'] = results

    fitnessFuncArgs = {}
    pops = {}
    available_pop_rates = set(results.keys())
    ## Exc pops
    Epops = [pop for pop in ['IT2', 'IT4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6'] if pop in available_pop_rates]
    Etune = {'target': 5, 'width': 5, 'min': 0.5}
    for pop in Epops:
        pops[pop] = Etune
    ## Inh pops
    Ipops = [
        pop for pop in [
            'NGF1', 'PV2', 'SOM2', 'VIP2', 'NGF2',
            'PV4', 'SOM4', 'VIP4', 'NGF4',
            'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',
            'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',
            'PV6', 'SOM6', 'VIP6', 'NGF6'
        ] if pop in available_pop_rates
    ]
    Itune = {'target': 10, 'width': 15, 'min': 0.25}
    for pop in Ipops:
        pops[pop] = Itune
    fitnessFuncArgs['pops'] = pops
    fitnessFuncArgs['maxFitness'] = 1000
    fitnessFuncArgs['tranges'] = cfg.printPopAvgRates

    # rateLoss = defs.rateFitnessFunc(sim.simData, **fitnessFuncArgs)
    
    rateLoss = defs.rateFitnessFuncTranges(sim.simData, **fitnessFuncArgs)

    results['loss'] = rateLoss
    out_json = json.dumps({**inputs, **results})

    print(out_json)
    sim.send(out_json)
