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
from spike_guard import collect_local_guard_metrics, summarize_guard_metrics, blockade_penalty, guard_summary_for_results


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

if getattr(cfg, 'spikeGuard', {}).get('enabled', False):
    sim.simData['spikeGuardLocal'] = collect_local_guard_metrics(sim.net.cells, cfg.spikeGuard)

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

if sim.rank == 0 and getattr(cfg, 'spikeGuard', {}).get('enabled', False):
    spikeGuardSummary = summarize_guard_metrics(
        sim.allSimData.get('spikeGuardLocal', {}),
        cfg.spikeGuard,
        bool(cfg.singleCellPops),
    )
    sim.simData['spikeGuard'] = spikeGuardSummary
    sim.allSimData['spikeGuard'] = spikeGuardSummary

sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#
sim.analysis.plotData()         			# plot spike raster etc

print('completed simulation...')

if sim.rank == 0:
    # netParams.save("{}/{}_params.json".format(cfg.saveFolder, cfg.simLabel))
    print('transmitting data...')
    if not hasattr(cfg, 'get_mappings'):
        raise AttributeError("cfg.get_mappings() is required for result export")
    inputs = cfg.get_mappings()
    # print(json.dumps({**inputs}))
    # results = sim.analysis.popAvgRates(tranges=cfg.timeRanges, show=False) #TODO: Avoid printing firing rates
    results = sim.analysis.popAvgRates(tranges=cfg.printPopAvgRates, show=False) #TODO: Avoid printing firing rates


    sim.simData['popRates'] = results

    fitnessFuncArgs = {}
    pops = {}
    ## Exc pops
    Epops = ['IT2', 'IT4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6']
    Etune = {'target': 5, 'width': 5, 'min': 0.5}
    for pop in Epops:
        pops[pop] = Etune
    ## Inh pops
    Ipops = ['NGF1', 'PV2', 'SOM2', 'VIP2', 'NGF2',
             'PV4', 'SOM4', 'VIP4', 'NGF4',
             'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',
             'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',
             'PV6', 'SOM6', 'VIP6', 'NGF6']
    Itune = {'target': 10, 'width': 15, 'min': 0.25}
    for pop in Ipops:
        pops[pop] = Itune
    fitnessFuncArgs['pops'] = pops
    fitnessFuncArgs['maxFitness'] = 1000
    fitnessFuncArgs['tranges'] = cfg.printPopAvgRates

    # rateLoss = defs.rateFitnessFunc(sim.simData, **fitnessFuncArgs)
    rateLoss = defs.rateFitnessFuncTranges(sim.simData, **fitnessFuncArgs)

    spikeGuardInfo = None
    if getattr(cfg, 'spikeGuard', {}).get('enabled', False):
        spikeGuardSummary = summarize_guard_metrics(
            sim.allSimData.get('spikeGuardLocal', {}),
            cfg.spikeGuard,
            bool(cfg.singleCellPops),
        )
        spikeGuardInfo = guard_summary_for_results(spikeGuardSummary)
        print("spikeGuard:", spikeGuardInfo)
        rateLoss += float(blockade_penalty(spikeGuardSummary, cfg.spikeGuard))

    print("popRates:", results)
    print("loss:", rateLoss)

    payload = {
        **inputs,
        "loss": float(rateLoss),
    }

    out_json = json.dumps(payload)
    print(out_json)
    sim.send(out_json)
