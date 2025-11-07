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
from netParams_UMAP import netParams, cfg
from pathlib import Path
import defs


sim.initialize(
    simConfig = cfg, 	
    netParams = netParams)  				# create network object and set cfg and net params
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations
sim.net.connectCells()            			# create connections between cells based on params
sim.net.addStims() 							# add network stimulation
sim.setupRecording()

# print(cfg.saveFolder, cfg.simLabel)

#------------------------------------------------------------------------------
# Simulation option 1: standard
if cfg.period == 'full_trial':
    print(cfg.modifyMechs)
    sim.runSimWithIntervalFunc(cfg.preTone, defs.modifyMechsFunc, funcArgs={'cfg': cfg})       # run parallel Neuron simulation (calling func to modify mechs)
else:
    sim.runSim()                              # run parallel Neuron simulation (calling func to modify mechs)

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
    print('transmitting data...')
    inputs = cfg.get_mappings()
    cfg.sampled_cells = defs.sampleNeuronsFromModel(sim, cfg, plot=False)

    ModelRates = defs.binnedRaster(sim.simData, cfg)

    ConcatenatedRates, ConcatenatedLabels = defs.concatenateExpModelRate(cfg.RawData, ModelRates)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(ConcatenatedRates, aspect='auto')
    plt.colorbar(label='Firing rate (Hz)')
    filename = cfg.saveFolder + "/" + cfg.simLabel + "_ratesConcat.png"
    plt.savefig(filename)
    plt.close()

    # TO DO: CALCULATE THE FITNESS FUNCTION WITH UMAP + LABELS
    umap_representation, umap_reduction, pearsonCorr, pvalue = defs.calculateUMAP(ConcatenatedRates, cfg)
    # print(np.shape(umap_representation), len(ConcatenatedLabels), pearsonCorr)
    defs.plot_embedding(umap_representation, ConcatenatedLabels, cfg)

    wasserstein_dist, sw_dist, D_rms, disparity = defs.umapFitnessFunc(umap_representation, ConcatenatedLabels)
    results = {}
    results['loss'] = D_rms.item() # sw_dist
    results['wasserstein_dist'] = wasserstein_dist # wasserstein_dist
    results['sw_dist'] = sw_dist # sw_dist
    out_json = json.dumps({**inputs, **results})

    # print(out_json)
    sim.send(out_json)