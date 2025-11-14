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
import matplotlib.pyplot as plt
from netpyne import sim
import json
from netParams_CEBRA import netParams, cfg
from pathlib import Path
import defs
import numpy as np

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
    print(cfg.sampled_cells)
    
    ModelRates = defs.binnedRaster(sim.simData, cfg).T

    ModelRatesNorm = (ModelRates - ModelRates.mean(axis=0, keepdims=True)) / np.where(ModelRates.std(axis=0, keepdims=True)==0, 1, ModelRates.std(axis=0, keepdims=True))
    ExpRatesNorm = (cfg.neural_data - cfg.neural_data.mean(axis=0, keepdims=True)) / np.where(cfg.neural_data.std(axis=0, keepdims=True)==0, 1, cfg.neural_data.std(axis=0, keepdims=True))

    cfg.neural_data = ExpRatesNorm

    n = cfg.neural_data.shape[0]      # target time length
    m = ModelRatesNorm.shape[0]

    if m < n:
        raise ValueError(f"ModelRatesNorm is shorter ({m}) than neural_data ({n}).")

    # end-align ModelRates to neural_data (keep the last n rows)
    ModelRates = ModelRatesNorm[m - n:, :]

    # Boolean mask: True for columns that are all zeros
    zero_cols_mask = np.all(ExpRatesNorm == 0, axis=0)
    zero_rows_mask = np.all(ExpRatesNorm == 0, axis=1)

    # Indices of zero-columns
    zero_col_indices = np.where(zero_cols_mask)[0]
    print("Zero columns at indices:", zero_col_indices)
    # Print the indices of zero-rows
    zero_row_indices = np.where(zero_rows_mask)[0]
    print("Zero rows at indices:", zero_row_indices)

    # Print the actual columns (as a submatrix)
    print("Zero columns:", np.shape(ExpRatesNorm[:, zero_cols_mask]))
    print("Zero rows:", np.shape(ExpRatesNorm[zero_rows_mask, :]))

    # Print the actual rows
    print("Length zero rows/columns:", len(zero_row_indices),  len(zero_col_indices))

    import numpy as np
    print("Shape of the Model and Experimental firing activity matrices")
    print(np.shape(ModelRates), np.shape(cfg.neural_data))

    plt.figure()
    plt.imshow(ModelRates.T, aspect='auto', cmap='viridis')
    plt.colorbar()
    filename = cfg.saveFolder + "/" + cfg.simLabel + f'_ModelRates.png'
    plt.savefig(filename)
    plt.close()

    plt.figure()
    plt.imshow(cfg.neural_data.T, aspect='auto', cmap='viridis')
    plt.colorbar()
    filename = cfg.saveFolder + "/" + cfg.simLabel + f'_ExpRates.png'
    plt.savefig(filename)
    plt.close()

    ####
    # Load CEBRA MODEL
    ###
    import cebra
    model_label = 'stg-joy' # 'joy' 'stg' 'stg-joy' 'time' 
    file_path = Path('src_CEBRA/CEBRA_data/230517_2759_1606VAL/decode_rs_iter100000_ndim3_conddelta/models/') / f'model_{model_label}.pt'
    model = cebra.CEBRA.load(file_path)
    print(f"Loaded model from {file_path}")

    from matplotlib.colors import LinearSegmentedColormap
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] # R -> G -> B
    if len(np.unique(cfg.stageId)) == 4:
        colors += [(1, 1, 0)] # R -> G -> B -> Y
    cmap_name = 'cebra'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=len(colors))

    labels = ['baseline', 'tone_on', 'after tone']
    values = [0, 1, 2]  # numeric codes in aux[:, 0]
    numplots = 1 #len(neural_data)
    fig, axs = plt.subplots(1, numplots, figsize=(5 * numplots, 6), subplot_kw={'projection': '3d'})

    embedding_exp = model.transform(cfg.neural_data)
    ax1 = cebra.plot_embedding(embedding_exp, embedding_labels=cfg.stageId, ax=axs, title="Embedding", cmap=cmap, alpha=1.0)
    filename = cfg.saveFolder + "/" + cfg.simLabel + f'_cebra_embedding_exp_{model_label}.png'
    fig = ax1.get_figure()
    fig.savefig(filename)
    ax1.clear()
    plt.close(fig)

    embedding_model = model.transform(ModelRates)
    ax2 = cebra.plot_embedding(embedding_model, embedding_labels=cfg.stageId, ax=axs, title="Embedding", cmap=cmap, alpha=1.0)
    filename = cfg.saveFolder + "/" + cfg.simLabel + f'_cebra_embedding_model_{model_label}.png'
    fig = ax2.get_figure()
    fig.savefig(filename)
    ax2.clear()
    plt.close(fig)

    r_2 = defs.linear_fit(embedding_exp, embedding_model) # r_2 goes from -inf to 1. We need to change it to be a loss function.
    loss = 1 - r_2

    results = {}
    results['loss'] = 1 - r_2
    out_json = json.dumps({**inputs, **results})

    print(out_json)
    sim.send(out_json)

# import cebra
# model_label = 'joy' # 'joy' 'stg' 'stg-joy' 'time' 
# file_path = Path('src_CEBRA/CEBRA_data/230517_2759_1606VAL/decode_rs_iter100000_ndim3_conddelta/models/') / f'model_{model_label}.pt'
# model = cebra.CEBRA.load(file_path)
# print(f"Loaded model from {file_path}")