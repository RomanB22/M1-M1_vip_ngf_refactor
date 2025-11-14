from batchtk.algos import optuna_search
from batchtk.utils import expand_path

from netpyne.batchtools.search import generate_constructors
from ClusterConfigs_CEBRA import slurm_args

import pandas as pd

from pathlib import Path
cwd = str(Path.cwd())

#option for local run
# dispatcher, submit = generate_constructors('sh', 'socket')
#option for slurm run
dispatcher, submit = generate_constructors('slurm', 'sfs')

num_individuals = 1
num_iterations = 5000
numTrials = int(num_individuals*num_iterations)

PercentageChange = 0.5
minChg = (1-PercentageChange)
maxChg = (1+PercentageChange)

dataFrame = pd.read_csv('./UMAP_manifold/BaselineModels.csv') 
include = ['weightLong.TPO', 'weightLong.TVL', 'weightLong.S1',
       'weightLong.S2', 'weightLong.cM1', 'weightLong.M2', 'weightLong.OC',
       'EEGain', 'IEweights.0', 'IEweights.1', 'IEweights.2', 'IIweights.0',
       'IIweights.1', 'IIweights.2', 'EICellTypeGain.PV', 'EICellTypeGain.SOM',
       'EICellTypeGain.VIP', 'EICellTypeGain.NGF']

chosenTrial = 0

row = dataFrame[include].iloc[chosenTrial]
params = {}

params = {
    col: [minChg * row[col], maxChg * row[col]]
    for col in include
}

params['dt'] = [0.1, 0.1]
params['recordStep'] = [0.1, 0.1]

results = optuna_search(
    study_label='optuna_batch',
    param_space=params,
    metrics={'loss': 'minimize'},
    num_trials=numTrials,
    num_workers=num_individuals,
    dispatcher_constructor=dispatcher,
    submit_constructor=submit,
    # submit_kwargs={'command': 'python -u src/init.py'}, # normal run
    submit_kwargs=slurm_args,
    interval=10,
    project_path=cwd,
    output_path=expand_path('./optimization/optuna_CEBRA', create_dirs=True),
)