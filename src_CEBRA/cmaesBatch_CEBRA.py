from batchtk.utils import expand_path
from batchtk.algos import cmaes_search

from netpyne.batchtools.search import generate_constructors
import pandas as pd
from ClusterConfigs_CEBRA import slurm_args

from pathlib import Path
cwd = str(Path.cwd())

#option for local run
# dispatcher, submit = generate_constructors('sh', 'socket')
#option for slurm run
dispatcher, submit = generate_constructors('slurm', 'sfs')

num_individuals = 20
num_generations = 50

PercentageChange = 0.2
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

param_space_samplers = ['float' for _ in range(len(params))]  # specify float sampling for all parameters
results = cmaes_search(
    study_label='cmaes_batch_umap',
    param_space=params,
    param_space_samplers=param_space_samplers,  # specify integer sampling for both parameters
    algo_kwargs={'seed': 42}, # for reproducibility
    metrics={'loss': 'minimize'},
    num_trials=num_generations*num_individuals, # num_generations = int(numpy.ceil(num_trials / sampler.population_size))
    num_workers=num_individuals, # Number of individuals per generation
    dispatcher_constructor=dispatcher,
    submit_constructor=submit,
    # submit_kwargs={'command': 'python -u src/init.py'}, # normal run
    submit_kwargs=slurm_args,
    interval=10,
    project_path=cwd,
    output_path=expand_path('./optimization/cmaes', create_dirs=True),
)

with open('./optimization/cmaes/cmaes_results.txt', 'w') as f:
    f.write(str(results))

print(results)