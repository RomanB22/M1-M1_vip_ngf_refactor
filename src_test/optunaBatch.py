from batchtk.algos import optuna_search
from batchtk.utils import expand_path

from netpyne.batchtools.search import generate_constructors
from ClusterConfigs import slurm_args
from batch_params import get_batch_params

from pathlib import Path
cwd = str(Path.cwd())

#option for local run
# dispatcher, submit = generate_constructors('sh', 'socket')
#option for slurm run
dispatcher, submit = generate_constructors('slurm', 'sfs')

num_individuals = 1
num_iterations = 500

PercentageChange = 0.5
minChg = (1-PercentageChange)
maxChg = (1+PercentageChange)

params = get_batch_params(minChg, maxChg)

results = optuna_search(
    study_label='optuna_batch',
    param_space=params,
    metrics={'loss': 'minimize'},
    num_trials=num_iterations*num_individuals,
    num_workers=num_individuals,
    dispatcher_constructor=dispatcher,
    submit_constructor=submit,
    # submit_kwargs={'command': 'python -u src/init.py'}, # normal run
    submit_kwargs=slurm_args,
    interval=10,
    project_path=cwd,
    output_path=expand_path('./optimization/optuna', create_dirs=True),
)
