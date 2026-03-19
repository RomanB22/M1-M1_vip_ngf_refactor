from batchtk.utils import expand_path
from batchtk.algos import cmaes_search

from netpyne.batchtools.search import generate_constructors
from netpyne.batchtools.submits import SGESubmitSFS
from ClusterConfigs import sge_config
from batch_params import get_batch_params

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
output_root = project_root / 'optimization'
results_path = output_root / 'cmaes' / 'cmaes_results_Feb25.txt'
submit_root = output_root / '.batchtk_jobs'
submit_root.mkdir(parents=True, exist_ok=True)

#option for local run
# dispatcher, submit = generate_constructors('sh', 'socket')
#option for slurm run
# dispatcher, submit = generate_constructors('slurm', 'sfs')
#option for sge run
dispatcher, submit = generate_constructors('sge', 'sfs')


class FixedSGESubmitSFS(SGESubmitSFS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        script_path = submit_root / '{label}.sh'
        self.submit_template.template = f'qsub {script_path}'
        self.path_template.template = str(script_path)


submit = FixedSGESubmitSFS

num_individuals = 10
num_generations = 50

PercentageChange = 0.5
minChg = (1-PercentageChange)
maxChg = (1+PercentageChange)

params = get_batch_params(minChg, maxChg)

param_space_samplers = ['float' for _ in range(len(params))]  # specify float sampling for all parameters
results = cmaes_search(
    study_label='cmaes_batch',
    param_space=params,
    param_space_samplers=param_space_samplers,  # specify integer sampling for both parameters
    algo_kwargs={'seed': 42}, # for reproducibility
    metrics={'loss': 'minimize'},
    num_trials=num_generations*num_individuals, # num_generations = int(numpy.ceil(num_trials / sampler.population_size))
    num_workers=num_individuals, # Number of individuals per generation
    dispatcher_constructor=dispatcher,
    submit_constructor=submit,
    # submit_kwargs={'command': 'python -u src/init.py'}, # normal run
    submit_kwargs=sge_config,
    interval=10,
    project_path=str(project_root),
    output_path=expand_path(str(output_root), create_dirs=True),
)

results_path.parent.mkdir(parents=True, exist_ok=True)
with open(results_path, 'w') as f:
    f.write(str(results))

print(results)
