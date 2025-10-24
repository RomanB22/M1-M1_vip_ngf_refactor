from batchtk.utils import expand_path
from batchtk.algos import cmaes_search

from netpyne.batchtools.search import generate_constructors
from ClusterConfigs import slurm_args

#option for local run
# dispatcher, submit = generate_constructors('sh', 'socket')
#option for slurm run
dispatcher, submit = generate_constructors('slurm', 'sfs')

num_individuals = 6
num_generations = 2

PercentageChange = 0.5
minChg = (1-PercentageChange)
maxChg = (1+PercentageChange)

params = {'weightLong.TPO': (0.1*minChg, 0.5*maxChg),
          'weightLong.TVL': (0.1*minChg, 0.5*maxChg),
          'weightLong.S1': (0.1*minChg, 0.5*maxChg),
          'weightLong.S2': (0.1*minChg, 0.5*maxChg),
          'weightLong.cM1': (0.1*minChg, 0.5*maxChg),
          'weightLong.M2': (0.1*minChg, 0.5*maxChg),
          'weightLong.OC': (0.1*minChg, 0.5*maxChg),
          'EEGain': (1.*minChg, 1.*maxChg),
          'IEweights.0': (1.*minChg, 1.*maxChg),    ## L2/3+4
          'IEweights.1': (1.*minChg, 1.*maxChg),    ## L5
          'IEweights.2': (1.*minChg, 1.*maxChg),    ## L6
          'IIweights.0': (1.*minChg, 1.*maxChg),    ## L2/3+4
          'IIweights.1': (1.*minChg, 1.*maxChg),    ## L5
          'IIweights.2': (1.*minChg, 1.*maxChg),    ## L6
        #   'EICellTypeGain.PV': (1.*minChg, 4.*maxChg),    
        #   'EICellTypeGain.SOM': (1.*minChg, 4.*maxChg),    
        #   'EICellTypeGain.VIP': (1.*minChg, 4.*maxChg),    
        #   'EICellTypeGain.NGF': (1.*minChg, 4.*maxChg),
        #   'scaleDensity': (0.15)   
          }



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
    submit_kwargs=slurm_args,
    interval=10,
    project_path='.',
    output_path=expand_path('./optimization/cmaes', create_dirs=True),
)

with open('./optimization/cmaes/cmaes_results.txt', 'w') as f:
    f.write(str(results))

print(results)