from batchtk.algos import optuna_search
from batchtk.utils import expand_path

from netpyne.batchtools.search import generate_constructors

#option for local run
dispatcher, submit = generate_constructors('sh', 'socket')

#option for slurm run
# dispatcher, submit = generate_constructors('slurm', 'sfs')
slurm_args = {
    'allocation': 'csd403',
    'realtime': '00:30:00',
    'nodes': '1',
    'coresPerNode': '1',
    'mem': '4G',
    'partition': 'shared',
    'email': '<user_email_here>',
    'custom': '',
    'command': 'python src/init.py',
}


numSamples = 1
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


# ENV_VARS = """
# export STRRUNTK16="saveFolder*=${_batchtk_path_pointer}"
# export STRRUNTK17="simLabel*=${_batchtk_label_pointer}"
# """

num_individuals = 2
num_iterations = 2

results = optuna_search(
    study_label='optuna_batch',
    param_space=params,
    metrics={'loss': 'minimize'},
    num_trials=num_iterations*num_individuals,
    num_workers=num_individuals,
    dispatcher_constructor=dispatcher,
    submit_constructor=submit,
    submit_kwargs={'command': 'python -u src/init.py'}, # normal run
    # submit_kwargs=slurm_args,
    interval=10,
    project_path='.',
    output_path=expand_path('./optimization/optuna', create_dirs=True),
)