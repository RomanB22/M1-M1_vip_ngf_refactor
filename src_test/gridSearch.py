from netpyne.batchtools.search import search
from solutionsChannelopathies import *
# -------------------------------------------------------------------------
# Batch parameters – grid search over simConfig.seeds sub-keys
# -------------------------------------------------------------------------
for name, chosen_params in {
    'params175': params175,
    'params269': params269,
    'params215': params215,
    'params207': params207,
}.items():
    params = {k: [v] for k, v in chosen_params.items()} | {
        'seeds.conn': [4321 + (17*i) for i in range(5)],
        'seeds.stim': [4321 + (17*i) for i in range(5)]
    }
    # -------------------------------------------------------------------------
    # Run the grid search (25 combinations: 5 x 5)
    # -------------------------------------------------------------------------
    search(
        job_type='suny',
        comm_type='sfs',
        params=params,
        run_config={
            'command': 'conda activate CompNeuroCourse \nexport PYTHONPATH=$PWD:$PWD/src_test  \nexport UCX_TLS=tcp,self \nexport LD_LIBRARY_PATH=~/miniconda3/envs/CompNeuroCourse/lib \nsrun --mpi=pmi2 nrniv -python -mpi python -u src_test/init.py',
            'cores': 52,
            'mem': '200G',
            'realtime': '10:30:00'
        },
        label=f'gridSearch_seeds_{name}',
        output_path='./batch_results',
        checkpoint_path='./batch_checkpoints',
    )