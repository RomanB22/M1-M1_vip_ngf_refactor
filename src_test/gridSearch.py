from netpyne.batchtools.search import search

# -------------------------------------------------------------------------
# Batch parameters – grid search over simConfig.seeds sub-keys
# -------------------------------------------------------------------------
params = {
    'seeds.conn': [1, 2, 3],
    'seeds.stim': [1, 2, 3],
    'seeds.loc':  [1, 2, 3],
}

# -------------------------------------------------------------------------
# Run the grid search (27 combinations: 3 x 3 x 3)
# -------------------------------------------------------------------------
search(
    job_type='suny',      # change to 'hpc_slurm' / 'sge' for cluster
    comm_type = 'sfs', # 'socket', 'sfs', None
    params=params,
    run_config={
        'command': 'conda activate CompNeuroCourse \nexport UCX_TLS=tcp,self \nexport LD_LIBRARY_PATH=~/miniconda3/envs/CompNeuroCourse/lib\nsrun --mpi=pmi2 nrniv -python -mpi python -u src/init.py',
        'cores': 52,
        'mem': '200G',
        'realtime': '10:30:00'
    },
    label='gridSearch_seeds',
    output_path='./batch_results',
    checkpoint_path='./batch_checkpoints',
)