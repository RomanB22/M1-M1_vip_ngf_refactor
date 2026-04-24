from netpyne.batchtools.search import search

# -------------------------------------------------------------------------
# Batch parameters – grid search over simConfig.seeds sub-keys
# -------------------------------------------------------------------------

params = {
    'seeds.conn': [4321+(17*i) for i in range(5)],
    'seeds.stim': [4321+(17*i) for i in range(5)],
    'saveJson': ['True'],
}

# -------------------------------------------------------------------------
# Run the grid search (25 combinations: 5 x 5)
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