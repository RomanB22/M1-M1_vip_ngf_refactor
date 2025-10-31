CONFIG_EXPANSE_CPU = """
# >>> Conda setup
source ~/.bashrc
module purge
conda activate NewBatchtk
# <<< End Conda setup

# Load modules
echo "Loading modules..."
module load openmpi/mlnx/gcc/64/4.1.5a1
         
# Add project root and src to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PWD
export PYTHONPATH=$PYTHONPATH:$PWD/src

time mpirun -n $SLURM_NTASKS nrniv -mpi -python src/init.py
"""

slurm_args = {
    'allocation': 'TG-MED240058', # 'TG-IBN140002' 'TG-MED240058' 'TG-MED240050'
    'realtime': '10:30:00',
    'nodes': '1',
    'coresPerNode': '120',
    'mem': '240G',
    'partition': 'compute',
    'email': 'romanbaravalle@gmail.com',
    'custom': '',
    'command': CONFIG_EXPANSE_CPU,  # ← FIXED: remove braces
}
