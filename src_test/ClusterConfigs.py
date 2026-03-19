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
export PYTHONPATH=$PYTHONPATH:$PWD/src_test

time mpirun -n $SLURM_NTASKS nrniv -mpi -python src_test/init.py
"""

slurm_args = {
    'allocation': 'TG-IBN140002', # 'TG-IBN140002' 'TG-MED240058' 'TG-MED240050'
    'realtime': '10:30:00',
    'nodes': '1',
    'coresPerNode': '120',
    'mem': '240G',
    'partition': 'compute',
    'email': 'romanbaravalle@gmail.com',
    'custom': '',
    'command': CONFIG_EXPANSE_CPU,  # ← FIXED: remove braces
}

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

CONFIG_EXPANSE_SGE = f"""
# >>> Conda setup
source ~/.bashrc
conda activate M1_dev
# <<< End Conda setup

export LD_LIBRARY_PATH=$HOME/miniconda3/envs/M1_dev/lib/python3.10/site-packages/mpi4py_mpich.libs

# Add project root and src to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$PWD"
export PYTHONPATH="$PYTHONPATH:$PWD/src_test"

mpiexec -n $NSLOTS -hosts $(hostname) nrniv -python -mpi src_test/init.py
"""

from batchtk.utils import expand_path
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
output_root = project_root / 'optimization'
results_path = output_root / 'cmaes' / 'cmaes_results_Feb25.txt'

sge_config = {
    'queue': 'cpu.q',
    'cores': 50,
    'vmem': '120G',
    'realtime': '15:00:00',
    'command': CONFIG_EXPANSE_SGE,
    'project_path': project_root,
}