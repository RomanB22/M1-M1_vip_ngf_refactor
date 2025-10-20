#!/bin/bash 
#SBATCH --job-name=m1_cell_7.5
#SBATCH -A csd403
#SBATCH -t 00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -o /home/salvadord/m1/sim/m1cell.run
#SBATCH -e /home/salvadord/m1/sim/m1cell.err
#SBATCH --mail-user=salvadordura@gmail.com
#SBATCH --mail-type=end


source ~/.bashrc_7.5
cd /home/salvadord/m1/sim/
ibrun -np 4 nrniv -python -mpi init_cell.py 
wait
                            