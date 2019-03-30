#!/bin/bash -l
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=trial_3
#SBATCH --account=m2871
#SBATCH --output=./test.not_parallel%j.txt
#SBATCH --constraint=haswell

srun python3 parallel.py	
