#!/bin/bash -l

#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --job-name=hmm_trial
#SBATCH --time=02:00:00
#SBATCH --output=./logs/hmm_trial.txt
#SBATCH --account=m2871
#SBATCH --constraint=haswell

srun -N 1 python3 hmm_experiments.py --n_jobs 24 --hidden_size 200 --max_iterations 1000 --num_sequences 100 --length 238 --name "hmm_trial_1" &
srun -N 1 python3 hmm_experiments.py --n_jobs 24 --hidden_size 500 --max_iterations 1000 --num_sequences 100 --length 238 --name "hmm_trial_2" & 
srun -N 1 python3 hmm_experiments.py --n_jobs 24 --hidden_size 200 --max_iterations 1e8 --num_sequences 1000 --length 238 --name "hmm_trial_3" & 
srun -N 1 python3 hmm_experiments.py --n_jobs 24 --hidden_size 500 --max_iterations 1e8 --num_sequences 1000 --length 238 --name "hmm_trial_4" & 
wait

