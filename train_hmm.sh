#!/bin/bash -l

#SBATCH --qos=debug
#SBATCH --nodes=4
#SBATCH --job-name=hmm_debug_trial
#SBATCH --time=00:10:00
#SBATCH --output=./logs/%j.txt
#SBATCH --account=m2871

srun -n 10  python3 hmm_experiments.py --n_jobs 10 --hidden_size 10 --max_iterations 100 --num_sequences 100 --name "hmm_amino_acid_hidden_10" &
srun -n 10 python3 hmm_experiments.py --n_jobs 10 --hidden_size 20 --max_iterations 1000 --num_sequences 100 --name "hmm_amino_acid_hidden_20" & 
srun -n 10 python3 hmm_experiments.py --n_jobs 10 --hidden_size 20 --max_iterations 1e8 --num_sequences 100 --name "hmm_amino_acid_hidden_20_iter_inf" & 
srun -n 10 python3 hmm_experiments.py --n_jobs 10 --hidden_size 50 --max_iterations 1000 --num_sequences 100 --name "hmm_amino_acid_hidden_50" & 
wait

