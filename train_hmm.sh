#!/bin/bash -l

#SBATCH --qos=debug
#SBATCH --nodes=4
#SBATCH --job-name=hmm_sample
#SBATCH --time=00:10:00
#SBATCH --output=./logs/hmm_sample.txt
#SBATCH --account=m2871

srun -N 1 -n 1 python3 hmm_experiments.py --n_jobs 1 --hidden_size 25 --max_iterations 100 --num_sequences 100 --length 25 --name "hmm_sample_1" &
srun -N 1 -n 24 python3 hmm_experiments.py --n_jobs 1 --hidden_size 25 --max_iterations 100 --num_sequences 100 --length 25 --name "hmm_sample_2" & 
srun -N 1 -c 24 python3 hmm_experiments.py --n_jobs 24 --hidden_size 25 --max_iterations 100 --num_sequences 100 --name "hmm_sample_3" & 
srun -N 1 python3 hmm_experiments.py --n_jobs 24 --hidden_size 25 --max_iterations 1000 --num_sequences 100 --name "hmm_sample_4" & 
wait

