#!/bin/bash -l
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J rnn_converge
#SBATCH -o ./logs/rnn_converges.txt
#SBATCH --constraint=haswell
srun -N 1 python3 rnn_experiments.py --model rnn --default 0 --dataset gfp_amino_acid_ --name rnn_converge --layers 2 --hidden_size 200 --learning_rate 0.005 --epochs 500 --num_data 100 --batch_size 10 
