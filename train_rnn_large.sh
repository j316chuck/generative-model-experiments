#!/bin/bash -l
#SBATCH -q regular
#SBATCH -N 5
#SBATCH -t 20:00:00
#SBATCH -J rnn_trial
#SBATCH -o ./logs/rnn_trial.txt
#SBATCH --constraint=haswell 

srun -N 1 python3 rnn_experiments.py --model rnn --default 0 --dataset gfp_amino_acid_ --name rnn_trial_1 --layers 2 --hidden_size 200 --learning_rate 0.005 --epochs 1000 --num_data 120 --batch_size 10 &
srun -N 1 python3 rnn_experiments.py --model rnn --default 0 --dataset gfp_amino_acid_ --name rnn_trial_2 --layers 2 --hidden_size 200 --learning_rate 0.005 --epochs 100 --num_data 46733 --batch_size 100 &
srun -N 1 python3 rnn_experiments.py --model rnn --default 0 --dataset gfp_amino_acid_ --name rnn_trial_3 --layers 4 --hidden_size 400 --learning_rate 0.005 --epochs 100 --num_data 120 --batch_size 10 &
srun -N 1 python3 rnn_experiments.py --model rnn --default 0 --dataset gfp_amino_acid_ --name rnn_trial_4 --layers 4 --hidden_size 400 --learning_rate 0.0001 --epochs 250 --num_data 120 --batch_size 32 &
srun -N 1 python3 rnn_experiments.py --model rnn --default 0 --dataset gfp_amino_acid_ --name rnn_trial_5 --layers 4 --hidden_size 400 --learning_rate 0.0001 --epochs 250 --num_data 46733 --batch_size 128 &
wait
