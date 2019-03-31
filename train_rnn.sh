#!/bin/bash -l
#SBATCH -q debug
#SBATCH -N 6
#SBATCH -t 0:30:00
#SBATCH -J rnn_sample
#SBATCH -o ./logs/rnn_sample.txt
#SBATCH --constraint=haswell 

srun -N 1 python3 rnn_experiments.py --model rnn --default 0 --dataset gfp_amino_acid_ --name rnn_test_sample_1 --layers 2 --hidden_size 200 --learning_rate 0.005 --epochs 10 --num_data 120 --batch_size 10 &
srun -N 1 python3 rnn_experiments.py --model rnn --default 0 --dataset gfp_amino_acid_ --name rnn_test_sample_2 --layers 8 --hidden_size 200 --learning_rate 0.005 --epochs 10 --num_data 120 --batch_size 10 &
srun -N 1 python3 rnn_experiments.py --model rnn --default 1 --dataset gfp_amino_acid_ --name rnn_test_sample_3 --layers 2 --hidden_size 200 --learning_rate 0.005 --epochs 10 --num_data 120 --batch_size 10 &
srun -N 1 python3 rnn_experiments.py --model rnn --default 0 --dataset gfp_amino_acid_ --name rnn_test_sample_4 --layers 2 --hidden_size 200 --learning_rate 0.005 --epochs 10 --num_data 120 --batch_size 40 &
srun -N 1 python3 rnn_experiments.py --model rnn --default 0 --dataset gfp_amino_acid_ --name rnn_test_sample_5 --layers 2 --hidden_size 800 --learning_rate 0.005 --epochs 10 --num_data 120 --batch_size 10 &
srun -N 1 python3 rnn_experiments.py --model rnn --default 0 --dataset gfp_amino_acid_ --name rnn_test_sample_6 --layers 1 --hidden_size 100 --learning_rate 0.005 --epochs 10 --num_data 120 --batch_size 10 & 
wait
