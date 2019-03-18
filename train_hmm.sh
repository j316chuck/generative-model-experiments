#!bin/bash


git clone https://github.com/j316chuck/generative-model-experiments.git
cd generative-model-experiments
mkdir models

python3 hmm_experiments.py --n_jobs 10 --hidden_size 100 --max_iterations 100 --num_sequences 100 --name "hmm_amino_acid_hidden_100"
python3 hmm_experiments.py --n_jobs 10 --hidden_size 200 --max_iterations 1000 --num_sequences 100 --name "hmm_amino_acid_hidden_200"
python3 hmm_experiments.py --n_jobs 10 --hidden_size 200 --max_iterations 1e8 --num_sequences 100 --name "hmm_amino_acid_hidden_200_iter_inf"
python3 hmm_experiments.py --n_jobs 10 --hidden_size 500 --max_iterations 1000 --num_sequences 100 --name "hmm_amino_acid_hidden_500"
python3 hmm_experiments.py --n_jobs 10 --hidden_size 200 --max_iterations 1000 --num_sequences 10000 --name "hmm_amino_acid_hidden_200_seq_10000"

