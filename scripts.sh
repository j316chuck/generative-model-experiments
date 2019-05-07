# hmm default medium
python3 run_experiment.py --model_type hmm --base_log logs/gfp/hmm/ --name hmm_default_medium --input 4998 --hidden_size 100 --latent_dim -1 --seq_length 238 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset gfp --num_data 1000 

# rnn default medium
python3 run_experiment.py --model_type rnn --base_log logs/gfp/rnn/ --name rnn_default_medium --input 4998 --hidden_size 200 --latent_dim -1 --seq_length 238 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset gfp --num_data 1000 

# vae default medium
python3 run_experiment.py --model_type vae --base_log logs/gfp/vae/ --name vae_default_medium --input 4998 --hidden_size 200 --latent_dim 20 --seq_length 238 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset gfp --num_data 1000

# hmm default small
python3 run_experiment.py --model_type hmm --base_log logs/gfp/hmm/ --name hmm_default_small --input 4998 --hidden_size 30 --latent_dim -1 --seq_length 238 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset gfp --num_data 100 

# rnn default small
python3 run_experiment.py --model_type rnn --base_log logs/gfp/rnn/ --name rnn_default_small --input 4998 --hidden_size 100 --latent_dim -1 --seq_length 238 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset gfp --num_data 100 

# vae default small
python3 run_experiment.py --model_type vae --base_log logs/gfp/vae/ --name vae_default_small --input 4998 --hidden_size 50 --latent_dim 20 --seq_length 238 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset gfp --num_data 100  
