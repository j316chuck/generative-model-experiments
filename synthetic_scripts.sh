# hmm_default_small
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_100_gaussian/hmm --name hmm_default_small --input 2000 --hidden_size 30 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_gaussian --num_data 100 

# rnn_default_small
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_100_gaussian/rnn --name rnn_default_small --input 2000 --hidden_size 100 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_gaussian --num_data 100 

# vae_default_small
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_100_gaussian/vae --name vae_default_small --input 2000 --hidden_size 50 --latent_dim 20 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_gaussian --num_data 100 

# hmm_default_small
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_100_skewed_gaussian/hmm --name hmm_default_small --input 2000 --hidden_size 30 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_skewed_gaussian --num_data 100 

# rnn_default_small
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_100_skewed_gaussian/rnn --name rnn_default_small --input 2000 --hidden_size 100 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_skewed_gaussian --num_data 100 

# vae_default_small
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_100_skewed_gaussian/vae --name vae_default_small --input 2000 --hidden_size 50 --latent_dim 20 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_skewed_gaussian --num_data 100 

# hmm_default_small
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_100_uniform/hmm --name hmm_default_small --input 2000 --hidden_size 30 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_uniform --num_data 100 

# rnn_default_small
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_100_uniform/rnn --name rnn_default_small --input 2000 --hidden_size 100 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_uniform --num_data 100 

# vae_default_small
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_100_uniform/vae --name vae_default_small --input 2000 --hidden_size 50 --latent_dim 20 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_uniform --num_data 100 

# hmm_default_small
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_20_gaussian/hmm --name hmm_default_small --input 400 --hidden_size 30 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_gaussian --num_data 100 

# rnn_default_small
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_20_gaussian/rnn --name rnn_default_small --input 400 --hidden_size 100 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_gaussian --num_data 100 

# vae_default_small
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_20_gaussian/vae --name vae_default_small --input 400 --hidden_size 50 --latent_dim 20 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_gaussian --num_data 100 

# hmm_default_small
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_20_skewed_gaussian/hmm --name hmm_default_small --input 400 --hidden_size 30 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_skewed_gaussian --num_data 100 

# rnn_default_small
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_20_skewed_gaussian/rnn --name rnn_default_small --input 400 --hidden_size 100 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_skewed_gaussian --num_data 100 

# vae_default_small
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_20_skewed_gaussian/vae --name vae_default_small --input 400 --hidden_size 50 --latent_dim 20 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_skewed_gaussian --num_data 100 

# hmm_default_small
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_20_uniform/hmm --name hmm_default_small --input 400 --hidden_size 30 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_uniform --num_data 100 

# rnn_default_small
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_20_uniform/rnn --name rnn_default_small --input 400 --hidden_size 100 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_uniform --num_data 100 

# vae_default_small
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_20_uniform/vae --name vae_default_small --input 400 --hidden_size 50 --latent_dim 20 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_uniform --num_data 100 

# hmm_default_small
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_50_gaussian/hmm --name hmm_default_small --input 1000 --hidden_size 30 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_gaussian --num_data 100 

# rnn_default_small
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_50_gaussian/rnn --name rnn_default_small --input 1000 --hidden_size 100 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_gaussian --num_data 100 

# vae_default_small
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_50_gaussian/vae --name vae_default_small --input 1000 --hidden_size 50 --latent_dim 20 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_gaussian --num_data 100 

# hmm_default_small
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_50_skewed_gaussian/hmm --name hmm_default_small --input 1000 --hidden_size 30 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_skewed_gaussian --num_data 100 

# rnn_default_small
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_50_skewed_gaussian/rnn --name rnn_default_small --input 1000 --hidden_size 100 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_skewed_gaussian --num_data 100 

# vae_default_small
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_50_skewed_gaussian/vae --name vae_default_small --input 1000 --hidden_size 50 --latent_dim 20 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_skewed_gaussian --num_data 100 

# hmm_default_small
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_50_uniform/hmm --name hmm_default_small --input 1000 --hidden_size 30 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_uniform --num_data 100 

# rnn_default_small
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_50_uniform/rnn --name rnn_default_small --input 1000 --hidden_size 100 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_uniform --num_data 100 

# vae_default_small
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_50_uniform/vae --name vae_default_small --input 1000 --hidden_size 50 --latent_dim 20 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_uniform --num_data 100

# hmm_default_medium
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_100_gaussian/hmm --name hmm_default_medium --input 2000 --hidden_size 100 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_gaussian --num_data 1000 

# rnn_default_medium
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_100_gaussian/rnn --name rnn_default_medium --input 2000 --hidden_size 200 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_gaussian --num_data 1000 

# vae_default_medium
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_100_gaussian/vae --name vae_default_medium --input 2000 --hidden_size 200 --latent_dim 20 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_gaussian --num_data 1000 

# hmm_default_medium
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_100_skewed_gaussian/hmm --name hmm_default_medium --input 2000 --hidden_size 100 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_skewed_gaussian --num_data 1000 

# rnn_default_medium
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_100_skewed_gaussian/rnn --name rnn_default_medium --input 2000 --hidden_size 200 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_skewed_gaussian --num_data 1000 

# vae_default_medium
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_100_skewed_gaussian/vae --name vae_default_medium --input 2000 --hidden_size 200 --latent_dim 20 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_skewed_gaussian --num_data 1000 

# hmm_default_medium
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_100_uniform/hmm --name hmm_default_medium --input 2000 --hidden_size 100 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_uniform --num_data 1000 

# rnn_default_medium
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_100_uniform/rnn --name rnn_default_medium --input 2000 --hidden_size 200 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_uniform --num_data 1000 

# vae_default_medium
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_100_uniform/vae --name vae_default_medium --input 2000 --hidden_size 200 --latent_dim 20 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_uniform --num_data 1000 

# hmm_default_medium
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_20_gaussian/hmm --name hmm_default_medium --input 400 --hidden_size 100 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_gaussian --num_data 1000 

# rnn_default_medium
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_20_gaussian/rnn --name rnn_default_medium --input 400 --hidden_size 200 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_gaussian --num_data 1000 

# vae_default_medium
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_20_gaussian/vae --name vae_default_medium --input 400 --hidden_size 200 --latent_dim 20 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_gaussian --num_data 1000 

# hmm_default_medium
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_20_skewed_gaussian/hmm --name hmm_default_medium --input 400 --hidden_size 100 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_skewed_gaussian --num_data 1000 

# rnn_default_medium
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_20_skewed_gaussian/rnn --name rnn_default_medium --input 400 --hidden_size 200 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_skewed_gaussian --num_data 1000 

# vae_default_medium
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_20_skewed_gaussian/vae --name vae_default_medium --input 400 --hidden_size 200 --latent_dim 20 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_skewed_gaussian --num_data 1000 

# hmm_default_medium
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_20_uniform/hmm --name hmm_default_medium --input 400 --hidden_size 100 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_uniform --num_data 1000 

# rnn_default_medium
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_20_uniform/rnn --name rnn_default_medium --input 400 --hidden_size 200 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_uniform --num_data 1000 

# vae_default_medium
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_20_uniform/vae --name vae_default_medium --input 400 --hidden_size 200 --latent_dim 20 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_uniform --num_data 1000 

# hmm_default_medium
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_50_gaussian/hmm --name hmm_default_medium --input 1000 --hidden_size 100 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_gaussian --num_data 1000 

# rnn_default_medium
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_50_gaussian/rnn --name rnn_default_medium --input 1000 --hidden_size 200 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_gaussian --num_data 1000 

# vae_default_medium
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_50_gaussian/vae --name vae_default_medium --input 1000 --hidden_size 200 --latent_dim 20 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_gaussian --num_data 1000 

# hmm_default_medium
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_50_skewed_gaussian/hmm --name hmm_default_medium --input 1000 --hidden_size 100 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_skewed_gaussian --num_data 1000 

# rnn_default_medium
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_50_skewed_gaussian/rnn --name rnn_default_medium --input 1000 --hidden_size 200 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_skewed_gaussian --num_data 1000 

# vae_default_medium
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_50_skewed_gaussian/vae --name vae_default_medium --input 1000 --hidden_size 200 --latent_dim 20 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_skewed_gaussian --num_data 1000 

# hmm_default_medium
python3 run_model.py --model_type hmm --base_log logs/synthetic_unimodal_data_length_50_uniform/hmm --name hmm_default_medium --input 1000 --hidden_size 100 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_uniform --num_data 1000 

# rnn_default_medium
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_50_uniform/rnn --name rnn_default_medium --input 1000 --hidden_size 200 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_uniform --num_data 1000 

# vae_default_medium
python3 run_model.py --model_type vae --base_log logs/synthetic_unimodal_data_length_50_uniform/vae --name vae_default_medium --input 1000 --hidden_size 200 --latent_dim 20 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_uniform --num_data 1000 
 
