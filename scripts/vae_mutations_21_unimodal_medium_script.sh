python3 run_model.py --model_type vae --base_log logs/synthetic_mutations_21_unimodal_data_length_50_uniform/vae --name vae_default_medium --input 1000 --hidden_size 200 --latent_dim 20 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_mutations_21_unimodal_data_length_50_uniform --num_data 1000 --early_stopping True --patience 10 & 
python3 run_model.py --model_type vae --base_log logs/synthetic_mutations_21_unimodal_data_length_100_uniform/vae --name vae_default_medium --input 2000 --hidden_size 200 --latent_dim 20 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_mutations_21_unimodal_data_length_100_uniform --num_data 1000 --early_stopping True --patience 10 &&

python3 run_model.py --model_type rnn --base_log logs/synthetic_mutations_21_unimodal_data_length_50_uniform/rnn --name rnn_default_medium --input 1000 --hidden_size 50 --latent_dim 20 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_mutations_21_unimodal_data_length_50_uniform --num_data 1000 --early_stopping True --patience 10 &
python3 run_model.py --model_type rnn --base_log logs/synthetic_mutations_21_unimodal_data_length_100_uniform/rnn --name rnn_default_medium --input 2000 --hidden_size 50 --latent_dim 20 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_mutations_21_unimodal_data_length_100_uniform --num_data 1000 --early_stopping True --patience 10 &&


python3 run_model.py --model_type hmm --base_log logs/synthetic_mutations_21_unimodal_data_length_50_uniform/hmm --name hmm_default_medium --input 1000 --hidden_size 25 --latent_dim 20 --seq_length 50 --pseudo_count 1 --n_jobs 5 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_mutations_21_unimodal_data_length_50_uniform --num_data 1000 --early_stopping True --patience 10 &
python3 run_model.py --model_type hmm --base_log logs/synthetic_mutations_21_unimodal_data_length_100_uniform/hmm --name hmm_default_medium --input 2000 --hidden_size 25 --latent_dim 20 --seq_length 100 --pseudo_count 1 --n_jobs 5 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_mutations_21_unimodal_data_length_100_uniform --num_data 1000 --early_stopping True --patience 10




