python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_100_gaussian/rnn --name rnn_default_small --input 2000 --hidden_size 50 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_gaussian --num_data 100 --early_stopping True --patience 10 
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_100_skewed_gaussian/rnn --name rnn_default_small --input 2000 --hidden_size 50 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_skewed_gaussian --num_data 100 --early_stopping True --patience 10 
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_100_uniform/rnn --name rnn_default_small --input 2000 --hidden_size 50 --latent_dim -1 --seq_length 100 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_100_uniform --num_data 100 --early_stopping True --patience 10 
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_20_gaussian/rnn --name rnn_default_small --input 400 --hidden_size 50 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_gaussian --num_data 100 --early_stopping True --patience 10 
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_20_skewed_gaussian/rnn --name rnn_default_small --input 400 --hidden_size 50 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_skewed_gaussian --num_data 100 --early_stopping True --patience 10 
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_20_uniform/rnn --name rnn_default_small --input 400 --hidden_size 50 --latent_dim -1 --seq_length 20 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_20_uniform --num_data 100 --early_stopping True --patience 10 
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_50_gaussian/rnn --name rnn_default_small --input 1000 --hidden_size 50 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_gaussian --num_data 100 --early_stopping True --patience 10 
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_50_skewed_gaussian/rnn --name rnn_default_small --input 1000 --hidden_size 50 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_skewed_gaussian --num_data 100 --early_stopping True --patience 10 
python3 run_model.py --model_type rnn --base_log logs/synthetic_unimodal_data_length_50_uniform/rnn --name rnn_default_small --input 1000 --hidden_size 50 --latent_dim -1 --seq_length 50 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 10 --batch_size 10 --layers 1 --dataset synthetic_unimodal_data_length_50_uniform --num_data 100 --early_stopping True --patience 10 
