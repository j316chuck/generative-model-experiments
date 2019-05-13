# rnn_default_medium
python3 run_model.py --model_type rnn --base_log logs/gfp/rnn/ --name rnn_default_medium_0 --input 4998 --hidden_size 200 --latent_dim -1 --seq_length 238 --pseudo_count 1 --n_jobs 1 --device cuda:0 --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset gfp --num_data 50 --early_stopping True --patience 10 &
# rnn_default_medium
python3 run_model.py --model_type rnn --base_log logs/gfp/rnn/ --name rnn_default_medium_1 --input 4998 --hidden_size 200 --latent_dim -1 --seq_length 238 --pseudo_count 1 --n_jobs 1 --device cuda:1 --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset gfp --num_data 50 --early_stopping True --patience 10 &
# rnn_default_medium
python3 run_model.py --model_type rnn --base_log logs/gfp/rnn/ --name rnn_default_medium_2 --input 4998 --hidden_size 200 --latent_dim -1 --seq_length 238 --pseudo_count 1 --n_jobs 1 --device cuda:2 --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset gfp --num_data 50 --early_stopping True --patience 10 &
# rnn_default_medium
python3 run_model.py --model_type rnn --base_log logs/gfp/rnn/ --name rnn_default_medium_3 --input 4998 --hidden_size 200 --latent_dim -1 --seq_length 238 --pseudo_count 1 --n_jobs 1 --device cuda:3 --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset gfp --num_data 50 --early_stopping True --patience 10 

