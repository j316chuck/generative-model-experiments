python3 run_model.py --model_type vae --base_log logs/synthetic_multimodal_data_modes_2_length_51_uniform/vae --name vae_default_medium --input 1020 --hidden_size 200 --latent_dim 20 --seq_length 51 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_multimodal_data_modes_2_length_51_uniform --num_data 1000 --early_stopping True --patience 10 
python3 run_model.py --model_type vae --base_log logs/synthetic_multimodal_data_modes_3_length_51_uniform/vae --name vae_default_medium --input 1020 --hidden_size 200 --latent_dim 20 --seq_length 51 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_multimodal_data_modes_3_length_51_uniform --num_data 1000 --early_stopping True --patience 10 
python3 run_model.py --model_type vae --base_log logs/synthetic_multimodal_data_modes_5_length_51_uniform/vae --name vae_default_medium --input 1020 --hidden_size 200 --latent_dim 20 --seq_length 51 --pseudo_count 1 --n_jobs 1 --device cpu --learning_rate 0.001 --epochs 100 --batch_size 10 --layers 1 --dataset synthetic_multimodal_data_modes_5_length_51_uniform --num_data 1000 --early_stopping True --patience 10 
