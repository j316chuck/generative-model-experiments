# vae
python3 main.py --model vae --name vae_test_sample_large --input 4998 --hidden_size 50 --latent_dim 20 --seq_length 238 --device cpu --learning_rate 0.001 --epochs 20 --batch_size 10 --layers 2 --dataset gfp_amino_acid --num_data 1000

# hmm 
python3.6 main.py --model_type hmm --name hmm_test_sample --hidden_size 50 --epochs 20 --batch_size 10 --dataset gfp_amino_acid --num_data 100 --n_jobs 10 --pseudo_count 1
