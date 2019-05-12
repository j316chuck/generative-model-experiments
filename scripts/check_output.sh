#!/bin/bash

python3 parallelize_experiments.py --num_processes 3 --script ./scripts/vae_unimodal_small_script.sh &&
python3 parallelize_experiments.py --num_processes 3 --script ./scripts/vae_unimodal_medium_script.sh &&
python3 parallelize_experiments.py --num_processes 3 --script ./scripts/vae_multimodal_medium_script.sh &&
python3 parallelize_experiments.py --num_processes 3 --script ./scripts/rnn_unimodal_medium_script.sh &&
python3 parallelize_experiments.py --num_processes 3 --script ./scripts/rnn_multimodal_medium_script.sh &&
python3 parallelize_experiments.py --num_processes 3 --script ./scripts/hmm_multimodal_medium_script.sh &&
python3 parallelize_experiments.py --num_processes 3 --script ./scripts/hmm_multimodal_medium_script.sh
