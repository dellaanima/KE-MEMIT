#!/bin/bash

# Run evaluation for each hparams file
for hparams_file in ~/workspace/memit/hparams/MEMIT/TinyLlama-1.1B-Chat-v1.0/layer_combinations_*.json
do
    CUDA_VISIBLE_DEVICES=1 python3 -m experiments.evaluate \
        --alg_name=MEMIT \
        --model_name=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --hparams_fname="$hparams_file" \
        --num_edits=1000 \
        --dataset_size_limit=1000
done
