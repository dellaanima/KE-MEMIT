#!/bin/bash

# Run evaluation for each hparams file
for hparams_file in ~/KE-MEMIT/hparams/MEMIT/gpt2-xl/layer_combinations_*.json
do
    # Evaluate using the standard method
    python3 -m experiments.evaluate \
        --alg_name=MEMIT \
        --model_name=gpt2-xl \
        --hparams_fname="$hparams_file" \
        --num_edits=10000 \
        --dataset_size_limit=10000
    
    
    # Evaluate using the original method
    python3 -m experiments.evaluate_original \
        --alg_name=MEMIT \
        --model_name=gpt2-xl \
        --hparams_fname="$hparams_file" \
        --num_edits=1000 \
        --dataset_size_limit=1000
    
done
