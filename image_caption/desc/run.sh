#!/bin/bash

# python generate_tgt.py --trial 'trial0' --device 'cuda:0' --extract
# python generate_tgt.py --prompt_option 1 --trial 'trial1' --extract
# python generate_tgt.py --prompt_option 2 --trial 'trial2' --extract
# python generate_tgt.py --prompt_option 1 --freq_pen 0.8 --trial 'trial3' --device 'cuda:0' --extract
# python generate_tgt.py --prompt_option 2 --freq_pen 0.8 --trial 'trial4' --device 'cuda:1' --extract
# python generate_src.py --dataset 'dreambooth' --device 'cuda:0' --extract

# Experiment
# DATASETS="sameswap CelebA tedbench"
DATASETS="CelebA tedbench"
FREQ="0 0.8"
for data in $DATASETS
    do 
    for freq in $FREQ
        do
        echo $data $opt $freq
        python3 generate.py --dataset $data --desc_prompt_option 6 --cap_prompt_option 1 --refresh_caption True --freq_pen $freq
        # For Comparison
        python3 generate.py --dataset $data --desc_prompt_option 2 --cap_prompt_option 0 --refresh_caption True --freq_pen $freq
        done
    done