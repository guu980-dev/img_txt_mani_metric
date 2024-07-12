#!/bin/bash

# python generate_tgt.py --trial 'trial0' --device 'cuda:0' --extract
# python generate_tgt.py --prompt_option 1 --trial 'trial1' --extract
# python generate_tgt.py --prompt_option 2 --trial 'trial2' --extract
# python generate_tgt.py --prompt_option 1 --freq_pen 0.8 --trial 'trial3' --device 'cuda:0' --extract
# python generate_tgt.py --prompt_option 2 --freq_pen 0.8 --trial 'trial4' --device 'cuda:1' --extract
# python generate_src.py --dataset 'dreambooth' --device 'cuda:0' --extract

DATASETS="sameswap CelebA tedbench"
OPTIONS="1 2 3 4"
FREQ="0 0.8"
for data in $DATASETS
    do 
    for opt in $OPTIONS
        do
        for freq in $FREQ
            do
            echo $data $opt $freq
            CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_src.py --dataset $data --prompt_option $opt --freq_pen $freq
            done
        done
    done