#!/bin/bash

# DEV : 0.617, LB : 0.610

# seed42 # DEV : 0.595, LB : 0.584
# seed1  # DEV : 0.611, LB : 0.604
# seed35 # DEV : 0.611, LB : 0.596
# seed5  # DEV : 0.618, LB : 0.626
# seed10 # DEV : 0.610, LB : 0.615

# accum = 8
# seed5  # DEV : 0.599, LB : 0.628
# seed10 # DEV :      , LB : 0.621

GPU_ID=0

while getopts g:h OPT
do
    case $OPT in
    g ) GPU_ID=$OPTARG
        ;;
    h ) echo "Usage: $0 [-g GPU_ID]" 1>&2
        exit 1
        ;;
    esac
done

# train
export CUDA_VISIBLE_DEVICES=${GPU_ID}
python ../source/train.py \
    --model "studio-ousia/luke-japanese-large" \
    --seed 5 \
    --learning_rate 3e-5 \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --epochs 100 \
    --early_stopping_patience 3 \
    --save_model_dir "/home/miyata/python/workspace/theme_competition3/model/luke_large_accum8_seed5" \
    --sub_file ../submission/luke_large_accum8_seed5.csv
