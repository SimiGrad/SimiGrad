#!/bin/bash

{
similarity_target=$1
bszbound=$2
bsz_lowerbound=$3

name=resnet18_bsz128_target${similarity_target}_bszbound${bsz_lowerbound}_${bszbound}

mkdir -p xavier_normal_log

deepspeed main_xavier_normal.py --name $name --seed 1 --similarity_target ${similarity_target} --batchsize_lower_bound ${bsz_lowerbound} --batchsize_upper_bound ${bszbound} --deepspeed --deepspeed_config ds_config_flexible.json |& tee -a xavier_normal_log/$name.txt

exit
}