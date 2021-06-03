#!/bin/bash

{
similarity_target=0.4
bszbound=2k
bsz_lowerbound=32

# name=resnet18_bsz128_target${similarity_target}_bszbound${bszbound}
name=resnet18_bsz128_target${similarity_target}_bszbound${bsz_lowerbound}_${bszbound}
# resnet18_bsz128_target0.4_bszbound32_2k
mkdir -p log

deepspeed main.py --name $name --seed 1 --similarity_target ${similarity_target} --batchsize_lower_bound ${bsz_lowerbound} --batchsize_upper_bound ${bszbound} --deepspeed --deepspeed_config ds_config_flexible.json |& tee -a log/$name.txt

exit
}