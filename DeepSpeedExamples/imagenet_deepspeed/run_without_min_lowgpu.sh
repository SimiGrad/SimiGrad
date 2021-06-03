#!/bin/bash
{
pkill python -u $UID
mkdir -p log

# python sync_with_gpuh.py &
sleep 10s

similarity_target=$1
bszbound=$2

name=resnet50_bsz2k_target${similarity_target}_bszbound${bszbound}
deepspeed --hostfile /dev/null main_deepspeed.py --arch resnet50 --epochs 90 --mixup 0.0 --name $name --max_mirco_batch_size 64 --similarity_target ${similarity_target} --batchsize_upper_bound ${bszbound} --deepspeed --deepspeed_config config_bsz2k_lowgpu.json |& tee log/$name.log
exit
}