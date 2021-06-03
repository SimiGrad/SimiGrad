#!/bin/bash
{
pkill python -u $UID
sleep 10s
mkdir -p log

bsz=$1
lr=$2

name=resnet50_bsz${bsz}_lr${lr}
deepspeed --hostfile /dev/null main_deepspeed.py --arch resnet50 --seed 1 --lr ${lr} --epochs 90 --mixup 0.0 --name $name --deepspeed --deepspeed_config tmp_config.json |& tee log/$name.log
exit
}