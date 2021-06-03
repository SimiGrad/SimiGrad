#!/bin/bash

{
lr=$2
bszbound=$1

name=resnet18_bsz${bszbound}_lr${lr}_warmup

mkdir -p log

deepspeed main.py --name $name --lr ${lr} --seed 1 --warmup --deepspeed --deepspeed_config ds_config_tmp.json |& tee -a log/$name.txt

exit
}