#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=lamb_seq128_adaptive_from64k_target0.75_bzbound128k_presetwarmup
# JOB_NAME=lamb_seq128_adaptive_from_128
OUTPUT_DIR=${base_dir}/bert_model_outputs

mkdir -p $OUTPUT_DIR

pkill python -u $UID
sleep 10s

NCCL_TREE_THRESHOLD=0 deepspeed -H hostfile ${base_dir}/deepspeed_train_p2p.py \
--cf ${base_dir}/bert_base_lamb.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 100 \
--lr_schedule "EE" \
--lr_offset 10e-4 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz64k_lamb_config_seq128.json \
--data_path_prefix /data/bert \
&> ${OUTPUT_DIR}/${JOB_NAME}.log


# --ckpt_to_save -2 \
# --lr_offset 0 \

