#!/bin/bash
{
base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=bert_large_lamb_seq128_bsz4k
# JOB_NAME=lamb_seq128_adaptive_from_128
OUTPUT_DIR=${base_dir}/bert_model_outputs

mkdir -p $OUTPUT_DIR

pkill python -u $UID
sleep 10s


# NCCL_TREE_THRESHOLD=0 deepspeed -H hostfile --master_port 29501 ${base_dir}/deepspeed_train.py \
deepspeed -H /dev/null ${base_dir}/deepspeed_train_fixed_microbatch.py \
--cf ${base_dir}/bert_large_lamb_bsz128.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--ckpt_to_save 146 147 148 149 150 \
--nowarmup \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 100 \
--lr_schedule "EE" \
--lr_offset 10e-4 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz4k_lamb_config_seq128.json \
--data_path_prefix /data/bert |& tee ${OUTPUT_DIR}/${JOB_NAME}.log


exit
}