#!/bin/bash


NGPU_PER_NODE=$1
NAME=$2
MODEL_FILE=$3
REPEATED_TEST=${4:-0}
LOCAL_BATCH_SIZE=${5:-0}
SQUAD_DIR=/data/BingBertSquad
LR=0.00003
# SEED=12345
MASTER_PORT=29500
echo "seed is $SEED"
echo "master port is $MASTER_PORT"
pkill python
sleep 5s
mkdir -p log
mkdir -p outputs
NUM_NODES=1
NGPU=$((NGPU_PER_NODE*NUM_NODES))
EFFECTIVE_BATCH_SIZE=24
MAX_GPU_BATCH_SIZE=${LOCAL_BATCH_SIZE}
PER_GPU_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/NGPU))
if [[ $PER_GPU_BATCH_SIZE -lt $MAX_GPU_BATCH_SIZE ]]; then
       GRAD_ACCUM_STEPS=1
else
       GRAD_ACCUM_STEPS=$((PER_GPU_BATCH_SIZE/MAX_GPU_BATCH_SIZE))
fi
JOB_NAME="${NAME}_SQuAD1.1_${EFFECTIVE_BATCH_SIZE}batch_size_${REPEATED_TEST}"

if [ -f log/${JOB_NAME}.log ]; then
   echo "File ${JOB_NAME}.log exists."
   exit 1
fi

OUTPUT_DIR=./outputs/${JOB_NAME}
config_json=deepspeed_bsz24_localbsz${LOCAL_BATCH_SIZE}_config.json
run_cmd="deepspeed --num_nodes ${NUM_NODES} --num_gpus ${NGPU_PER_NODE} \
       --master_port=${MASTER_PORT} \
       nvidia_run_squad_deepspeed.py \
       --bert_model bert-large-uncased \
       --do_train \
       --do_lower_case \
       --predict_batch_size ${LOCAL_BATCH_SIZE} \
       --do_predict \
       --train_file $SQUAD_DIR/train-v1.1.json \
       --predict_file $SQUAD_DIR/dev-v1.1.json \
       --train_batch_size $PER_GPU_BATCH_SIZE \
       --learning_rate ${LR} \
       --num_train_epochs 2.0 \
       --max_seq_length 384 \
       --doc_stride 128 \
       --output_dir $OUTPUT_DIR \
       --job_name ${JOB_NAME} \
       --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
       --fp16 \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --deepspeed_transformer_kernel \
       --model_file $MODEL_FILE \
       --preln \
       &> log/${JOB_NAME}.log"
echo ${run_cmd}
eval ${run_cmd}

if ! [ -f ${OUTPUT_DIR}/predictions.json ]; then
   echo "File ${OUTPUT_DIR}/predictions.json does not exists."
   exit 1
fi

run_cmd="python evaluate-v1.1.py --prediction_file ${OUTPUT_DIR}/predictions.json
       &> ${OUTPUT_DIR}/result.json"
echo ${run_cmd}
eval ${run_cmd}

