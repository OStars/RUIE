#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/reward_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data"
fi


deepspeed --include localhost:0,1 --master_port 38888 src/train_cross_encoder.py --deepspeed ds_config.json \
    --model_name_or_path google/electra-base-discriminator \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --do_train \
    --fp16 \
    --seed 987 \
    --reward_max_length 512 \
    --train_file "${DATA_DIR}/Meta-Llama-3-8B_bm25_train.jsonl.gz" \
    --train_n_passages 8 \
    --topk_as_positive 3 --bottomk_as_negative 16 \
    --dataloader_num_workers 1 \
    --max_steps 3000 \
    --learning_rate 1e-5 \
    --warmup_steps 400 \
    --logging_steps 50 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --save_total_limit 5 \
    --save_strategy steps \
    --save_steps 1000 \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm True \
    --keyword_augment \
    --report_to none "$@"
