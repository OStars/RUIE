#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH=""
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

SPLIT="bm25_train"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    SPLIT=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${MODEL_NAME_OR_PATH}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data"
fi


PYTHONPATH=src/ torchrun --nproc_per_node 2 --master_port 38888 src/scores/gen_reward_scores.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --do_kd_gen_score \
    --fp16 \
    --data_dir "${DATA_DIR}" \
    --reward_max_length 512 \
    --kd_gen_score_split "${SPLIT}" \
    --kd_gen_score_batch_size 128 \
    --dataloader_num_workers 1 \
    --output_dir "${OUTPUT_DIR}" \
    --report_to none "$@" \
    --keyword_augment \
    --dry_run
