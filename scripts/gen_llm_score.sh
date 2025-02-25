#!/usr/bin/env bash

set -x
set -e


DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

SPLIT="bm25_train"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    SPLIT=$1
    shift
fi

if [ -z "$DATA_DIR" ]; then
    DATA_DIR="${DIR}/data/"
fi


PYTHONPATH=src/ torchrun --nproc_per_node 2 --master_port 29999 src/scores/gen_llm_scores.py \
    --llm_model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --fp16 \
    --search_split "${SPLIT}" \
    --search_topk 32 \
    --llm_batch_size_per_device 2 \
    --output_dir "/tmp/" \
    --data_dir "${DATA_DIR}" \
    --report_to none "$@" \
    --dry_run