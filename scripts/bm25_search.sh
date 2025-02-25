#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

SPLIT="bm25_train"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    SPLIT=$1
    shift
fi

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/"
fi
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DATA_DIR}"
fi

PYTHONPATH=src/ python -u src/scores/bm25_search.py \
    --do_search \
    --bm25_index "ruie_bm25" \
    --search_split "${SPLIT}" \
    --search_topk 100 \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --report_to none "$@" \
    --dry_run
