#!/usr/bin/env bash

set -x
set -e


DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH="bm25"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

# LLM_MODEL_NAME_OR_PATH="api_deepseek"
LLM_MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    LLM_MODEL_NAME_OR_PATH=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${MODEL_NAME_OR_PATH}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data"
fi

N_SHOTS=8
# EVAL_TASKS=("all")
EVAL_TASKS=("NER_CrossNER_AI" "NER_CrossNER_literature" "NER_CrossNER_music" "NER_CrossNER_politics" "NER_CrossNER_science" "NER_mit-movie" "NER_mit-restaurant" "RE_fewrel_0" "RE_fewrel_1" "RE_fewrel_2" "RE_fewrel_3" "RE_fewrel_4" "RE_wiki_0" "RE_wiki_1" "RE_wiki_2" "RE_wiki_3" "RE_wiki_4" 'EEA_RAMS_artifactexistence' 'EEA_RAMS_conflict' 'EEA_RAMS_contact' 'EEA_RAMS_disaster' 'EEA_RAMS_government' 'EEA_RAMS_inspection' 'EEA_RAMS_justice' 'EEA_RAMS_life' 'EEA_RAMS_manufacture' 'EEA_RAMS_movement' 'EEA_RAMS_personnel' 'EEA_RAMS_transaction' 'EET_RAMS_artifactexistence' 'EET_RAMS_conflict' 'EET_RAMS_contact' 'EET_RAMS_disaster' 'EET_RAMS_government' 'EET_RAMS_inspection' 'EET_RAMS_justice' 'EET_RAMS_life' 'EET_RAMS_manufacture' 'EET_RAMS_movement' 'EET_RAMS_personnel' 'EET_RAMS_transaction' "EET_wikievents_Control" "EET_wikievents_Justice" "EET_wikievents_ArtifactExistence" "EET_wikievents_Movement" "EET_wikievents_Transaction" "EET_wikievents_Medical" "EET_wikievents_Disaster" "EET_wikievents_Personnel" "EET_wikievents_Cognitive" "EET_wikievents_GenericCrime" "EET_wikievents_Contact" "EET_wikievents_Life" "EET_wikievents_Conflict" "EEA_wikievents_Control" "EEA_wikievents_Justice" "EEA_wikievents_ArtifactExistence" "EEA_wikievents_Movement" "EEA_wikievents_Transaction" "EEA_wikievents_Medical" "EEA_wikievents_Disaster" "EEA_wikievents_Personnel" "EEA_wikievents_Cognitive" "EEA_wikievents_GenericCrime" "EEA_wikievents_Contact" "EEA_wikievents_Life" "EEA_wikievents_Conflict" "EET_CrudeOilNews" "EEA_CrudeOilNews")

PYTHONPATH=src/ python -u src/scores/gen_few_shot_prompt.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --seed 1234 \
    --fp16 \
    --llm_eval_tasks "${EVAL_TASKS[@]}" \
    --llm_eval_split test \
    --llm_k_shot "${N_SHOTS}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --max_test_samples "500000" \
    --retrieval_range "single" \
    --add_qd_prompt \
    --keyword_augment


python src/uie_eval.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --seed 1234 \
    --fp16 \
    --do_llm_eval \
    --llm_model_name_or_path "${LLM_MODEL_NAME_OR_PATH}" \
    --llm_batch_size_per_device 4 \
    --llm_eval_tasks "${EVAL_TASKS[@]}" \
    --llm_eval_split test \
    --llm_k_shot "${N_SHOTS}" \
    --llm_max_input_length 1792 \
    --llm_max_decode_length 256 \
    --repetition_penalty 1.0 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to none "$@" \
    --use_vllm