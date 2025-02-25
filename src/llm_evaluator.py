import os
import json
import numpy as np

from typing import Dict, List, Optional
from transformers import AutoTokenizer
from datasets import Dataset

from config import Arguments
from logger_config import logger
from llms import BaseLLM
from model_utils import parse_model_id
from data_utils import save_llm_decoding_results
from utils import save_json_to_file, DictTrie, build_trie, wait_until_all_files_show_up


class LLMEvaluator:
    def __init__(self, args: Arguments, llm: BaseLLM):
        self.args = args
        self.llm = llm
        self.model_id: str = parse_model_id(self.args.model_name_or_path)
        self.llm_model_id: str = parse_model_id(self.args.llm_model_name_or_path)

    def eval_single_task(self, eval_dataset: Dataset, task_name: str):
        out_path: str = '{}/{}/{}_{}_decoding_results.jsonl'.format(self.args.output_dir, self.llm_model_id, task_name, self.model_id)
        if os.path.exists(out_path):
            logger.info('Task {} has already been evaluated'.format(task_name))
            return

        task_ds: Dataset = eval_dataset.filter(lambda x: x['task_name'] == task_name)
        logger.info('Task: {}, # of examples: {}'.format(task_name, len(task_ds)))
        if len(task_ds) == 0:
            logger.error('No examples for task: {}'.format(task_name))
            return
        sharded_task_ds = task_ds.shard(num_shards=self.args.world_size, index=self.args.process_index, contiguous=True)
        logger.info('Worker {} needs to process {} examples'.format(self.args.process_index, len(sharded_task_ds)))

        queries: List[str] = sharded_task_ds['query']
        input_prompts: List[str] = sharded_task_ds['input_prompt']
        instructions: List[str] = sharded_task_ds['instruction']
        assert len(input_prompts) == len(queries)
        assert all(not q.endswith('\n') for q in queries)


        input_texts: List[str] = [
            '{}\n\n{}\n---\n{}\n---\n\n{}\n\n{}'.format(instruction, "Here are some examples (For Analysis Only, Not for Extraction):", prompt, "Now strictly following my instruction to handle this input:", query) if prompt else '{}\n\n{}'.format(instruction, query) for instruction, prompt, query in
            zip(instructions, input_prompts, queries)
        ]

        decoded_texts: List[str] = self.llm.single_decode(input_texts) if not self.args.use_vllm else self.llm.vllm_decode(input_texts)

        parsed_decoded_texts = decoded_texts
        save_json_to_file({
            'input_texts': input_texts,
            'decoded_texts': decoded_texts,
            'parsed_decoded_texts': parsed_decoded_texts,
        }, self._get_tmp_path(self.args.process_index, task_name))

        if self.args.process_index <= 0:
            wait_until_all_files_show_up(
                [self._get_tmp_path(worker_idx, task_name) for worker_idx in range(self.args.world_size)]
            )
            self._save_results(task_ds=task_ds, task_name=task_name, out_path=out_path)

    def _save_results(self, task_ds: Dataset, task_name: str, out_path: str):
        # merge results from all workers
        input_texts: List[str] = []
        decoded_texts: List[str] = []
        for worker_idx in range(self.args.world_size):
            tmp_path: str = self._get_tmp_path(worker_idx, task_name)
            tmp_results: Dict = json.load(open(tmp_path, 'r', encoding='utf-8'))
            input_texts.extend(tmp_results['input_texts'])
            decoded_texts.extend(tmp_results['decoded_texts'])

        answers: List = task_ds['answers']
        if max(len(answer) for answer in answers) == 1:
            # single answer
            answers: List[str] = [answer[0] for answer in answers]

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        model_id: str = parse_model_id(self.args.model_name_or_path)
        llm_model_id: str = parse_model_id(self.args.llm_model_name_or_path)
        save_llm_decoding_results(
            out_path='{}/{}/{}_{}_decoding_results.jsonl'.format(self.args.output_dir, llm_model_id, task_name, model_id),
            input_texts=input_texts,
            decoded_texts=decoded_texts,
            answer_texts=answers,
            golden_labels=task_ds['golden_labels'],
            task_names=task_ds['task_name'],
            texts=task_ds["text"],
            queries=task_ds['query'],
            query_ids=task_ds['query_id']
        )

        for worker_idx in range(self.args.world_size):
            tmp_path: str = self._get_tmp_path(worker_idx, task_name)
            os.remove(tmp_path)

    def _get_tmp_path(self, worker_idx: int, task_name: str) -> str:
        tmp_dir = self.args.output_dir if self.args.world_size <= 1 else 'tmp/'
        llm_model_id: str = parse_model_id(self.args.llm_model_name_or_path)
        return '{}/{}/{}_{}.json'.format(tmp_dir, llm_model_id, task_name, worker_idx)
