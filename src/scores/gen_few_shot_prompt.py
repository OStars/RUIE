import os
import sys
sys.path.insert(0, 'src/')

from datasets import Dataset, load_dataset, DownloadMode, concatenate_datasets
from typing import List, Tuple
from transformers import HfArgumentParser
from retriv import SparseRetriever

from config import Arguments
from logger_config import logger
from utils import save_dataset
from evaluation import BaseEval
from model_utils import build_eval_model


parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]

def get_prompt_save_path(args: Arguments) -> str:
    from model_utils import parse_model_id
    model_id: str = parse_model_id(args.model_name_or_path)
    out_path: str = '{}/{}_{}_k{}.jsonl.gz'.format(
        args.output_dir, model_id, args.llm_eval_split, args.llm_k_shot
    )

    return out_path

def main():
    out_path: str = get_prompt_save_path(args=args)
    if os.path.exists(out_path):
        logger.info('Prompt file {} exists. Skip.'.format(out_path))
        return

    corpus: Dataset = load_dataset(
        'json', data_files='{}/passages.jsonl.gz'.format(args.data_dir), split='train',
        download_mode=DownloadMode.FORCE_REDOWNLOAD
    )
    # columns: query_id / query / answers / task_name
    eval_dataset: Dataset = load_dataset(
        'json', data_files='{}/{}.jsonl.gz'.format(args.data_dir, args.llm_eval_split), split='train',
        download_mode=DownloadMode.FORCE_REDOWNLOAD
    )

    if not args.llm_eval_tasks or args.llm_eval_tasks[0] == 'all':
        args.llm_eval_tasks = sorted(eval_dataset.unique('task_name'))
        logger.info('Eval all {} tasks'.format(len(args.llm_eval_tasks)))

    model: BaseEval = build_eval_model(args=args, corpus=corpus)

    task_ds_list: List[Dataset] = []
    for task_name in args.llm_eval_tasks:
        task_ds: Dataset = eval_dataset.filter(lambda x: x['task_name'] == task_name)

        if len(task_ds) > args.max_test_samples:
            logger.info('Task: {}, random sample {}/{} for evaluation'.format(
                task_name, args.max_test_samples, len(task_ds))
            )
            task_ds = task_ds.shuffle(seed=args.seed).select(range(args.max_test_samples))
        logger.info('Task: {}, {} samples for evaluation'.format(task_name, len(task_ds)))

        if args.llm_k_shot <= 0:
            task_ds = task_ds.add_column('input_prompt', ['' for _ in range(len(task_ds))])
            task_ds_list.append(task_ds)
            continue

        queries: List[str] = task_ds['query']
        # Use a larger k in case we retrieve the docs from other tasks
        topk_score_doc_ids: List[List[Tuple[float, str]]] = model.get_topk_score_doc_ids(
            queries, k=5 * args.llm_k_shot, task_names=task_ds['task_name']
        )
        topk_score_doc_ids = [[score_doc_id for score_doc_id in score_doc_ids if test_sample["text"] != (corpus[int(score_doc_id[1])]["text"] if not args.keyword_augment else corpus[int(score_doc_id[1])]["original_text"])] 
                              for test_sample, score_doc_ids in zip(task_ds, topk_score_doc_ids)]
        # The most relevant doc should be close to the test example
        topk_score_doc_ids = [score_doc_ids[:args.llm_k_shot][::-1] for score_doc_ids in topk_score_doc_ids]
        topk_doc_ids: List[List[str]] = [
            [doc_id for _, doc_id in score_doc_ids] for score_doc_ids in topk_score_doc_ids
        ]
        topk_scores: List[List[float]] = [
            [round(score, 4) for score, _ in score_doc_ids] for score_doc_ids in topk_score_doc_ids
        ]
        input_prompts: List[str] = [model.get_prompt_by_doc_ids(doc_ids) for doc_ids in topk_doc_ids]
        task_ds = task_ds.add_column('input_prompt', input_prompts)
        task_ds = task_ds.add_column('topk_doc_ids', topk_doc_ids)
        task_ds = task_ds.add_column('topk_scores', topk_scores)
        task_ds_list.append(task_ds)

    if args.model_name_or_path == "bm25":
        SparseRetriever.delete("bm25_{}_eval".format(args.retrieval_range))
    few_shot_ds: Dataset = concatenate_datasets(task_ds_list)
    save_dataset(few_shot_ds, out_path)
    logger.info('Save {} examples to {}'.format(len(few_shot_ds), out_path))


if __name__ == '__main__':
    main()
