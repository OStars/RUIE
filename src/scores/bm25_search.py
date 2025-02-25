import os
import sys
import gzip
import numpy as np

sys.path.insert(0, 'src/')
from tqdm import tqdm
from retriv import SparseRetriever
from typing import Dict, List
from logger_config import logger
from utils import save_dataset
from config import Arguments
from transformers import HfArgumentParser
from datasets import Dataset, load_dataset, DownloadMode

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]

def un_gz(path):
    ungz_path = path.replace(".gz", "")
    gz_file = gzip.GzipFile(path)
    open(ungz_path, "wb+").write(gz_file.read())
    gz_file.close()

    return ungz_path

def build_bm25_index():
    try:
        bm25 = SparseRetriever.load(args.bm25_index)
    except:
        bm25 = SparseRetriever(
            index_name=args.bm25_index,
            model="bm25",
            min_df=1,
            tokenizer="whitespace",
            stemmer="english",
            stopwords="english",
            do_lowercasing=True,
            do_ampersand_normalization=True,
            do_special_chars_normalization=True,
            do_acronyms_normalization=True,
            do_punctuation_removal=True,
        )
        corpus_path: str = '{}/passages.jsonl.gz'.format(args.data_dir)
        ungz_corpus_path: str = un_gz(corpus_path)
        bm25 = bm25.index_file(
            ungz_corpus_path,
            show_progress=True,
            callback=lambda doc: {
                "id": doc["id"],
                "text": doc["original_contents"] if "original_contents" in doc else doc["contents"],
                "raw_text": doc["original_text"] if "original_text" in doc else doc["text"],
            }
        )
        os.remove(ungz_corpus_path)

    return bm25


def bm25_search():
    data_path: str = '{}/train.jsonl.gz'.format(args.data_dir)
    dataset: Dataset = load_dataset(
        'json', data_files=data_path, split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD
    )
    if args.dry_run:
        dataset = dataset.shuffle(seed=args.seed).select(range(100))

    bm25 = build_bm25_index()
    doc_ids, doc_scores = [], []
    for query_sample in tqdm(dataset):
        text = query_sample["original_text"] if "original_text" in query_sample else query_sample["text"]
        query = query_sample["original_query"] if "original_query" in query_sample else query_sample["query"]
        search_result = bm25.search(query=query, return_docs=True, cutoff=args.search_topk)

        doc_ids.append([res["id"] for res in search_result if text != res["raw_text"]])
        doc_scores.append([res["score"] for res in search_result if text != res["raw_text"]])
    dataset = dataset.add_column("doc_ids", doc_ids)
    dataset = dataset.add_column("doc_scores", doc_scores)

    return dataset


def main():
    logger.info('Args={}'.format(str(args)))
    out_path: str = '{}/{}.jsonl.gz'.format(
        args.data_dir, args.search_split
    )
    if os.path.exists(out_path):
        logger.info('Output file {} exists. Skip.'.format(out_path))
        return

    bm25_dataset = bm25_search()
    save_dataset(bm25_dataset, out_path)

if __name__ == "__main__":
    main()
