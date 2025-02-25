import random

from typing import List, Tuple, Dict
from datasets import Dataset
from retriv import SparseRetriever
from tqdm import tqdm

from evaluation.base_eval import BaseEval
from config import Arguments
from logger_config import logger


# This class selects k-shot examples using bm25 from the training set
class BM25Eval(BaseEval):

    def __init__(self, args: Arguments, corpus: Dataset, **kwargs):
        super().__init__(args, corpus, **kwargs)
        self.all_doc_ids: List[str] = self.corpus['id']
        self.cached_task_name_to_doc_ids: Dict[str, List[str]] = {}
        self.bm25_retriever = SparseRetriever(
            index_name="bm25_{}_eval".format(args.retrieval_range),
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
        if args.retrieval_range == "mix":
            self.bm25 = self.bm25_retriever.index(
                corpus,
                callback = lambda doc: {      # Callback defaults to None.
                    "id": doc["id"],
                    "text": doc["contents"],
                    "raw_text": doc["text"],
                }
            )

    def get_topk_score_doc_ids(self, queries: List[str], k: int, task_names: List[str]) -> List[List[Tuple[float, str]]]:
        assert len(queries) == len(task_names)

        if self.args.retrieval_range == "single":
            self.bm25 = self.bm25_retriever.index(
                self.corpus.filter(lambda x: x["task_name"] == task_names[0]),
                callback = lambda doc: {      # Callback defaults to None.
                    "id": doc["id"],
                    "text": doc["contents"],
                    "raw_text": doc["text"],
                }
            )

        topk_score_doc_ids: List[List[Tuple[float, str]]] = []
        for query, task_name in tqdm(zip(queries, task_names)):
            bm25_search_result: List[Dict] = self._single_get_topk_doc_ids(query, k, task_name)
            topk_score_doc_ids.append([(res["score"], res["id"]) for res in bm25_search_result])

        return topk_score_doc_ids

    def _single_get_topk_doc_ids(self, query: str, k: int, task_name: str) -> List[str]:
        return self.bm25.search(query=query, return_docs=True, cutoff=k)
