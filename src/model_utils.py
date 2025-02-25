import os

from datasets import Dataset

from llms import BaseLLM, LLaMA, ApiModel
from evaluation import BaseEval, RandomEval, BM25Eval, DenseEval
from config import Arguments
from logger_config import logger


def build_llm(args: Arguments) -> BaseLLM:
    model_name_or_path: str = args.llm_model_name_or_path
    if 'llama' in model_name_or_path or "qwen" in model_name_or_path or "mistral" in model_name_or_path:
        llm = LLaMA(args=args, model_name_or_path=model_name_or_path)
    elif model_name_or_path.startswith("api_"):
        llm = ApiModel(args=args, model_name_or_path=model_name_or_path)
    else:
        raise ValueError('Invalid model name or path: {}'.format(model_name_or_path))

    return llm


def build_eval_model(args: Arguments, corpus: Dataset) -> BaseEval:
    model_name_or_path: str = args.model_name_or_path
    if model_name_or_path == 'random':
        return RandomEval(args=args, corpus=corpus)
    elif model_name_or_path == 'bm25':
        return BM25Eval(args=args, corpus=corpus)
    else:
        return DenseEval(args=args, corpus=corpus)


def parse_model_id(model_name_or_path: str) -> str:
    return os.path.basename(model_name_or_path.strip('/'))
