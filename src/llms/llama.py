import torch
import tqdm
import numpy as np

from contextlib import nullcontext
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from typing import List
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama import LlamaTokenizerFast, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from utils import move_to_device
from logger_config import logger
from config import Arguments
from llms.base_llm import BaseLLM
from collators.llama_collator import ScoreCollator

from vllm import LLM, SamplingParams


class LLaMA(BaseLLM):

    def __init__(self, args: Arguments, model_name_or_path: str = 'meta-llama/Llama-3.1-8B-Instruct', **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.args = args
        self.tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.truncation_side = 'left'
        self.batch_size_per_device = args.llm_batch_size_per_device

        dtype = torch.float16 if args.fp16 else torch.float32
        
        if self.args.use_vllm:
            self.model = LLM(
                model_name_or_path,
                gpu_memory_utilization=0.95,
                max_model_len=self.args.llm_max_input_length+self.args.llm_max_decode_length, 
                seed=self.args.seed, 
                dtype=dtype,
            )
        else:
            self.model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.model.eval()

    @torch.no_grad()
    def batch_score(
            self, input_texts: List[str], output_texts: List[str],
            delimiter: str = '\n', **kwargs
    ) -> List[float]:
        assert len(input_texts) == len(output_texts), '{} != {}'.format(len(input_texts), len(output_texts))
        assert not all(output in ['A', 'B', 'C', 'D'] for output in output_texts), 'output_texts should not be letters'

        collator = ScoreCollator(
            tokenizer=self.tokenizer,
            max_length=self.args.llm_max_input_length,
            pad_to_multiple_of=8,
            delimiter=delimiter,
        )

        dataset = Dataset.from_dict({
            'input_texts': input_texts,
            'output_texts': output_texts
        })
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=2,
            collate_fn=collator,
            pin_memory=True
        )

        avg_log_probs: List[float] = []
        for batch_dict in tqdm.tqdm(data_loader, desc='batch score', mininterval=10):
            if 'llama' in self.model_name_or_path and 'token_type_ids' in batch_dict:
                del batch_dict['token_type_ids']

            batch_dict = move_to_device(batch_dict, device=self.model.device)
            with torch.amp.autocast('cuda') if self.args.fp16 else nullcontext():
                outputs: CausalLMOutputWithCrossAttentions = self.model(
                    **batch_dict, return_dict=True, use_cache=False
                )

                labels = batch_dict['labels']
                # Shift so that tokens < n predict n
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction='none')
                per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                per_sequence_loss = per_token_loss.view(batch_dict['input_ids'].size(0), -1).sum(dim=1)
                # divide by the number of valid labels
                num_valid_labels = torch.sum(labels != -100, dim=1).float()
                avg_log_probs += (-per_sequence_loss / num_valid_labels).cpu().tolist()

                logger.debug('num_valid_labels: {}, loss: {}, per_token_loss: {}, avg_per_token_loss: {}'.format(
                    num_valid_labels, outputs.loss, per_token_loss,
                    per_token_loss.sum() / torch.sum(labels != -100).float())
                )

        return avg_log_probs

    def vllm_decode(self, input_texts: List[str], **kwargs) -> List[str]:
        if "llama-3" in self.args.llm_model_name_or_path:
            input_texts = self.apply_template(input_texts)

        sampling_params = SamplingParams(
            temperature=0.0,
            repetition_penalty=self.args.repetition_penalty,
            max_tokens=self.args.llm_max_decode_length,
            skip_special_tokens=True,
            stop=["\n\n"] if "llama-3" not in self.args.llm_model_name_or_path else None,
            stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")] if "llama-3" in self.args.llm_model_name_or_path else None
        )

        input_token_ids = self.tokenizer(
            input_texts,
            max_length=self.args.llm_max_input_length,
            truncation=True,
        )["input_ids"]

        outputs = self.model.generate(prompt_token_ids=input_token_ids, sampling_params=sampling_params)
        return [output.outputs[0].text for output in outputs]

    def apply_template(self, input_texts: List[str]) -> List[str]:
        input_token_ids = self.tokenizer(
            input_texts,
            max_length=self.args.llm_max_input_length - 32,
            truncation=True,
        )["input_ids"]

        truncation_texts = [self.tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in input_token_ids]
        processed_texts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": text}
                ],
                tokenize=False,
            ) for text in truncation_texts
        ]

        return processed_texts