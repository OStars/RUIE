import os
import torch
import random
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from time import sleep
from typing import List, Optional, Union

from utils import move_to_device
from logger_config import logger
from config import Arguments
from llms.base_llm import BaseLLM

class ApiModel(BaseLLM):

    def __init__(self, args: Arguments, model_name_or_path: str = 'api_deepseek', **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.args = args
        self.model_name_or_path = model_name_or_path[4:]
        if "deepseek" in self.model_name_or_path:
            self.model = "deepseek-chat"
            self.client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/")
        elif "zhipu" in self.model_name_or_path:
            self.model = "GLM-4-Air"
            self.client = OpenAI(api_key=os.environ.get("ZHIPU_API_KEY"), base_url="https://open.bigmodel.cn/api/paas/v4/")
        else:
            raise ValueError('Unsupported Api-based LLM: {}'.format(self.model_name_or_path))

    
    def single_decode(self, input_texts: List[str], prefix_trie=None, **kwargs) -> List[str]:
        decoded_texts: List[str] = []
        for idx, input_text in tqdm(enumerate(input_texts), mininterval=10, desc='single decode'):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": input_text},
                    ],
                    max_tokens=self.args.llm_max_decode_length,
                    temperature=0.0,
                    stream=False
                )

                decoded_texts.append(response.choices[0].message.content)
                sleep(random.random())
            except Exception as e:
                if "zhipu" in self.model_name_or_path and e.code == "1301":
                    decoded_texts.append("None")
                    print(e)
                    print("Error input idx: {}. Error query: {}".format(idx, input_text.split("\n\n")[-1]))
                    continue
                elif "deepseek" in self.model_name_or_path and e.code == "400" or e.code == "invalid_request_error":
                    decoded_texts.append("None")
                    print(e)
                    print("Error input idx: {}. Error query: {}".format(idx, input_text.split("\n\n")[-1]))
                    continue
                print(e)
                print("Error input idx: {}. Error query: {}".format(idx, input_text.split("\n\n")[-1]))
                break
        
        if len(decoded_texts) != len(input_texts):
            logger.error("Some errors occur in api decoding!!!")
            decoded_texts = decoded_texts + [""] * (len(input_texts) - len(decoded_texts))

        return decoded_texts

    def vllm_decode(self, input_texts: List[str], prefix_trie=None, **kwargs) -> List[str]:
        return self.single_decode(input_texts, prefix_trie, **kwargs)

    def cuda(self, device: Optional[Union[int, torch.device]] = 0):
        return self
