from typing import Any
# from smart_open import open
from torch.utils.data import Dataset
import pandas as pd
import math
from transformers import AutoTokenizer

class Dataset(Dataset):
    def __init__(self, path: str):
        self.src = []
        self.tgt = []
        df = pd.read_csv(path, delimiter="\t")
        self.src = list(df.text)
        self.tgt = list(df.label)

    def __getitem__(self, index):
        return self.src[index], self.tgt[index]

    def __len__(self):
        return len(self.src)


class EncoderDecoderCollator:
    def __init__(self, tokenizer, max_length, task_prefix): 
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_prefix = task_prefix

    def __call__(self, samples):
        src = [self.task_prefix + src + " |> " for src, _ in samples]
        tgt = [tgt for _, tgt in samples]

        src = self.tokenizer(
            src,
            max_length=math.floor(self.max_length/3)*2,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        tgt = self.tokenizer(
            tgt,
            max_length=math.floor(self.max_length/3),
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        labels = tgt.input_ids
        # replace padding tokens id with -100 so it will be ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return src.input_ids, src.attention_mask, labels # tgt.input_ids
