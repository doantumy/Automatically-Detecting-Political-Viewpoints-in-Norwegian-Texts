from typing import Any
from smart_open import open
from torch.utils.data import Dataset
import pandas as pd
import math
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import torch

class Dataset(Dataset):
    def __init__(self, path: str, task_name: str, label_list_per_language:str):
        self.src = []
        self.tgt = []
        self.label_list_per_language = label_list_per_language
        df = pd.read_csv(path, delimiter="\t")
        self.src = list(df.text.str.strip())
        
        if task_name == "party_classification":
            self.tgt = list(df.party.str.lower().str.strip())
        else: # task_name == "leaning_classification" Left/Right
            value_dict = {"L": self.label_list_per_language[0], "R": self.label_list_per_language[1]}
            # print(f"value_dict: {value_dict}")
            df.block = df.block.map(value_dict)
            self.tgt = list(df.block.str.strip())

    def __getitem__(self, index):
        return self.src[index], self.tgt[index]

    def __len__(self):
        return len(self.src)


class EncoderDecoderCollator:
    def __init__(self, tokenizer, max_length, task_prefix, mapping=None): 
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_prefix = task_prefix

    def __call__(self, samples):
        src = [self.task_prefix + src + " |> " for src, _ in samples]
        tgt = [tgt for _, tgt in samples]  
        src = self.tokenizer(
            src,
            max_length=math.floor(self.max_length/4)*3,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        tgt = self.tokenizer(
            tgt,
            max_length=math.floor(self.max_length/4),
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        labels = tgt.input_ids
        # replace padding tokens id with -100 so it will be ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return src.input_ids, src.attention_mask, labels


class EvaluationMetrics:
    def __init__(self, labels_list=None):
        # Initialize empty lists to accumulate predictions and labels
        self.preds = []
        self.labels = []        
        self.labels_list = labels_list  # The unique list of labels (if available)

    def reset(self):
        self.preds = []
        self.labels = []
        
    def add_batch(self, batch_preds, batch_labels):
        # Accumulate predictions and labels for each batch
        self.preds.extend(batch_preds)
        self.labels.extend(batch_labels)

    def compute_score(self, print_matrix=False):
        # Convert lists to numpy arrays
        preds_arr = np.array(self.preds)
        labels_arr = np.array(self.labels)
        print(f"preds_arr {set(preds_arr)}")
        print(f"labels_arr {set(labels_arr)}")
        
        # Ensure the lengths of predictions and labels are the same
        if len(preds_arr) != len(labels_arr):
            raise ValueError("Mismatch between number of predictions and labels.")
        
        # Calculate accuracy
        results = preds_arr == labels_arr
        accuracy = np.sum(results) / len(labels_arr)
    
        # Calculate macro F1 score
        f1_macro = f1_score(labels_arr, preds_arr, average='macro', zero_division=0)
        
        if print_matrix:
            # Print confusion matrix
            print("Confusion Matrix:")
            print(confusion_matrix(labels_arr, preds_arr, labels=self.labels_list))
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(labels_arr, preds_arr, target_names=self.labels_list))
        
        return {
            "accuracy": accuracy,
            "f1": f1_macro
        }
        