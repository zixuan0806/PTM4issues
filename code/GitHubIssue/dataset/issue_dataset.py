import pytorch_lightning as pl
import torch
import transformers
import tqdm
import json
import os
import re
import numpy as np
import pickle
from typing import Sequence, Union


class IssueDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Union[str, Sequence], all_labels: Sequence, tokenizer=None, lazy=False):
        self.data = []
        if isinstance(dataset, str):
            with open(dataset, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))
        else:
            self.data = dataset

        self.text_list = []
        self.label_list = []

        # id to label and label to id
        self.label_to_id = {}
        for i, c in enumerate(all_labels):
            self.label_to_id[c] = i

        self.id_to_label = {}
        for key, value in self.label_to_id.items():
            self.id_to_label[value] = key

        # convert data to matrices
        for obj in self.data:
            text = obj['title'] + ' ' + obj['description']
            # TODO: set max length
            text_ids = tokenizer(text, truncation=True, max_length=512, padding='max_length')['input_ids']

            labels = obj['labels']
            labels_ids = np.zeros((len(all_labels),))

            label_id = self.label_to_id[labels]
            labels_ids[label_id] = 1
        
            # for c in labels:
            #     label_id = self.label_to_id[c]
            #     labels_ids[label_id] = 1

            self.text_list.append(text_ids)
            self.label_list.append(labels_ids)

    def __getitem__(self, i):
        return (
            torch.tensor(self.text_list[i], dtype=torch.long),
            torch.tensor(self.label_list[i], dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)
