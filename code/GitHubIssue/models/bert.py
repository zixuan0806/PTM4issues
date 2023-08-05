import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torchmetrics

import torch

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import AlbertTokenizer, AlbertModel
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ..metrics.accuracy import MultiLabelAccuracy
from ..metrics.precision import MultiLabelPrecision
from ..metrics.recall import MultiLabelRecall


MODEL_CONFIG = {
    "bert-base-uncased": BertModel,
    "albert-base-v2": AlbertModel,
    "roberta-base": RobertaModel,
    "microsoft/codebert-base": RobertaModel,

}

SEQUENCE_MODEL_CONFIG = {
    "xlnet-base-cased": XLNetForSequenceClassification,
    "bert-base-uncased": BertForSequenceClassification,
    "roberta-base": RobertaForSequenceClassification,
    "huggingface/CodeBERTa-language-id": RobertaForSequenceClassification,
    "seBERT": BertForSequenceClassification,
    "jeniya/BERTOverflow": AutoModelForTokenClassification,
}


class Bert(pl.LightningModule):
    def __init__(self, num_classes: int, model_name: str='bert-base-uncased', use_sequence: bool=False, disablefinetune: bool=False, local_model: bool=False):
        super().__init__()
        self.class_num = num_classes
        self.model_name = model_name
        self.use_sequence = use_sequence
        self.disablefinetune = disablefinetune
        self.local_model = local_model

        print(f"current model is :{model_name}")
        print(f"current num_classes is :{num_classes}")

        # 本地模型需要从路径中提取出模型名称
        if self.local_model:
            self.model_path = self.model_name # 保存本地模型路径
            model_name = self.model_name.split('/')[-1]
            self.model_name = model_name

        if not self.use_sequence:
            self.bert = MODEL_CONFIG[model_name].from_pretrained(model_name)
        else:
            if self.local_model:
                # self.config = "/".join(self.model_path.split(r'/')[:-1] + ['config.json'])
                # real_model_name = model_name.split('/')[-2]
                    # 本地模型需要从路径中提取出模型名称
                self.bert = SEQUENCE_MODEL_CONFIG[model_name].from_pretrained(self.model_path, num_labels=num_classes, ignore_mismatched_sizes=True)

                # self.bert = SEQUENCE_MODEL_CONFIG[model_name].from_pretrained(self.model_path, config=self.config, num_labels=num_classes, ignore_mismatched_sizes=True)
            else:
                # ignore_mismatched_sizes will randomly generate the initial parameters for classifier
                # after transformers version == 4.9.0
                print(f"model_name:{model_name}")
                self.bert = SEQUENCE_MODEL_CONFIG[model_name].from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
                # code for transformers version == 4.5.1
                # self.bert = SEQUENCE_MODEL_CONFIG[model_name].from_pretrained(model_name, num_labels=num_classes)
            
            if disablefinetune:  # disable finetune for sequence model
                for name, param in self.bert.named_parameters():
                    # if 'logits' not in name: # classifier layer for xlnet
                    #     param.requires_grad = False

                    if 'classifier' not in name: # classifier layer for bert base model
                        param.requires_grad = False

        self.hid_dim = self.bert.config.hidden_size
        
        # self.fc_0 = nn.Linear(self.hid_dim, self.hid_dim, bias=True)

        self.fc = nn.Linear(self.hid_dim, self.class_num, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        #self.loss = nn.CrossEntropyLoss()BCE
        self.loss = nn.BCELoss()

        self.metrics = nn.ModuleDict()
        stage_name = ['train', 'valid', 'test']
        for stage in stage_name:
            for k in range(1, 3):
                self.metrics[f"{stage}_acc_{k}"] = MultiLabelAccuracy(top_k=k)
                self.metrics[f"{stage}_precision_{k}"] = torchmetrics.Precision(top_k=k)
                self.metrics[f"{stage}_recall_{k}"] = torchmetrics.Recall(top_k=k)
                # self.metrics[f"{stage}_acc_{k}"] = MultiLabelAccuracy(top_k=k)
                # self.metrics[f"{stage}_precision_{k}"] = MultiLabelPrecision(top_k=k)
                # self.metrics[f"{stage}_recall_{k}"] = MultiLabelRecall(top_k=k)
                self.metrics[f"{stage}_f1_marco_{k}"] = torchmetrics.F1(average='macro', num_classes=num_classes, top_k=k)
                self.metrics[f"{stage}_f1_marco_weight_{k}"] = torchmetrics.F1(average='weighted', num_classes=num_classes, top_k=k)
                self.metrics[f"{stage}_f1_mirco_{k}"] = torchmetrics.F1(average='micro', num_classes=num_classes, top_k=k)

    def forward(self, input_ids):
        # in lightning, forward defines the prediction/inference actions
        # with torch.no_grad():
        output = self.bert(input_ids)
        
        if not self.use_sequence:
            last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
            if self.disablefinetune:
                pooler_output = pooler_output.detach()
            x = self.dropout(pooler_output)
            logits = torch.sigmoid(self.fc(x))
            # logits = torch.sigmoid(self.fc(self.fc_0(x)))
        else:
            logits = torch.sigmoid(output.logits)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        logits = self.forward(x)
        # print(f'type of logits: {type(logits)}, logits.shape: {logits.shape} ,logits:{logits}')
        # print(f'type of y: {type(y)}, y.shape: {y.shape}, logits{y}')


        loss = self.loss(logits, y.float())
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        #
        for name, metric in self.metrics.items():
            if name.startswith('train_'):
                self.log(f"{name}_step", metric(logits, y))
        return loss

    def training_epoch_end(self, outputs):
        for name, metric in self.metrics.items():
            if name.startswith('train_'):
                self.log(f"{name}_epoch", metric.compute())
                metric.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y.float())
        self.log('val_loss', loss)

        for name, metric in self.metrics.items():
            if name.startswith('valid_'):
                self.log(f"{name}_step", metric(logits, y))

    def validation_epoch_end(self, outs):
        for name, metric in self.metrics.items():
            if name.startswith('valid_'):
                self.log(f"{name}_epoch", metric.compute())
                metric.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        for name, metric in self.metrics.items():
            if name.startswith('test_'):
                self.log(f"{name}_step", metric(logits, y))
        # return {'loss': loss, 'pred': pred}

    def test_epoch_end(self, outs):
        # log epoch metric
        for name, metric in self.metrics.items():
            if name.startswith('test_'):
                self.log(f"{name}_epoch", metric.compute())
                metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        # optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer
    