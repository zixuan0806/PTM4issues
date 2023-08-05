import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torchmetrics

import torch

from transformers import BertTokenizer, BertModel

from ..metrics.accuracy import MultiLabelAccuracy
from ..metrics.precision import MultiLabelPrecision
from ..metrics.recall import MultiLabelRecall


class Bert(pl.LightningModule):
    def __init__(self, num_classes: int, model_name:str='bert-base-uncased'):
        super().__init__()
        self.class_num = num_classes

        self.bert = BertModel.from_pretrained(model_name)
        self.hid_dim = self.bert.config.hidden_size
        self.fc = nn.Linear(self.hid_dim, self.class_num, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.loss = nn.BCELoss()

        self.metrics = nn.ModuleDict()
        stage_name = ['train', 'valid', 'test']
        for stage in stage_name:
            for k in range(1, 3):
                self.metrics[f"{stage}_acc_{k}"] = MultiLabelAccuracy(top_k=k)
                self.metrics[f"{stage}_precision_{k}"] = MultiLabelPrecision(top_k=k)
                self.metrics[f"{stage}_recall_{k}"] = MultiLabelRecall(top_k=k)
                # self.metrics[f"{stage}_f1_{k}"] = torchmetrics.F1(average=avg, num_classes=num_classes, top_k=k)

    def forward(self, input_ids):
        # in lightning, forward defines the prediction/inference actions
        output = self.bert(input_ids)
        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        x = self.dropout(pooler_output)
        logits = torch.sigmoid(self.fc(x))
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
        return optimizer
