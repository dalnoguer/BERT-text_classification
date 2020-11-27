import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, random_split
import pytorch_lightning as pl
import transformers

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from dataloader import TextClassificationDataset


class TextClassificationModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self._config = config

        self._num_classes = self._config['Dataset']['num_classes']

        self.save_hyperparameters()

        BertConfig = transformers.DistilBertConfig(**self._config['Model'])
        self._bert = transformers.DistilBertModel.from_pretrained(
            'distilbert-base-uncased', config=BertConfig)

        self._pre_classifier = torch.nn.Linear(self._bert.config.hidden_size,
                                               self._bert.config.hidden_size)
        self._classifier = torch.nn.Linear(
            self._bert.config.hidden_size,
            self._config['Dataset']['num_classes'])
        self._dropout = torch.nn.Dropout(self._bert.config.seq_classif_dropout)

        self._relu = torch.nn.ReLU()

        self._tokenizer = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased')

    def forward(self, input_ids, attention_mask, labels):

        output_seq = self._bert(input_ids=input_ids,
                                attention_mask=attention_mask)

        hidden_state = output_seq[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self._pre_classifier(pooled_output)
        pooled_output = self._relu(pooled_output)
        pooled_output = self._dropout(pooled_output)
        logits = self._classifier(pooled_output)

        return logits

    def training_step(self, batch, batch_nb):

        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        y_pred = self(input_ids, attention_mask, label)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_pred.view(-1, self._num_classes), label.view(-1))

        tensorboard_logs = {
            'train_loss': loss,
            'learn_rate': self._optimizer.param_groups[0]['lr']
        }

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):

        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        y_pred = self(input_ids, attention_mask, label)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_pred.view(-1, self._num_classes), label.view(-1))

        _, y_pred = torch.max(y_pred, dim=1)
        val_acc = accuracy_score(y_pred.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)

        tensorboard_logs = {'val_loss': loss, 'val_acc': val_acc}

        return {
            'val_loss': loss,
            'val_acc': val_acc,
            'progress_bar': tensorboard_logs
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_val_acc}
        return {
            'val_loss': avg_loss,
            'progress_bar': tensorboard_logs,
            'log': tensorboard_logs
        }

    def on_batch_end(self):
        if self._scheduler is not None:
            self._scheduler.step()

    def on_epoch_end(self):
        if self._scheduler is not None:
            self._scheduler.step()

    def test_step(self, batch, batch_nb):
        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        y_pred = self(input_ids, attention_mask, label)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_pred.view(-1, self._num_classes), label.view(-1))

        _, y_pred = torch.max(y_pred, dim=1)
        test_acc = accuracy_score(y_pred.cpu(), label.cpu())

        return {'test_loss': loss, 'test_acc': torch.tensor(test_acc)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {
            'avg_test_loss': avg_loss,
            'avg_test_acc': avg_test_acc
        }
        return {
            'avg_test_acc': avg_test_acc,
            'log': tensorboard_logs,
            'progress_bar': tensorboard_logs
        }

    def configure_optimizers(self):

        self._optimizer = torch.optim.Adam(
            params=[p for p in self.parameters() if p.requires_grad],
            lr=float(self._config['Training']['optimizer']['lr']),
            eps=float(self._config['Training']['optimizer']['eps']))

        self._scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self._optimizer,
            max_lr=float(self._config['Training']['scheduler']['max_lr']),
            total_steps=self._config['Training']['scheduler']['total_steps'])
        return [self._optimizer], [self._scheduler]

    def train_dataloader(self):

        df_train = pd.read_csv(
            os.path.join(self._config['Dataset']['data_dir'], 'train.csv'))
        train_dataset = TextClassificationDataset(notes=df_train['reviews'],
                                                  targets=df_train['target'],
                                                  tokenizer=self._tokenizer,
                                                  max_len=500)

        return DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=self._config['Training']['batch_size'],
                          num_workers=self._config['Training']['num_workers'])

    def val_dataloader(self):

        df_val = pd.read_csv(
            os.path.join(self._config['Dataset']['data_dir'], 'valid.csv'))
        val_dataset = TextClassificationDataset(notes=df_val['reviews'],
                                                targets=df_val['target'],
                                                tokenizer=self._tokenizer,
                                                max_len=500)

        return DataLoader(val_dataset,
                          batch_size=self._config['Training']['batch_size'],
                          num_workers=self._config['Training']['num_workers'])

    def test_dataloader(self):

        df_test = pd.read_csv(
            os.path.join(self._config['Dataset']['data_dir'], 'test.csv'))
        test_dataset = TextClassificationDataset(notes=df_test['reviews'],
                                                 targets=df_test['target'],
                                                 tokenizer=self._tokenizer,
                                                 max_len=500)

        return DataLoader(test_dataset,
                          batch_size=self._config['Training']['batch_size'],
                          num_workers=self._config['Training']['num_workers'])