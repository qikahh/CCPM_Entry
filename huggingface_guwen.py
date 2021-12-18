import logging
import os
import sys
from dataclasses import dataclass, field
from types import coroutine
from typing import Optional, Union
from tqdm.auto import tqdm

from co_occurrence import get_co_subset
from data_parser import get_dataset, save_dataset, gen_choice_dataset

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from accelerate import Accelerator

import transformers
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import AdamW
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from transformers import AutoTokenizer


import json
from os.path import join
import jsonlines

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
hugging_face_dir = "huggingface" # 本地huggingface保存路径
checkpoint = "ethanyt/guwenbert-base" 
#checkpoint = "hfl/chinese-roberta-wwm-ext"
output_dir = os.path.join("model_weights", "guwenbert-base")
use_local = False # Whether to use locally saved checkpoint in output_dir
data_files = {"train": join('data','train.jsonl'), "valid": join('data','valid.jsonl')}
batch_size = 256
num_epochs = 30

def change_roberta_to_bert(model):
    try:
        o_embedding = model.roberta.embeddings.token_type_embeddings
        n_embedding = torch.nn.Embedding(2,768)
        n_embedding.weight = torch.nn.Parameter(o_embedding.weight.repeat(2,1))
        model.roberta.embeddings.token_type_embeddings = n_embedding
        print("change roberta to bert")
    except Exception as e:
        print("[ERROR] change roberta to bert(" + str(e) + ")")
    return model

class BlastFurnace():
    def __init__(self, data_files):
        self.raw_datasets = load_dataset("json", data_files=data_files, cache_dir=hugging_face_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, cache_dir=hugging_face_dir)
        self.model = change_roberta_to_bert(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=hugging_face_dir,model_max_length=512,block_size=512)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, pad_to_multiple_of=8, max_length=512)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        def tokenize_function(example):
            return self.tokenizer(example["translation"], example["choice"], truncation=True, max_length=512)
        self.train_datasets = self.raw_datasets['train'].map(
                        tokenize_function,
                        batched=True,
                    )
        self.train_datasets = self.train_datasets.remove_columns(["choice", "translation"])
        self.valid_datasets = self.raw_datasets['valid'].map(
                        tokenize_function,
                        batched=True,
                    )
        self.valid_datasets = self.valid_datasets.remove_columns(["choice", "translation"])
        self.train_dataloader = DataLoader(
            self.train_datasets, shuffle=True, batch_size=batch_size, collate_fn=self.data_collator
            , 
        )
        self.valid_dataloader = DataLoader(
            self.valid_datasets, batch_size=batch_size, collate_fn=self.data_collator
        )
        self.accelerator = Accelerator(fp16 = True)
        self.device = self.accelerator.device
        self.train_dataloader, self.valid_dataloader, self.model, self.optimizer = self.accelerator.prepare(
        self.train_dataloader, self.valid_dataloader, self.model, self.optimizer
        )

    def save_model(self, model_path):
        torch.save(self.model.state_dict(),join(model_path,"state_dict_model.pth"))
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        output_model_file = os.path.join(model_path, WEIGHTS_NAME)
        output_config_file = os.path.join(model_path, CONFIG_NAME)

        model_to_save.config.to_json_file(output_config_file)
        self.tokenizer.save_vocabulary(model_path)

    def load_model(self, model_path): 
        self.model.load_state_dict(torch.load(join(model_path,"state_dict_model.pth")))
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # Add specific options if needed

    def train_module(self, num_epochs=10):
        self.model.train()
        num_training_steps = num_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            self.model.train()
            progress_bar = tqdm(range(len(self.train_dataloader)))
            for batch in self.train_dataloader:
                progress_bar.set_description('Epoch %i' % epoch)
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
    
                progress_bar.set_postfix(loss=loss.mean().item())
                progress_bar.update(1)
            
            self.model.eval()
            metric = load_metric("accuracy", cache_dir=hugging_face_dir)
            for batch in self.valid_dataloader:
                with torch.no_grad():
                    outputs = self.model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])
            metric_value = metric.compute()
            accuracy = metric_value['accuracy']
            print("valid acc: ", accuracy)
            if accuracy > best_accuracy:
                modules.save_model(output_dir)
                best_accuracy = accuracy

    def tokenize_test_function(self, example):
        choices = example["choices"]
        features = []
        for choice in choices:
            features.append(self.tokenizer(example["translation"], choice, truncation=True))
        batch_length = [
                max([len(v) for v in feature.values()]) for feature in features
            ]
        batch_length = max(batch_length)
        batch = self.tokenizer.pad(
                features,
                padding=True,
                max_length=batch_length,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
        example["features"] = batch
        return example

    def valid_module(self, dataset):
        self.model.eval()
        metric = load_metric("accuracy", cache_dir=hugging_face_dir)
        for example in tqdm(dataset):
            batch = example['features']
            batch = {k: v.to(self.device) for k, v in batch.items()}
            co_set = get_co_subset(example['translation'], example['choices'])
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits[:,1]
                predictions = torch.softmax(logits, dim=-1)
                for idx in co_set:
                    predictions[idx]+=10
                predictions = torch.argmax(predictions, dim=0)
                prediction = predictions.item()
            metric.add(prediction=prediction, reference=example['answer'])
        metric_value = metric.compute()
        print(metric_value)

    def test_module(self, dataset):
        new_dataset = []
        self.model.eval()
        for example in tqdm(dataset):
            new_example = {}
            batch = example['features']
            batch = {k: v.to(self.device) for k, v in batch.items()}
            co_set = get_co_subset(example['translation'], example['choices'])
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits[:,1]
                predictions = torch.softmax(logits, dim=-1)
                for idx in co_set:
                    predictions[idx]+=10
                predictions = torch.argmax(predictions, dim=0)
                prediction = predictions.item()
            new_example['answer'] = prediction
            new_example['translation'] = example['translation']
            new_example['choices'] = example['choices']
            new_dataset.append(new_example)
        save_dataset(new_dataset, join('result','CCPM.jsonl'))

def gen_dataset():
    train_data = get_dataset(join('raw_data','train.jsonl'))
    val_data = get_dataset(join('raw_data','valid.jsonl'))
    raw_data = {'train': train_data, 'valid': val_data}
    for key, dataset in raw_data.items():
        gen_choice_dataset(dataset, key)

if __name__ == '__main__':
    gen_dataset()

    modules = BlastFurnace(data_files)
    modules.train_module(num_epochs=num_epochs)

    if use_local:
        modules.load_model(output_dir)
    valid_use_data = get_dataset(join('raw_data','valid.jsonl'))
    valid_use_datasets = []
    for example in valid_use_data:
        valid_use_datasets.append(modules.tokenize_test_function(example))
    modules.valid_module(valid_use_datasets)

    test_data = get_dataset(join('raw_data','test_public.jsonl'))
    test_datasets = []
    for example in test_data:
        test_datasets.append(modules.tokenize_test_function(example))
    modules.test_module(test_datasets)