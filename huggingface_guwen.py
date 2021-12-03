import logging
import os
import sys
from dataclasses import dataclass, field
from types import coroutine
from typing import Optional, Union
from tqdm.auto import tqdm

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from accelerate import Accelerator

import transformers
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import BertTokenizer
from transformers import AutoModel
from transformers import AdamW
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
from transformers import get_scheduler
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

import json
from os.path import join
import jsonlines

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
hugging_face_dir = os.path.join("/","var","data","qikahh","huggingface")
checkpoint = "ethanyt/guwenbert-base"
#checkpoint = "hfl/chinese-roberta-wwm-ext"
output_dir = os.path.join("model_weights", "guwenbert-entry")
batch_size = 256
num_epochs = 30

import json
from os.path import join
import jsonlines

def get_dataset(file_path):
    dataset = []
    with open(file_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            dataset.append(item)
    return dataset

def save_dataset(dataset, file_path):
    with open(file_path, "w+", encoding="utf8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

import numpy as np

def argmax(scores):
    max_score = max(scores)
    return [i for i, s in enumerate(scores) if s == max_score]
def argmin(scores):
    min_id = []
    min_score = min(scores)
    for i, s in enumerate(scores):
        if s == min_score:
            min_id.append(i)
    return min_id
def choice_cooccurrence(choices):
    num = len(choices)
    choices_return = []
    scores = [0] * num
    choice_tokens = [[] for _ in range(num)]
    for i, choice in enumerate(choices):
        choices_return.append(choice)
        for c in choice:
            if c in choice_tokens[i]:
                continue
            choice_tokens[i].append(c)
    for i in range(num):
        for j in range(num):
            if i == j:
                continue
            for c in choice_tokens[i]:
                if c in choice_tokens[j]:
                    scores[i] += 1
    for i in range(num):
        if len(choices[i]) > 0:
            scores[i]/=len(choices[i])
    return np.array(scores)  
def translation_cooccurrence(translation, choices):
    scores = [0] * len(choices)
    for i, choice in enumerate(choices):
        choice_tokens = []
        for c in choice:
            if c in choice_tokens:
                continue
            if c in translation:
                scores[i] += 1
            choice_tokens.append(c)
    for i in range(len(scores)):
        if len(choices[i]) > 0:
            scores[i]/=len(choices[i])
    return np.array(scores)
def sum_score(scores1, scores2):
    return scores1+scores2*2
def get_cooccurrence(val_data):
    wrong_list = [[] for _ in range(5)]
    muti_list = []
    nums = [0 for _ in range(5)]
    acc_nums = [0 for _ in range(5)]
    acc_num = 0
    num = 0

    for e in val_data:
        scores_choice_co = choice_cooccurrence(e['choices_entries'])
        scores_translation_co = translation_cooccurrence(e['translation'], e['choices_entries'])

        scores = scores_choice_co+scores_translation_co
        max_score = max(scores)
        max_idx = [idx for idx, s in enumerate(scores) if s == max_score]
        if scores[e['answer']] == max_score:
            acc_nums[len(max_idx)] += 1
        else:
            wrong_list[len(max_idx)].append((e,max_idx))
        nums[len(max_idx)] += 1
    #print(acc_num/len(val_data))
    for i in range(5):
        print(f'{i}:{acc_nums[i]}/{nums[i]}')
        if nums[i] == 0:
            print(f'{i}:{0}/{0}')
        else:
            print(acc_nums[i] / nums[i])

def get_entries(example):
    entries = []
    split_choices = example['split_choices']
    example['choices_entries'] = []
    for split_choice in split_choices:
        choice_entries = []
        for tokens in split_choice:
            if len(tokens) > 0 and tokens not in entries:
                entries.append(tokens)
            if tokens in entries:
                choice_entries.append(tokens)
        example['choices_entries'].append(choice_entries)
    example['entries'] = entries
    return example
def make_entries_dataset(dataset):
    entries_dataset = []
    for example in tqdm(dataset):
        answer = example['answer']
        for idx,choice in enumerate(example['choices']):
            continue
            new_data = {}
            new_data['translation'] = example['translation']
            new_data['choice'] = choice
            new_data['answer'] = int(answer == idx)
            entries_dataset.append(new_data)
        for entry in example['entries']:
            if len(entry) > 3:
                continue
            new_data = {}
            new_data['translation'] = example['translation']
            new_data['choice'] = entry
            new_data['answer'] = int(entry in example['choices_entries'][answer])
            entries_dataset.append(new_data)
    return entries_dataset
def gen_entries_dataset():
    train_data = get_dataset(join('data','train_split.jsonl'))
    val_data = get_dataset(join('data','valid_split.jsonl'))
    test_data = get_dataset(join('data','test_public.jsonl'))

    train_data = list(map(get_entries, train_data))
    val_data = list(map(get_entries, val_data))

    train_entries_dataset = make_entries_dataset(train_data)
    val_entries_dataset = make_entries_dataset(val_data)

    save_dataset(train_entries_dataset, join('data','train_entries.jsonl'))
    save_dataset(val_entries_dataset, join('data','valid_entries.jsonl'))

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
    def __init__(self):
        data_files = {"train": join('data','train_entries.jsonl'), "valid": join('data','valid_entries.jsonl')}
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
        self.train_datasets = self.train_datasets.rename_column("answer", "labels")
        self.valid_datasets = self.raw_datasets['valid'].map(
                        tokenize_function,
                        batched=True,
                    )
        self.valid_datasets = self.valid_datasets.remove_columns(["choice", "translation"])
        self.valid_datasets = self.valid_datasets.rename_column("answer", "labels")
        self.train_dataloader = DataLoader(
            self.train_datasets, shuffle=True, batch_size=batch_size, collate_fn=self.data_collator
            , 
        )
        self.valid_dataloader = DataLoader(
            self.valid_datasets, batch_size=batch_size, collate_fn=self.data_collator
        )
        self.accelerator = Accelerator(fp16 = True)
        self.device = self.accelerator.device

    def save_model(self, model_path):
        #self.model = change_bert_to_roberta(self.model)
        torch.save(self.model.state_dict(),join(model_path,"state_dict_model.pth"))
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        #如果使用预定义的名称保存，则可以使用`from_pretrained`加载
        output_model_file = os.path.join(model_path, WEIGHTS_NAME)
        output_config_file = os.path.join(model_path, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        self.tokenizer.save_vocabulary(model_path)

    def load_model(self, model_path): 
        self.model.load_state_dict(torch.load(join(model_path,"state_dict_model.pth")))
        #self.model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir=hugging_face_dir)
        #Sself.model = change_roberta_to_bert(self.model)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # Add specific options if needed

    def train_module(self, num_epochs=10):
        self.model.train()
        self.train_dataloader, self.valid_dataloader, self.model, self.optimizer = self.accelerator.prepare(
        self.train_dataloader, self.valid_dataloader, self.model, self.optimizer
        )
        num_training_steps = num_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(num_epochs):
            self.model.train()
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
            print(metric.compute())

    def tokenize_test_function(self,example):
        choices = example["choices"]
        features = []
        for choice in choices:
            features.append(self.tokenizer(example["translation"], choice, truncation=True))
        example['features'] = features
        if 'entries' in example:
            entries_features = []
            for entry in example['entries']:
                entries_features.append(self.tokenizer(example['translation'], entry, truncation=True))
            example['entries_features'] = entries_features
        return example

    def valid_module(self, dataset):
        self.model.eval()
        metric = load_metric("accuracy", cache_dir=hugging_face_dir)
        for example in tqdm(dataset):
            samples = example['features']
            label = example['answer']
            choices_score = choice_cooccurrence(example['choices'])
            translation_score = translation_cooccurrence(example['translation'], example['choices'])
            score = sum_score(choices_score,translation_score)
            myAns = [idx for idx, s in enumerate(score) if s == max(score)]
            batch_length = [
                max([len(v) for v in feature.values()]) for feature in samples
            ]
            batch_length = max(batch_length)
            batch = self.tokenizer.pad(
                samples,
                padding=True,
                max_length=batch_length,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            batch = {k: v.view(4, -1) for k, v in batch.items()}
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if 'entries_features' in example:
                entries_features = example['entries_features']
                batch_length = [
                    max([len(v) for v in feature.values()]) for feature in entries_features
                ]
                batch_length = max(batch_length)
                entries_batch = self.tokenizer.pad(
                    entries_features,
                    padding=True,
                    max_length=batch_length,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
                entries_batch = {k: v.to(self.device) for k, v in entries_batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits[:,1]
                predictions = torch.softmax(logits, dim=-1)
                for idx in myAns:
                    predictions[idx]+=10
                if 'entries_features' in example:
                    entries_outputs = self.model(**entries_batch)
                    entries_logits = entries_outputs.logits
                    entries_predictions = torch.softmax(entries_logits, dim=-1)
                    for idx, entry in enumerate(example['entries']):
                        if len(entry) == 1:
                            continue
                        entry_score = (entries_predictions[idx,1] - entries_predictions[idx,0])/2.0
                        for jdx, choice in enumerate(example['choices']):
                            if entry in choice:
                                predictions[jdx]+=entry_score*(len(entry)/(1.0*len(choice)))*0
                predictions = torch.argmax(predictions, dim=0)
                prediction = predictions.item()
            metric.add(prediction=prediction, reference=label)
        print(metric.compute())

    def test_module(self, dataset):
        new_dataset = []
        self.model.eval()
        metric = load_metric("accuracy", cache_dir=hugging_face_dir)
        for example in tqdm(dataset):
            new_example = {}
            samples = example['features']
            choices_score = choice_cooccurrence(example['choices'])
            translation_score = translation_cooccurrence(example['translation'], example['choices'])
            score = sum_score(choices_score,translation_score)
            myAns = [idx for idx, s in enumerate(score) if s == max(score)]
            batch_length = [
                max([len(v) for v in feature.values()]) for feature in samples
            ]
            batch_length = max(batch_length)
            batch = self.tokenizer.pad(
                samples,
                padding=True,
                max_length=batch_length,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            batch = {k: v.view(4, -1) for k, v in batch.items()}
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if 'entries_features' in example:
                entries_features = example['entries_features']
                batch_length = [
                    max([len(v) for v in feature.values()]) for feature in entries_features
                ]
                batch_length = max(batch_length)
                entries_batch = self.tokenizer.pad(
                    entries_features,
                    padding=True,
                    max_length=batch_length,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
                entries_batch = {k: v.to(self.device) for k, v in entries_batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits[:,1]
                predictions = torch.softmax(logits, dim=-1)
                for idx in myAns:
                    predictions[idx]+=10
                if 'entries_features' in example:
                    entries_outputs = self.model(**entries_batch)
                    entries_logits = entries_outputs.logits
                    entries_predictions = torch.softmax(entries_logits, dim=-1)
                    for idx, entry in enumerate(example['entries']):
                        if len(entry) == 1:
                            continue
                        entry_score = (entries_predictions[idx,1] - entries_predictions[idx,0])/2.0
                        for jdx, choice in enumerate(example['choices']):
                            if entry in choice:
                                predictions[jdx]+=entry_score*(len(entry)/(1.0*len(choice)))*0
                predictions = torch.argmax(predictions, dim=0)
                prediction = predictions.item()
            new_example['answer'] = predictions.item()
            new_example['translation'] = example['translation']
            new_example['choices'] = example['choices']
            new_dataset.append(new_example)
        save_dataset(new_dataset, join('data','CCPM.jsonl'))

if __name__ == '__main__':
    gen_entries_dataset()
    modules = BlastFurnace()
    modules.train_module(num_epochs=num_epochs)
    modules.save_model(output_dir)
    modules.load_model(output_dir)
    valid_use_data = get_dataset(join('data','valid_split.jsonl'))
    valid_use_datasets = []
    for example in valid_use_data:
        example = get_entries(example)
        valid_use_datasets.append(modules.tokenize_test_function(example))
    modules.valid_module(valid_use_datasets)
    test_data = get_dataset(join('data','test_split.jsonl'))
    test_datasets = []
    for example in test_data:
        test_datasets.append(modules.tokenize_test_function(example))
    modules.test_module(test_datasets)


