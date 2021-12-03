import logging
import os
import sys
from dataclasses import dataclass, field
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
from transformers import AutoModelForSequenceClassification
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
hugging_face_dir = os.path.join("/","var","data","qikahh","huggingface")
checkpoint = "hfl/chinese-roberta-wwm-ext"
output_dir = os.path.join("model_weights","roberta_rule")
batch_size = 128

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

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "labels"
        # print(features[0].keys())
        labels = None
        if label_name in features[0].keys():
            labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        batch_length = [
            max([len(v) for v in feature.values()]) for feature in flattened_features
        ]
        batch_length = max(batch_length)
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=min(self.max_length,batch_length),
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        if labels:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

#-------------data_loader-----------------
data_files = {"train": join('data','train_dichotomies.jsonl'), "valid": join('data','valid_dichotomies.jsonl')}
raw_datasets = load_dataset("json", data_files=data_files, cache_dir=hugging_face_dir)
test_data = get_dataset(join('data','test_public.jsonl'))
valid_use_data = get_dataset(join('data','valid.jsonl'))

tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=hugging_face_dir,model_max_length=512,block_size=512)
# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8, max_length=512)

ending_names = [f"ending{i}" for i in range(4)]
context_name = "translation"
choice_name = "choices"

def tokenize_function(example):
    return tokenizer(example["translation"], example["choice"], truncation=True, max_length=512)

# Preprocessing the datasets.
def preprocess_function(examples):
    translation = [[context] * 4 for context in examples[context_name]]
    classic_poetry = [
        [c for c in choices] for choices in examples[choice_name]
    ]

    # Flatten out
    first_sentences = sum(translation, [])
    second_sentences = sum(classic_poetry, [])

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=512,
    )
    results = {}
    results.update({k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()})
    results['labels'] = [ answer for answer in examples['answer']]
    # print(results)
    # Un-flatten
    return results 


def tokenize_test_function(example):
    choices = example["choices"]
    examples = []
    for choice in choices:
        examples.append(tokenizer(example["translation"], choice, truncation=True))
    example['features'] = examples
    return example

def save_model(model, tokenizer, model_path):
    model_to_save = model.module if hasattr(model, 'module') else model
    #如果使用预定义的名称保存，则可以使用`from_pretrained`加载
    output_model_file = os.path.join(model_path, WEIGHTS_NAME)
    output_config_file = os.path.join(model_path, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(model_path)

def load_model(model_path): 
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # Add specific options if needed
    return model, tokenizer

train_datasets = raw_datasets['train']
train_datasets = train_datasets.map(
                tokenize_function,
                batched=True,
            )
train_datasets = train_datasets.remove_columns(["choice", "translation"])
valid_datasets = raw_datasets['valid']
valid_datasets = valid_datasets.map(
                tokenize_function,
                batched=True,
            )
valid_datasets = valid_datasets.remove_columns(["choice", "translation"])
train_dataloader = DataLoader(
    train_datasets, shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    valid_datasets, batch_size=batch_size, collate_fn=data_collator
)

#-------------model-----------------
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, cache_dir=hugging_face_dir)
optimizer = AdamW(model.parameters(), lr=5e-5)
accelerator = Accelerator(fp16 = True)
device = accelerator.device
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)
num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# Metric
def compute_metrics(predictions, label_ids):
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

#-------------train-----------------

def train_module():
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            progress_bar.set_description('Epoch %i' % epoch)
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_postfix(loss=loss.mean().item())
            progress_bar.update(1)
        
        model.eval()
        metric = load_metric("accuracy", cache_dir=hugging_face_dir)
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        print(metric.compute())


#-------------regulation-----------------

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
    return np.array(scores)
    min_choice = argmin(scores)
    if len(min_choice) > 3:
        return choices_return
    for i in min_choice:
        choices_return[i] = ''
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
    return np.array(scores)

def sum_score(score1, score2):
    return score1+score2*2

#-------------test-----------------

def valid_module(model, tokenizer, valid_datasets):
    model = model.to(device)
    model.eval()
    metric = load_metric("accuracy", cache_dir=hugging_face_dir)
    new_dataset = []
    for example in tqdm(valid_datasets):
        samples = example['features']
        choices_score = choice_cooccurrence(example['choices'])
        translation_score = translation_cooccurrence(example['translation'], example['choices'])
        score = sum_score(choices_score,translation_score)
        myAns = [idx for idx, s in enumerate(score) if s == max(score)]
        batch_length = [
            max([len(v) for v in feature.values()]) for feature in samples
        ]
        batch_length = max(batch_length)
        batch = tokenizer.pad(
            samples,
            padding=True,
            max_length=batch_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch = {k: v.view(4, -1) for k, v in batch.items()}
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits[:,1]
            predictions = torch.softmax(logits, dim=-1)
            for idx in myAns:
                predictions[idx]+=1
            predictions = torch.argmax(predictions, dim=0)
            prediction = predictions.item()
        metric.add(prediction=prediction, reference=example["answer"])

    print(metric.compute())

def test_module(model, tokenizer, test_datasets):
    print("test")
    model = model.to(device)
    model.eval()
    new_dataset = []
    for example in tqdm(test_datasets):
        samples = example['features']
        choices_score = choice_cooccurrence(example['choices'])
        translation_score = translation_cooccurrence(example['translation'], example['choices'])
        score = sum_score(choices_score,translation_score)
        myAns = [idx for idx, s in enumerate(score) if s == max(score)]
        batch_length = [
            max([len(v) for v in feature.values()]) for feature in samples
        ]
        batch_length = max(batch_length)
        batch = tokenizer.pad(
            samples,
            padding=True,
            max_length=batch_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch = {k: v.view(4, -1) for k, v in batch.items()}
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits[:,1]
            predictions = torch.softmax(logits, dim=-1)
            for idx in myAns:
                predictions[idx]+=1
            predictions = torch.argmax(predictions, dim=0)
        example['answer'] = predictions.item()
        example.pop('features')
        new_dataset.append(example)

    save_dataset(new_dataset, join('data','CCPM.jsonl'))

if __name__ == "__main__":
    train_module()
    save_model(model, tokenizer, model_path=output_dir)
    model, tokenizer = load_model(model_path=output_dir)
    test_datasets = []
    for example in test_data:
        test_datasets.append(tokenize_test_function(example))
    valid_use_datasets = []
    for example in valid_use_data:
        valid_use_datasets.append(tokenize_test_function(example))
    valid_module(model, tokenizer, valid_use_datasets)
    test_module(model, tokenizer, test_datasets)