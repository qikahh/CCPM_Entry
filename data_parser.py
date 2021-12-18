import json
from os.path import join
import jsonlines
from tqdm import tqdm
import re
from co_occurrence import get_co_subset

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

# make data as:
# [translation, choice, if_correct]
def gen_choice_dataset(dataset, data_name):
    new_dataset = []
    for data in tqdm(dataset):
        choices = data['choices']
        for idx,choice in enumerate(choices):
            new_data = {'translation': data['translation'], 'choice': choice, 'labels': int(idx == data['answer'])}
            new_dataset.append(new_data)
    save_dataset(new_dataset, join('data','{}.jsonl'.format(data_name)))

def split_verse(verse):
    verse_list = re.split('[，。]', verse)
    entry_list = []
    for verse in verse_list:
        if len(verse) == 5:
            entry_list.append(verse[:2])
            entry_list.append(verse[2:])
        elif len(verse) == 7:
            entry_list.append(verse[:2])
            entry_list.append(verse[2:4])
            entry_list.append(verse[4:])
        else:
            entry_list.append(verse)
    return entry_list

def gen_entries_dataset(dataset, data_name):
    for data in tqdm(dataset):
        choices = data['choices']
        data['split_choices'] = []
        for choice in choices:
            tokens = split_verse(choice)
            data['split_choices'].append(tokens)
    save_dataset(dataset, join('data','{}.jsonl'.format(data_name)))

