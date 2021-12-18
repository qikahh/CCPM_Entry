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

# choice-choice co-occurrence
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

# translation-choice co-occurrence 
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

def sum_score(choices_score, translation_score, alpha = 1.0):
    return choices_score+translation_score*alpha

def get_co_subset(translation, choices):
    choices_score = choice_cooccurrence(choices)
    translation_score = translation_cooccurrence(translation, choices)
    score = sum_score(choices_score, translation_score)
    myAns = [idx for idx, s in enumerate(score) if s == max(score)]
    return myAns

def get_cooccurrence(dataset):
    wrong_list = [[] for _ in range(5)]
    muti_list = []
    nums = [0 for _ in range(5)]
    acc_nums = [0 for _ in range(5)]
    acc_num = 0
    num = 0
    for e in dataset:
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