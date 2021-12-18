# CCPM-baseline

本项目为北京大学2021年秋季学期计算语言学课程项目，古诗文识记数据集的模型代码

本方案将该任务建模成 SequenceClassification 任务, 利用 GuwenBERT 作为 backbone 进行 fine-tuning

训练集：21,778 条；验证集：2,720 条；测试集：2,720 条

数据集来自 BAAI CUGE benchmark:

> @article{li2021CCPM,
> 
> title = {CCPM: A Chinese Classical Poetry Matching Dataset},
> 
> author = {Li, Wenhao and Qi, Fanchao and Sun, Maosong and Yi, Xiaoyuan and Zhang, Jiarui},
> 
> journal={arXiv preprint arXiv:2106.01979},
> 
> year = {2021}
> }
## Co-occurrence Score
根据共现情况初步筛选候选诗句字共现性分析，对应代码 $co\_occurrence.py$
### translation-choice co-occurrence
翻译描述中应该包含全部正确诗句的内容，并与错误诗句的内容有偏差。
对于一条诗句，其字在翻译描述中的数量为其分数，记为$score_1$。
### choice-choice co-occurrence
一般的错误诗句会和正确诗句仅存在部分差别，导致正确诗句往往处于诗句分布密集处。可以用诗句选项间的共现性过滤掉与正确诗句偏差更大的错误选项。
对于一条诗句，其字在其他诗句选项中出现数量为其分数，记为$score_2$。

### choice-choice co-occurrence
最终共现得分为$\alpha*score_1 + score_2$。只保留最高得分的诗句作为候选集合。

## Dataset
将原始数据集切分为$\{translation, choice, if_correct\}$构成的数据集，对应代码 $data_parser.py$

## Model&Training
对应代码 $huggingface_guwen.py$

$gen\_dataset$函数根据原始数据集生成切分数据集，保存在data\下

创建模型 $modules = BlastFurnace(data_files)$

训练-保存模型 $modules.train\_module(num\_epochs=num\_epochs)$

测试模型 $modules.test\_module(test\_datasets)$