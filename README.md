# W2NER_predict
\[Unofficial\] Predict code for AAAI 2022 paper: Unified Named Entity Recognition as Word-Word Relation Classification

# 说明
该项目是对上述论文的一个复现，并添加了预测代码。以及部分注释。<br>
源论文：https://arxiv.org/pdf/2112.10070.pdf<br>
源代码：https://github.com/ljynlp/w2ner

# 步骤
## 1、安装以下依赖
```
- numpy (1.21.4)
- torch (1.10.0)
- gensim (4.1.2)
- transformers (4.13.0)
- pandas (1.3.4)
- scikit-learn (1.0.1)
- prettytable (2.4.0)
```

## 2、下载数据
这里以resume-zh为例，数据下载地址：<a href="https://drive.google.com/drive/folders/1NdvUeIUUL3mlS8QwwnqM628gCK7_0yPv?usp=sharing">link</a>。需要翻墙。<br>
下载好resume-zh将其放置在data文件夹下。

## 3、训练、验证及测试
执行训练语句：
```python
python main.py --config ./config/resume-zh.json
```
训练好后会在主目录下生成model.pt文件。结果：
```
......
2022-03-21 04:06:44 - INFO: Epoch: 9
2022-03-21 04:08:37 - INFO: 
+---------+--------+--------+-----------+--------+
| Train 9 |  Loss  |   F1   | Precision | Recall |
+---------+--------+--------+-----------+--------+
|  Label  | 0.0005 | 0.9762 |   0.9791  | 0.9735 |
+---------+--------+--------+-----------+--------+
2022-03-21 04:08:41 - INFO: EVAL Label F1 [0.99977967 0.98848258 0.99099099 1.         1.         0.95695839
 0.97695853 0.94339623 0.92307692 1.        ]
2022-03-21 04:08:41 - INFO: 
+--------+--------+-----------+--------+
| EVAL 9 |   F1   | Precision | Recall |
+--------+--------+-----------+--------+
| Label  | 0.9780 |   0.9661  | 0.9911 |
| Entity | 0.9577 |   0.9542  | 0.9613 |
+--------+--------+-----------+--------+
2022-03-21 04:08:46 - INFO: TEST Label F1 [0.99978522 0.98747049 0.99115044 1.         1.         0.95752896
 0.98666667 0.94429708 0.90410959 1.        ]
2022-03-21 04:08:46 - INFO: 
+--------+--------+-----------+--------+
| TEST 9 |   F1   | Precision | Recall |
+--------+--------+-----------+--------+
| Label  | 0.9771 |   0.9655  | 0.9905 |
| Entity | 0.9617 |   0.9602  | 0.9632 |
+--------+--------+-----------+--------+
2022-03-21 04:08:46 - INFO: Best DEV F1: 0.9597
2022-03-21 04:08:46 - INFO: Best TEST F1: 0.9621
2022-03-21 04:08:51 - INFO: TEST Label F1 [0.99978775 0.98756339 0.97391304 1.         1.         0.95696853
 0.98666667 0.95169946 0.89189189 1.        ]
2022-03-21 04:08:51 - INFO: 
+------------+--------+-----------+--------+
| TEST Final |   F1   | Precision | Recall |
+------------+--------+-----------+--------+
|   Label    | 0.9748 |   0.9614  | 0.9906 |
|   Entity   | 0.9621 |   0.9591  | 0.9650 |
+------------+--------+-----------+--------+
```

## 4、预测
预测的数据处理代码在data_loader.py里面，主运行代码在predict.py里面。执行指令：
```python
python predict.py --config ./config/resume-zh.json
```
预测结果：
```
texts = [
  "高勇，男，中国国籍，无境外居留权。",
  "常见量，男。"
]

[
  ('高勇，男，中国国籍，无境外居留权。', ('中国国籍', 'cont', 5, 8), ('高勇', 'name', 0, 1)), 
  ('常见量，男。', ('常见量', 'name', 0, 2))
]

```
最后需要注意的是源代码默认是使用gpu的，如果要切换的cpu，那么相关的地方还要进行修改。

# Acknowledgement
> https://github.com/ljynlp/w2ner

# 补充
[信息抽取三剑客：实体抽取、关系抽取、事件抽取](https://github.com/taishan1994/chinese_information_extraction)<br>
[基于机器阅读理解的命名实体识别](https://github.com/taishan1994/BERT_MRC_NER_chinese)<br>
[pytorch_bert_bilstm_crf命名实体识别](https://github.com/taishan1994/pytorch_bert_bilstm_crf_ner)<br>
