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
下载好resume-zh将其放置在data文件夹下。然后执行训练语句：
```python
python main.py --config ./config/resume-zh.json
```
训练好后会在主目录下生成model.pt文件。

## 3、预测
预测的数据处理代码在data_loader.py里面，主运行代码在predict.py里面。执行指令：
```python
python predict.py --config ./config/resume-zh.json
```
结果：
```
+------------+--------+-----------+--------+
| TEST Final |   F1   | Precision | Recall |
+------------+--------+-----------+--------+
|   Label    | 0.9748 |   0.9614  | 0.9906 |
|   Entity   | 0.9621 |   0.9591  | 0.9650 |
+------------+--------+-----------+--------+
```

# Acknowledgement
> https://github.com/ljynlp/w2ner


