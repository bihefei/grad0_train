# demo1ver2
基于BERT的今日头条新闻文本分类模型

## 📌 项目简介
本项目使用预训练语言模型BERT实现今日头条新闻文本分类任务，对新闻标题进行自动分类，共覆盖15个新闻类别。

## 数据集来源
https://pan.baidu.com/s/10XRGQAIKGDI5eWLjmaB9Xg?pwd=1111#list/path=%2F

## 📁 项目文件说明
| 文件名 | 作用说明 |
| :--- | :--- |
| `main.py` | 项目主入口，负责启动训练、验证与测试流程 |
| `model.py` | BERT分类模型结构定义 |
| `dataset.py` | 数据集加载与预处理，包括数据读取、标签映射、批处理 |
| `train.py` | 训练与验证逻辑，包含模型训练、评估、最优模型保存 |
| `utils.py` | 工具函数，如日志打印、结果保存等 |
| `config.json` | 训练超参数配置文件（学习率、批次大小、序列长度等） |
| `labels.json` | 标签映射字典，存储类别与数字标签的对应关系 |
| `requirements.txt` | 项目依赖包清单，可通过`pip install -r requirements.txt`一键安装 |
| `test_result.txt` | 本次实验的测试结果、训练过程与Swanlab可视化链接 |
| `data/` | 数据集文件夹，存放训练、验证、测试文本数据 |

## 实验结果
- 最优验证集准确率：82.90%
- 测试集准确率：82.52%
- 训练过程可视化：[Swanlab链接](https://swanlab.cn/@sunfeifei/bert-news-classify/runs/ihiay28rdpt9iodxk7u00)

## 使用方法
1. 安装依赖：`pip install -r requirements.txt`
2. 运行主程序：`python main.py`