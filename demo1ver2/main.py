import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import swanlab
from collections import Counter

from dataset import NewsDataset
from model import BertWithDropout
from train import train_epoch, eval_epoch
from utils import load_data

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

config_path = "config.json"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"配置文件不存在：{config_path}")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_texts, train_labels = load_data(cfg['train_path'])
dev_texts, dev_labels = load_data(cfg['dev_path'])
test_texts, test_labels = load_data(cfg['test_path'])

with open("labels.json", "r", encoding="utf-8") as f:
    id2label = json.load(f)
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

train_labels = [label2id[l] for l in train_labels]
dev_labels = [label2id[l] for l in dev_labels]
test_labels = [label2id[l] for l in test_labels]

print("="*50)
print("查看前5条原始数据")
print("="*50)
for i in range(5):
    print(f"文本：{train_texts[i]}")
    print(f"标签：{train_labels[i]}")
    print("-"*30)

print("\n" + "="*50)
print("训练集标签分布统计")
print("="*50)
label_count = Counter(train_labels)
for l, cnt in label_count.most_common():
    label_name = id2label[l]
    print(f"标签 {l} ({label_name})：{cnt} 条")

tokenizer = BertTokenizer.from_pretrained(
    cfg['model_name'],
    local_files_only=True
)

def build_loader(texts, labels, shuffle):
    ds = NewsDataset(texts, labels, tokenizer, cfg['max_len'])
    return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle)

train_loader = build_loader(train_texts, train_labels, shuffle=True)
dev_loader = build_loader(dev_texts, dev_labels, shuffle=False)
test_loader = build_loader(test_texts, test_labels, shuffle=False)

swanlab.init(project="bert-news-classify", experiment_name="final")

model = BertWithDropout(
    cfg['model_name'],
    num_labels=cfg['num_classes'],
    dropout_rate=cfg['dropout_rate']
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
total_steps = len(train_loader) * cfg['epochs']
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=cfg['warmup_steps'],
    num_training_steps=total_steps
)
criterion = nn.CrossEntropyLoss()

best_dev_acc = 0

for epoch in range(cfg['epochs']):
    print(f"\nEpoch {epoch+1}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
    dev_loss, dev_acc, _, _ = eval_epoch(model, dev_loader, criterion, device)

    swanlab.log({
        "train/loss": train_loss, "train/acc": train_acc,
        "dev/loss": dev_loss, "dev/acc": dev_acc
    })

    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"已保存最佳模型，acc={dev_acc:.4f}")

test_loss, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)
print(f"Test acc: {test_acc:.4f}")

print("\n===== 详细分类结果 =====")
print(classification_report(
    test_labels, 
    test_preds, 
    target_names=list(id2label.values()),
    digits=4
))

swanlab.log({"test/loss": test_loss, "test/acc": test_acc})
swanlab.finish()