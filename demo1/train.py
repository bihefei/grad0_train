import os
# 镜像配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import swanlab
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# --------------------------
# 1. 配置参数
# --------------------------
class Config:
    model_name = "bert-base-chinese"
    train_path = "./data/train_3k.txt"
    dev_path = "./data/dev_1k.txt"
    test_path = "./data/test_1k.txt"
    max_len = 64
    batch_size = 16
    lr = 1e-5
    epochs = 5
    num_classes = 15
    dropout_rate = 0.5  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# --------------------------
# 2. 数据加载
# --------------------------
def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('_!_')
            if len(parts) >= 4:
                text = parts[3]
                label_name = parts[2]
                texts.append(text)
                labels.append(label_name)
    return texts, labels

train_texts, train_labels = load_data(config.train_path)
dev_texts, dev_labels = load_data(config.dev_path)
test_texts, test_labels = load_data(config.test_path)

all_labels = sorted(list(set(train_labels + dev_labels + test_labels)))
label2id = {label: i for i, label in enumerate(all_labels)}
id2label = {i: label for i, label in enumerate(all_labels)}

train_labels = [label2id[label] for label in train_labels]
dev_labels = [label2id[label] for label in dev_labels]
test_labels = [label2id[label] for label in test_labels]

# --------------------------
# 3. 加载模型
# --------------------------
tokenizer = BertTokenizer.from_pretrained(config.model_name)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = NewsDataset(train_texts, train_labels, tokenizer, config.max_len)
dev_dataset = NewsDataset(dev_texts, dev_labels, tokenizer, config.max_len)
test_dataset = NewsDataset(test_texts, test_labels, tokenizer, config.max_len)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# --------------------------
# 4. 初始化 SwanLab
# --------------------------
swanlab.init(
    project="bert-news-classify",
    experiment_name="run1",
    mode="cloud",  
    config={
        "model": config.model_name,
        "max_len": config.max_len,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "epochs": config.epochs,
        "num_classes": config.num_classes,
        "dropout_rate": config.dropout_rate,
        "device": str(config.device)
    }
)

# --------------------------
# 5. 模型
# --------------------------
class BertWithDropout(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # 返回和BertForSequenceClassification相同格式，保证代码不用改
        return type('Output', (object,), {'logits': logits})()

model = BertWithDropout(
    config.model_name,
    num_labels=config.num_classes,
    dropout_rate=config.dropout_rate
).to(config.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss()

# --------------------------
# 6. 训练
# --------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    preds_all = []
    labels_all = []
    for batch in tqdm(loader, desc="Train"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        out = model(input_ids, attention_mask=attention_mask)
        loss = criterion(out.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(out.logits, dim=1).cpu().numpy()
        preds_all.extend(preds)
        labels_all.extend(labels.cpu().numpy())

    return total_loss / len(loader), accuracy_score(labels_all, preds_all)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids, attention_mask=attention_mask)
            loss = criterion(out.logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(out.logits, dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(labels_all, preds_all), preds_all, labels_all

# --------------------------
# 开始跑
# --------------------------
for epoch in range(config.epochs):
    print(f"\n======== Epoch {epoch+1} ========")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, config.device)
    dev_loss, dev_acc, _, _ = eval_epoch(model, dev_loader, criterion, config.device)

    swanlab.log({
        "train/loss": train_loss,
        "train/acc": train_acc,
        "dev/loss": dev_loss,
        "dev/acc": dev_acc
    })

    print(f"Train loss: {train_loss:.4f} acc: {train_acc:.4f}")
    print(f"Dev loss: {dev_loss:.4f} acc: {dev_acc:.4f}")

# 测试集
test_loss, test_acc, test_pred, test_label = eval_epoch(model, test_loader, criterion, config.device)
print("\n===== 测试集结果 =====")
print(f"Test loss: {test_loss:.4f}")
print(f"Test acc: {test_acc:.4f}")

swanlab.log({"test/loss": test_loss, "test/acc": test_acc})
swanlab.finish()

# 保存当前训练好的模型
torch.save(model.state_dict(), "news_model7.pth")
print("模型已保存：news_model7.pth")