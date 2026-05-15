import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
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
        scheduler.step()

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