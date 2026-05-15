import torch
import torch.nn as nn
from transformers import BertModel

class BertWithDropout(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            model_name,
            local_files_only=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return type('Output', (object,), {'logits': logits})()