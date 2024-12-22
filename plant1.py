from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

model_name_or_path = "./plant_RNA/"

class CustomEsmForSequenceClassification(nn.Module):
    def __init__(self, model_name_or_path, num_labels):
        super(CustomEsmForSequenceClassification, self).__init__()
        self.esm = AutoModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Sequential(
            nn.Linear(self.esm.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.esm(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # 假设使用[CLS]标记的表示
        logits = self.classifier(pooled_output)
        return logits

num_labels = 2  # 根据您的任务设置标签数
model = CustomEsmForSequenceClassification(model_name_or_path, num_labels)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

