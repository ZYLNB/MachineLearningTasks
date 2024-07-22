import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
import numpy as np

# 加载模型和tokenizer
def load_model(model_path, tokenizer_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

model, tokenizer = load_model("./cola_model", "./cola_tokenizer")

# 预测函数
def predict(texts, model, tokenizer, max_length=128):
    inputs = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
    return predictions

# 示例文本
texts = ["This is a great movie!"]
predictions = predict(texts, model, tokenizer)
print(predictions)