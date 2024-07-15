import os
import re
from tokenizers import BertWordPieceTokenizer
from transformers import BertConfig, BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader

def clean_text(text):
    # 简单的文本清理函数
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data_path = './Trump.txt'
cleaned_texts = []

# 读取并清理文本文件
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()
    cleaned_text = clean_text(text)
    cleaned_texts.append(cleaned_text)

# 将清理后的文本保存到文件中
cleaned_file_path = './cleaned_texts.txt'
with open(cleaned_file_path, 'w', encoding='utf-8') as file:
    for text in cleaned_texts:
        file.write(text + '\n')

# 初始化并训练分词器
tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=[cleaned_file_path], vocab_size=30522, min_frequency=2, limit_alphabet=1000, wordpieces_prefix='##')
tokenizer.save_model('./', 'bert-tokenizer')

# 创建 BERT 配置
config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12
)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'token_type_ids': inputs['token_type_ids'].squeeze()
        }

# 加载清理后的文本
with open(cleaned_file_path, 'r', encoding='utf-8') as file:
    texts = file.readlines()

# 初始化分词器
tokenizer = BertTokenizer.from_pretrained('d:/workshop/python/week3/bert-tokenizer-vocab.txt')

# 创建数据集和数据加载器
dataset = TextDataset(texts, tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 初始化模型
model = BertForMaskedLM(config=config)

# 定义优化器和调度器
num_epochs = 3  # 设定训练轮数
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 训练模型
model.train()

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=input_ids)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# 保存模型和分词器
model.save_pretrained('./model')
tokenizer.save_pretrained('./tokenizer')


