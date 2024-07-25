import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
import numpy as np

# 任务和模型设置
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "mrpc"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

# 数据预处理
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[task]

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, padding='max_length', max_length=128)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding='max_length', max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# 自定义Dataset类来适配DataLoader
class GlueDataset(Dataset):
    def __init__(self, encoded_dataset):
        self.dataset = encoded_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # 打印调试信息
        # print(f"Raw item: {item}")
        # 将所有特征转换为PyTorch张量
        item = {key: torch.tensor(val) for key, val in item.items() if isinstance(val, (int, float, list))}
        # 单独处理标签
        if 'label' in item:
            item['labels'] = torch.tensor(item.pop('label'))
        # 移除 idx 字段，避免传递到模型 forward 方法中
        if 'idx' in item:
            item.pop('idx')
        return item

train_dataset = GlueDataset(encoded_dataset['train'])
validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
eval_dataset = GlueDataset(encoded_dataset[validation_key])

# 自定义collate_fn函数
def collate_fn(batch):
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}

# 数据加载
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)

# 模型设置
num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
model.to(device)

# 训练设置
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# 训练循环
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# 评估
model.eval()
all_predictions = []
all_labels = []
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    
    predictions = outputs.logits.argmax(dim=-1) if task != "stsb" else outputs.logits[:, 0]
    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(batch["labels"].cpu().numpy())

results = metric.compute(predictions=np.array(all_predictions), references=np.array(all_labels))
print("eval:", results)

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification
from datasets import load_dataset

# 定义超参数搜索空间
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-5),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
    }

# 定义模型初始化函数
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# 定义训练参数
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 创建 Trainer 对象
trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=lambda p: {'accuracy': (p.predictions.argmax(-1) == p.label_ids).mean()}
)

# 进行超参数搜索
best_run = trainer.hyperparameter_search(
    n_trials=2,
    direction="maximize",
    hp_space=hp_space
)

# 打印最佳超参数组合
print(best_run)

# 将最佳超参数设置到 trainer.args 中
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

# 训练模型
trainer.train()

# 保存模型和tokenizer
model.save_pretrained("mrpc_model")
tokenizer.save_pretrained("mrpc_tokenizer")

