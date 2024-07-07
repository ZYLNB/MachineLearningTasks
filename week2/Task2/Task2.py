import os
import random
import pandas as pd
#导入pytorch
import torch
#导入进度条
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# data prarameters
# 用于数据处理时的参数
concat_nframes = 1              # 要连接的帧数,n必须为奇数（总共2k+1=n帧）
train_ratio = 0.8               # 用于训练的数据比率，其余数据将用于验证
seed = 0
# 第一个模型的参数
batch_size = 512                # 批次数目
learning_rate = 0.0001          # 学习率


# 定义导入feature函数
def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)
# 将前后的特征联系在一起，如concat_n = 11 则前后都接5
def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n 必须是奇数
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n) 
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

# 数据预处理函数
def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41 # NOTE: 预先计算，不需要更改
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
      phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

      for line in phone_file:
          line = line.strip('\n').split(' ')
          label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # 分割训练和验证数据
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
      y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
          y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode != 'test':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
      print(y.shape)
      return X, y
    else:
      return X
# 返回的X代表数据的维度，如果不链接则为39 如果链接即为n*39 n为连接的特征总数,y为标签

import torch
#导入数据集
from torch.utils.data import Dataset
#导入数据加载工具Dataloader
from torch.utils.data import DataLoader
#定义数据集，一个数据集类应该包含初始化，_getitem__（获取一个元素）以及__len__（获取数据长度）方法
class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

import numpy as np

# 固定随机种子
def same_seeds(seed): # 固定随机种子（CPU）
    torch.manual_seed(seed) # 固定随机种子（GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True # 固定网络结构

# 固定随机种子
same_seeds(seed)

# 引入gc模块进行垃圾回收
import gc

# 预处理数据
train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)

# 将数据导入
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# 删除原始数据以节省内存
del train_X, train_y, val_X, val_y
gc.collect()

# 利用dataloader加载数据
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

class SimpleNN1(nn.Module):
    # 更深的模型
    def __init__(self, input_size, num_classes):
        super(SimpleNN1, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

        # 添加drop out
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc5(out)
        return out
    
class SimpleNN2(nn.Module):
    # 更浅的模型
    def __init__(self, input_size, num_classes):
        super(SimpleNN2, self).__init__()
        self.fc1 = nn.Linear(input_size, 1700)
        self.fc2 = nn.Linear(1700, num_classes)
        self.relu = nn.ReLU()

        # 添加drop out
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# 获取输入大小和类别数
input_size = train_set[0][0].shape[0]
num_classes = len(torch.unique(train_set[:][1]))

# 初始化模型
model = SimpleNN1(input_size, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    total = 0
    
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)
        
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_acc = 100 * train_correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/total:.4f}, Accuracy: {train_acc:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'model1')

# 第二个模型
model = SimpleNN2(input_size, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    total = 0
    
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)
        
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_acc = 100 * train_correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/total:.4f}, Accuracy: {train_acc:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'model2')

# 加载第一个模型
model = SimpleNN1(input_size, num_classes)
model.load_state_dict(torch.load('model1'))
model = model.to(device)
model.eval()
print("handling1")

# 进行预测
predictions1 = []
actuals = []

with torch.no_grad():
    for data, labels in val_loader:
        data = data.to(device)
        labels = labels.to(device)
        
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        
        predictions1.extend(predicted.cpu().numpy())
        actuals.extend(labels.cpu().numpy())

# 将预测结果与实际结果写入CSV文件
accuracy1 = accuracy_score(actuals, predictions1)
print(f'Prediction Accuracy1: {accuracy1 * 100:.2f}%')

# 加载第2个模型
model = SimpleNN2(input_size, num_classes)
model.load_state_dict(torch.load('model2'))
model = model.to(device)
model.eval()
print("handling2")

# 进行预测
predictions2 = []
actuals = []

with torch.no_grad():
    for data, labels in val_loader:
        data = data.to(device)
        labels = labels.to(device)
        
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        
        predictions2.extend(predicted.cpu().numpy())
        actuals.extend(labels.cpu().numpy())

# 将预测结果与实际结果写入CSV文件
accuracy2 = accuracy_score(actuals, predictions2)
print(f'Prediction Accuracy2: {accuracy2 * 100:.2f}%')

if(accuracy1>=accuracy2):
    results = pd.DataFrame({'Actual': actuals, 'Predicted1': predictions1})
    results.to_csv('predictions.csv', index=False)
else:
    results = pd.DataFrame({'Actual': actuals, 'Predicted2': predictions2})
    results.to_csv('predictions.csv', index=False)


