import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, RegressorMixin

config = {
    'seed': 42,      # 随机种子，可以自己填写. :)
    'valid_ratio': 0.2,   # 验证集大小(validation_size) = 训练集大小(train_size) * 验证数据占比(valid_ratio)
    'n_epochs': 50,     # 数据遍历训练次数           
    'batch_size': 32, 
    'learning_rate': 0.001,
    'hidden_units_list': [50, 50],
}

# 设置随机种子
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])


# 读取数据集
covid_data = pd.read_csv('covid.train.csv')

# 分离特征和目标变量
X = covid_data.drop('tested_positive', axis=1)
y = covid_data['tested_positive']

# 检查数据形状
print(f"Feature shape: {X.shape}, Target shape: {y.shape}")

# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config['valid_ratio'], random_state=42)

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # 转换为NumPy数组并调整形状
X_val_tensor = torch.tensor(X_test, dtype=torch.float32)
y_val_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)  # 转换为NumPy数组并调整形状

# 打印张量的形状以确保正确
print(X_train_tensor.shape)
print(y_train_tensor.shape)
print(X_val_tensor.shape)
print(y_val_tensor.shape)

# 定义模型
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_units_list):
        super(DNNModel, self).__init__()
        layers = []
        in_features = input_size
        for hidden_units in hidden_units_list:
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())
            in_features = hidden_units
        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 获取输入特征数
input_size = X_train.shape[1]

# 定义隐藏层单元数列表
hidden_units_list = config['hidden_units_list']

model = DNNModel(input_size=input_size, hidden_units_list=hidden_units_list)

# 打印模型结构
print(model)

criterion = nn.MSELoss()
# Adam优化器
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# batch_size调节
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

for epoch in range(config['n_epochs']):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{config["n_epochs"]}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'dnn_model')
print("模型已保存")

# 加载模型并进行预测
model_loaded = DNNModel(input_size=input_size, hidden_units_list=hidden_units_list)
model_loaded.load_state_dict(torch.load('dnn_model'))
model_loaded.eval()

with torch.no_grad():
    predictions = model_loaded(X_val_tensor)

# 计算MSE并打印预测值与实际值
mse = criterion(predictions, y_val_tensor).item()
print(f'Mean Squared Error: {mse:.4f}')

# 打印预测值与实际值
for i in range(10):
    print(f'Actual value: {y_val_tensor[i].item():.4f}, Predicted value: {predictions[i].item():.4f}')

test_data = pd.read_csv('covid.test.csv')
scaler = StandardScaler()
test_data = scaler.fit_transform(test_data)
test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
with torch.no_grad():
    predictions = model_loaded(test_data_tensor)

predictions_np = predictions.detach().cpu().numpy()

# 将预测结果转换为 DataFrame
predictions_df = pd.DataFrame(predictions_np, columns=['Prediction'])

# 将 DataFrame 保存到 CSV 文件
predictions_df.to_csv('pred.csv', index=False)

print('Predictions saved to pred.csv')