import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
data = pd.read_csv('D:/random.csv')
series = data.set_index(['Date'], drop=True)
new_data = series.values
# new_data=data['value']
train_data=new_data[:30]
train_prices=train_data.reshape(-1,1)
# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_prices)
# 创建训练数据集
X_train = []
y_train = []
timesteps = 4  # 时间步长，可根据需求进行调整
for i in range(timesteps, len(train_scaled)):
    X_train.append(train_scaled[i - timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
# 训练数据转为数组形式
X_train, y_train = np.array(X_train), np.array(y_train)    # 调整输入数据的维度
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# 构建图数据
x = torch.tensor([[0], [1], [2], [3], [4]], dtype=torch.float)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
y = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index, y=y)

# 定义图神经网络模型
class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 初始化模型和优化器
model = GNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = nn.MSELoss()(out, data.y)
    loss.backward()
    optimizer.step()

# 预测未来的时间序列
model.eval()
future_x = torch.tensor([[5], [6], [7]], dtype=torch.float)
future_edge_index = torch.tensor([[4, 5, 5, 6, 6, 7], [5, 4, 6, 5, 7, 6]], dtype=torch.long)
future_data = Data(x=future_x, edge_index=future_edge_index)
future_pred = model(future_data)

print("预测结果：", future_pred)
