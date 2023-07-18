import torch
import torch.nn as nn
import torch.nn.functional
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
plt.rcParams['font.family']='simhei'
plt.rcParams['axes.unicode_minus']=False
scaler = MinMaxScaler(feature_range=(-1, 1))

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    input_data = np.array(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]#预测time_step之后的第一个数值
        inout_seq.append((train_seq, train_label))#inout_seq内的数据不断更新，但是总量只有tw+1个
    return inout_seq

def getdata(input_data,pre_size,train_window):
    # data=input_data.values.astype(float)
    train_data = data[:-pre_size]
    test_data = data[-pre_size:]
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
    # 将数据集转换为tensor，因为PyTorch模型是使用tensor进行训练的，并将训练数据转换为输入序列和相应的标签
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
    return train_inout_seq,train_data_normalized


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        # 创建LSTM层和linear层，LSTM层提取特征，linear层用作最后的预测
        ##LSTM算法接受三个输入：先前的隐藏状态，先前的单元状态和当前输入。
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        #初始化隐含状态及细胞状态C，hidden_cell变量包含先前的隐藏状态和单元状态
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        #lstm的输出是当前时间步的隐藏状态ht和单元状态ct以及输出lstm_out
        #按照lstm的格式修改input_seq的形状，作为linear层的输入
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]#返回predictions的最后一个元素
# 训练
def train(train_inout_seq):
    # 创建LSTM()类的对象，定义损失函数和优化器
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 建立优化器实例
    print(model)
    epochs = 1500
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            # 清除网络先前的梯度值
            optimizer.zero_grad()  # 训练模型时需要使模型处于训练模式，即调用model.train()。缺省情况下梯度是累加的，需要手工把梯度初始化或者清零，调用optimizer.zero_grad()
            # 初始化隐藏层数据
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            # 实例化模型
            y_pred = model(seq)
            # 计算损失，反向传播梯度以及更新模型参数
            single_loss = loss_function(y_pred, labels)  # 训练过程中，正向传播生成网络的输出，计算输出和实际值之间的损失值
            single_loss.backward()  # 调用loss.backward()自动生成梯度，
            optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络

        # 查看模型训练的结果
        if i % 25 == 1:
            print(f'epoch:{i:3} loss:{single_loss.item():10.8f}')
    print(f'epoch:{i:3} loss:{single_loss.item():10.10f}')
    return model

# 预测
def predict(pre_size,train_window,train_data_normalized,model):
    test_inputs = train_data_normalized[-train_window:].tolist()
    # 更改模型为测试或者验证模式
    model.eval()  # 把training属性设置为false,使模型处于测试或验证状态
    for i in range(pre_size):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())
    return test_inputs

# 绘图
def plot_pre(input_data,test_inputs,train_window,pre_size):
    actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
    x = np.arange(len(input_data)-pre_size+1, len(input_data)-pre_size+1 + pre_size, 1)
    plt.ylabel('AQI')
    plt.xlabel('时间')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(input_data)
    plt.plot(x, actual_predictions)
    plt.show()


def LSTM_pred(input_data,train_window,pre_size):
    train_inout_seq,train_data_normalized=getdata(input_data, pre_size, train_window)
    model=train(train_inout_seq)
    test_inputs=predict(pre_size,train_window,train_data_normalized,model)
    plot_pre(input_data, test_inputs, train_window, pre_size)

# data=np.random.uniform(0,1,50)
data = [0.18, 0.05333333333333334, 0.02, 0.10666666666666667, 0.3, 0.2733333333333333, 0.013333333333333334, 0.04,
        0.08133333333333334, 0.5, 0.006666666666666667, 0.006666666666666667, 0.0, 0.20666666666666667,
        0.09333333333333334, 0.02, 0.04666666666666667, 0.05333333333333334, 0.8466666666666667, 0.8066666666666666,
        0.06,
        0.006666666666666667, 0.44, 0.41333333333333333, 0.5533333333333333, 0.29333333333333333, 0.06, 0.6,
        0.14666666666666667, 0.04, 0.013333333333333334, 0.24, 0.02666666666666667, 0.21333333333333335,
        0.22266666666666668, 0.17333333333333334, 0.14933333333333335, 0.37066666666666664, 0.23333333333333334, 0.0,
        0.06,
        0.8866666666666667, 0.94, 0.94, 1.0, 0.9333333333333333, 0.38, 0.28933333333333333, 0.23466666666666666,
        0.17333333333333334, 0.5866666666666667, 0.45466666666666666, 0.25333333333333335, 0.05333333333333334,
        0.3933333333333333, 0.25333333333333335, 0.7466666666666667, 0.52, 0.7333333333333333, 0.6133333333333333,
        0.9466666666666667, 1.0, 0.8066666666666666, 0.41333333333333333, 0.188, 0.9933333333333333, 1.0,
        0.41333333333333333, 0.2866666666666667, 0.9866666666666667, 0.9933333333333333, 1.0, 0.9866666666666667,
        0.9866666666666667, 0.58, 0.9933333333333333, 0.14, 0.0, 0.006666666666666667, 0.006666666666666667,
        0.8133333333333334, 1.0, 0.3933333333333333, 0.02666666666666667, 0.0, 0.03066666666666667, 0.02666666666666667,
        0.006666666666666667, 0.08, 0.0, 0.02, 0.0, 0.0, 0.02, 0.0, 0.04, 0.0013333333333333335, 0.0, 0.0, 0.133]
LSTM_pred(data,15,10)



