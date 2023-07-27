import json
import sys
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout


def cnnlstm(data):
    '''
    初次预测函数
    Args:
        data: 接收到的perclose数据

    Returns: 返回预测结果

    '''
    timesteps = 7  # 时间步长，可根据需求进行调整
    new_data = np.array(data)
    train_data = new_data[:len(data) - timesteps]
    train_prices = train_data.reshape(-1, 1)
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(new_data.reshape(-1, 1))
    train_scaled = scaler.transform(train_prices)
    # 创建训练数据集
    X_train = []
    y_train = []

    for i in range(timesteps, len(train_scaled)):
        X_train.append(train_scaled[i - timesteps:i, 0])
        y_train.append(train_scaled[i, 0])
    # 训练数据转为数组形式
    X_train, y_train = np.array(X_train), np.array(y_train)  # 调整输入数据的维度
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # 构建LSTM-CNN模型
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # 模型训练
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=1000, batch_size=32, verbose=0)
    model.save('series_model.h5')
    predict = model.predict(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), verbose=0)
    train_scaled = scaler.inverse_transform(predict)
    # 提取测试数据
    test_data = new_data[len(data) - timesteps:]
    test_prices = test_data.reshape(-1, 1)
    # 数据归一化
    test_scaled = scaler.transform(test_prices)
    i = 0
    while True:
        input = test_scaled[i:i + timesteps]
        X_test = np.array(input)
        # print(X_test)
        X_test = np.reshape(X_test, (1, 7, 1))
        # 模型预测
        predicted = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), verbose=0)
        # test_scaled.append(predicted[:])
        test_scaled = np.append(test_scaled, predicted[:])
        i = i + 1
        if i == 7:
            break
    return scaler.inverse_transform(test_scaled[timesteps:].reshape(-1, 1))


def cnnlstms(data):
    '''
    第二次预测函数,与之不同的是要加载对应的保存好的预测模型
    Args:
        data: 接收到的perclose数据

    Returns: 返回预测结果

    '''
    model = keras.models.load_model('series_model.h5')
    timesteps = 7  # 时间步长，可根据需求进行调整
    new_data = np.array(data)
    train_data = new_data[:len(data) - timesteps]
    train_prices = train_data.reshape(-1, 1)
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(new_data.reshape(-1, 1))
    train_scaled = scaler.transform(train_prices)
    # 创建训练数据集
    X_train = []
    y_train = []

    for i in range(timesteps, len(train_scaled)):
        X_train.append(train_scaled[i - timesteps:i, 0])
        y_train.append(train_scaled[i, 0])
    # 训练数据转为数组形式
    X_train, y_train = np.array(X_train), np.array(y_train)  # 调整输入数据的维度
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # 模型训练
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=1000, batch_size=32, verbose=0)
    model.save('series_model.h5')
    predict = model.predict(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), verbose=0)
    train_scaled = scaler.inverse_transform(predict)
    # 提取测试数据
    test_data = new_data[len(data) - timesteps:]
    test_prices = test_data.reshape(-1, 1)
    # 数据归一化
    test_scaled = scaler.transform(test_prices)
    i = 0
    while True:
        input = test_scaled[i:i + timesteps]
        X_test = np.array(input)
        # print(X_test)
        X_test = np.reshape(X_test, (1, 7, 1))
        # 模型预测
        predicted = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), verbose=0)
        # test_scaled.append(predicted[:])
        test_scaled = np.append(test_scaled, predicted[:])
        i = i + 1
        if i == 7:
            break
    return scaler.inverse_transform(test_scaled[timesteps:].reshape(-1, 1))
