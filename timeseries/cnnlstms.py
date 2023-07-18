import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout


def cnnlstm(data):
    timesteps = 7  # 时间步长，可根据需求进行调整
    new_data = np.array(data)
    train_data = new_data[:len(data) -  timesteps]
    train_prices = train_data.reshape(-1, 1)
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(new_data.reshape(-1,1))
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
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=1000, batch_size=32)
    model.save('series_model.h5')
    predict = model.predict(X_train.reshape(X_train.shape[0], X_train.shape[1], 1))
    train_scaled = scaler.inverse_transform(predict)
    plt.plot(train_scaled[timesteps:]+1)
    plt.plot(scaler.inverse_transform(train_scaled)[timesteps:len(data) - timesteps + 1])
    plt.show()
    # 提取测试数据
    test_data = new_data[len(data) -  timesteps:]
    test_prices = test_data.reshape(-1, 1)
    # 数据归一化
    test_scaled = scaler.transform(test_prices)
    print(len(data))
    print(len(test_scaled))
    # list(test_scaled)
    i = 0
    while True:
        input = test_scaled[i:i + timesteps]
        X_test = np.array(input)
        print(X_test)
        X_test = np.reshape(X_test, (1, 7, 1))
        # 模型预测
        predicted = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
        # test_scaled.append(predicted[:])
        test_scaled = np.append(test_scaled, predicted[:])
        i = i + 1
        if i == 15:
            break
    return scaler.inverse_transform(test_scaled[timesteps:].reshape(-1, 1))


def cnnlstms(data):
    model=keras.models.load_model('series_model.h5')
    timesteps = 7  # 时间步长，可根据需求进行调整
    new_data = np.array(data)
    train_data = new_data[:len(data) -  timesteps]
    train_prices = train_data.reshape(-1, 1)
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(new_data.reshape(-1,1))
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
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=1000, batch_size=32)
    model.save('series_model.h5')
    predict = model.predict(X_train.reshape(X_train.shape[0], X_train.shape[1], 1))
    train_scaled = scaler.inverse_transform(predict)
    plt.plot(train_scaled[timesteps:]+1)
    plt.plot(scaler.inverse_transform(train_scaled)[timesteps:len(data) - timesteps + 1])
    plt.show()
    # 提取测试数据
    test_data = new_data[len(data) -  timesteps:]
    test_prices = test_data.reshape(-1, 1)
    # 数据归一化
    test_scaled = scaler.transform(test_prices)
    print(len(data))
    print(len(test_scaled))
    # list(test_scaled)
    i = 0
    while True:
        input = test_scaled[i:i + timesteps]
        X_test = np.array(input)
        print(X_test)
        X_test = np.reshape(X_test, (1, 7, 1))
        # 模型预测
        predicted = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
        # test_scaled.append(predicted[:])
        test_scaled = np.append(test_scaled, predicted[:])
        i = i + 1
        if i == 15:
            break
    return scaler.inverse_transform(test_scaled[timesteps:].reshape(-1, 1))


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
# print(cnnlstm(data))
# print(cnnlstms(data))

# p = sum(i > 0.38 for i in data)
# print(p)
# # 创建测试数据集
# X_test = []
# y_test = []
# for i in range(timesteps, len(test_scaled)):
#     X_test.append(test_scaled[i - timesteps:i, 0])
#     y_test.append(test_scaled[i, 0])
# X_test, y_test = np.array(X_test), np.array(y_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#
# # 模型预测
# predicted = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
# predicted_CL_scaled = scaler.inverse_transform(predicted)
# test_prices_scaled = scaler.inverse_transform(test_scaled)  # 反归一化预测结果
# X_test1=np.array(test_scaled)
# X_test1 = np.reshape(X_test1, (test_scaled.shape[0], test_scaled[1], 1))
# predicted1 = model.predict(X_test1.reshape(X_test1.shape[0], X_test1.shape[1], 1))
# predict.append(predicted1)
#
# X_test2=np.zeros((1,4))
# X_test2[:,:3]=test_scaled[1:,0]
# X_test2[:,3]=predicted1.value
# X_test2 = np.reshape(X_test2, (test_scaled.shape[0], test_scaled[1], 1))
# predicted2 = model.predict(X_test2.reshape(X_test1.shape[0], X_test1.shape[1], 1))
# predict.append(predicted2)
#
# X_test3 = np.zeros((1, 4))
# X_test3[:, :2] = test_scaled[2:, 0]
# X_test3[:, 2] = predicted1.value
# X_test3[:, 3] = predicted2.value
# X_test3 = np.reshape(X_test3, (test_scaled.shape[0], test_scaled[1], 1))
# predicted3 = model.predict(X_test3.reshape(X_test1.shape[0], X_test1.shape[1], 1))
# predict.append(predicted3)
#
# X_test4 = np.zeros((1, 4))
# X_test4[:, 0] = test_scaled[3, 0]
# X_test4[:, 1] = predicted1.value
# X_test4[:, 2] = predicted2.value
# X_test4[:, 3] = predicted3.value
# X_test4 = np.reshape(X_test4, (test_scaled.shape[0], test_scaled[1], 1))
# predicted4 = model.predict(X_test4.reshape(X_test1.shape[0], X_test1.shape[1], 1))
# predict.append(predicted4)
