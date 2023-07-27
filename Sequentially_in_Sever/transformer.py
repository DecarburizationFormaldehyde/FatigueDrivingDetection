import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))


# 定义Transformer模型
class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, output_dim):
        '''
        Args:
            num_layers: 总层数
            d_model: 模型维度
            num_heads: 多头注意力头数
            dff:时间窗口
            maximum_position_encoding: 最大位置编码
            output_dim:输出层维度
        '''
        super(TransformerModel, self).__init__()

        self.encoder = layers.Dense(d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.transformer_blocks = [TransformerBlock(d_model, num_heads, dff) for _ in range(num_layers)]
        self.dropout = layers.Dropout(0.1)

        self.final_layer = layers.Dense(output_dim)

    def call(self, x):
        '''

        Args:
            x: 输入的序列数据

        Returns:

        '''
        seq_len = tf.shape(x)[1]
        x = self.encoder(x)

        x += self.pos_encoding[:, :seq_len, :]

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.dropout(x)

        x = tf.reshape(x, [-1, x.shape[1] * x.shape[2]])

        x = self.final_layer(x)
        return x


# 定义Transformer块
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        '''
        Args:
            d_model:模型维度
            num_heads: 多头注意力头数
            dff: 时间窗口
            rate: 学习率
        '''
        super(TransformerBlock, self).__init__()

        self.multi_head_attention = layers.MultiHeadAttention(num_heads, d_model)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x):
        '''
        Args:
            x: 输入序列

        Returns:

        '''
        attn_output = self.multi_head_attention(x, x)

        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2


# Define the positional encoding function
def positional_encoding(position, d_model):
    '''

    Args:
        position:位置
        d_model: 模型维度

    Returns:位置编码

    '''
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(position, i, d_model):
    '''
    Args:
        position: 位置
        i:
        d_model: 模型维度

    Returns:修正后的位置编码

    '''
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return position * angle_rates


def split_data(data, timesteps):
    '''
    Args:
        data: 数据
        timesteps: 划分数据集窗口

    Returns:划分后的数据

    '''
    data_splits = []
    for i in range(len(data) - timesteps):
        data_splits.append(data[i:i + timesteps])
    return data_splits


def data_get(x):
    '''
    Args:
        x: 初始时间序列数据

    Returns:
        切割好的数据集
    '''
    data_splits = split_data(x, 16)
    data_splits = np.reshape(np.array(data_splits), (len(data_splits), 16, 1))
    X_train, X_test, y_train, y_test = train_test_split(data_splits[:, :9, :], data_splits[:, 9:, 0], test_size=0.1,
                                                        shuffle=False)
    a1, b1, c1 = X_train.shape[0], X_train.shape[1], X_train.shape[2]
    X_train = scaler.fit_transform(X_train.reshape(a1, b1 * c1))
    a2, b2, c2 = X_test.shape[0], X_test.shape[1], X_test.shape[2]
    X_test = scaler.fit_transform(X_test.reshape(a2, b2 * c2))
    X_train = X_train.reshape(a1, b1, c1)
    X_test = X_test.reshape(a2, b2, c2)
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.fit_transform(y_test)

    return X_train, X_test, y_train, y_test


def transformer_class(x):
    '''
    用于时间序列预测
    Args:
        x: 要预测的时间序列

    Returns: 预测结果

    '''
    X_train, X_test, y_train, y_test = data_get(x)
    num_layers = 4
    d_model = 32
    num_heads = 4
    dff = 128
    maximum_position_encoding = 100
    output_dim = 7
    # Create an instance of the Transformer model
    transformer = TransformerModel(num_layers, d_model, num_heads, dff, maximum_position_encoding, output_dim)
    # Compile the model
    transformer.compile(optimizer='adam', loss='mse')
    # Train the model
    history = transformer.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), verbose=0)
    transformer.save('series_model2')
    # Predict the next number in the sequence
    x = x.reshape(-1, 1)
    x = scaler.fit_transform(x)
    x_pre = x[-9:]
    x_pre = x_pre[np.newaxis, :]
    # print(x_pre.shape)
    pre = transformer.predict(x_pre, verbose=0)
    pre = scaler.inverse_transform(pre)
    return pre


def transformer_classes(x):
    '''
    第二次预测
    Args:
        x: 要预测的时间序列x

    Returns:第二次预测的结果

    '''
    transformer = keras.models.load_model('series_model2')
    X_train, X_test, y_train, y_test = data_get(x)
    history = transformer.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), verbose=0)
    transformer.save('series_model2')
    x = x.reshape(-1, 1)
    x = scaler.fit_transform(x)
    x_pre = x[-9:]
    x_pre = x_pre[np.newaxis, :]
    pre = transformer.predict(x_pre, verbose=0)
    pre = scaler.inverse_transform(pre)
    return pre
