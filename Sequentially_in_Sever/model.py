from cnn_lstm import cnnlstm, cnnlstms
from transformer import transformer_class, transformer_class
import numpy as np
import sys
import json

# 获取命令行参数
args = sys.argv[1:]  # 第一个参数是脚本路径，从第二个参数开始是传递的参数
data = json.loads(args[0])
# print(data)
mode = data.pop()
flag = data.pop()
if (mode == 0):
    if (flag == 0):
        print(cnnlstm(data))
    else:
        print(cnnlstms(data))
else:
    data = np.array(data)
    if (flag == 0):
        print(cnnlstm(data))
    else:
        print(cnnlstms(data))
