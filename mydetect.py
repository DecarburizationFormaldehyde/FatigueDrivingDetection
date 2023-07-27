import numpy as np
import cv2
import torch
from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    set_logging
from utils.torch_utils import select_device, time_synchronized


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # 将图像大小调整为 32 像素倍的矩形
    '''
    Args:
        img:图像帧
        new_shape:定义640*640的图像帧大小
        color:颜色矩阵
        auto:
        scaleFill:标准化填充
        scaleup:

    Returns:
        img:调整后的图像
        ratio: 宽度高度比例
        (dw,dh):长宽

    '''
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 比例 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只按比例缩小，不按比例放大（为了更好的测试 mAP）
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r  # 宽度、高度比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # w，h填充
    if auto:  # 最小矩形
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # w，h填充
    elif scaleFill:  # 拉紧
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # 将填充分为两侧
    dh /= 2

    if shape[::-1] != new_unpad:  # 调整大小
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边框
    return img, ratio, (dw, dh)


# 权重路径
weights = r'weights/best.pt'
# 选择cpu
opt_device = '0'  # device = 'cpu' or '0' or '0,1,2,3'
imgsz = 640
opt_conf_thres = 0.6
opt_iou_thres = 0.45

# 初始化
set_logging()
device = select_device(opt_device)
half = device.type != '0'  # half precision only supported on CUDA

# 加载模型
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16

# 获取名称和颜色
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


def predict(im0s):
    '''
    Args:
        im0s: 图片帧

    Returns:
        label: 标签信息 'face' 'smoke' 'drink' 'phone'
        prob: 为对应的置信度
        xyxy: 为对应的位置信息（外框）

    '''
    # 运行推理
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # 设置数据载入
    img = letterbox(im0s, new_shape=imgsz)[0]
    # 转变
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 推理
    # pred = model(img, augment=opt.augment)[0]
    pred = model(img)[0]

    # 使用 NMS
    pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres)

    # 预测过程
    ret = []
    for i, det in enumerate(pred):  # 检测每张图片
        if len(det):
            # 重标准化
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            # 写出预测结果
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]}'
                prob = round(float(conf) * 100, 2)  # round 2
                ret_i = [label, prob, xyxy]
                ret.append(ret_i)
    return ret
