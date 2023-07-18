# 疲劳检测，检测眼睛和嘴巴的开合程度

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np  # 数据处理的库 numpy
import argparse
import imutils
import time
import dlib
import cv2
import math
import time
from threading import Thread

def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = dist.euclidean(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼睛的长宽比
    return ear

def mouth_aspect_ratio(mouth):  # 嘴部
    A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[8])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

# 初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
print("[INFO] loading facial landmark predictor...")
# 使用dlib.get_frontal_face_detector() 获得脸部位置检测器
detector = dlib.get_frontal_face_detector()
# 使用dlib.shape_predictor获得脸部特征位置检测器
predictor = dlib.shape_predictor('weights/shape_predictor_68_face_landmarks.dat')
# 分别获取左右眼面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


"""
头部姿态识别模块
"""
# 获取最大的人脸
def _largest_face(dets):
    """
    @Desc    :   从一个由 dlib 库检测到的人脸框列表中，找到最大的人脸框，并返回该框在列表中的索
                如果只有一个人脸，直接返回
                 Args:
                   dets： 一个由 `dlib.rectangle` 类型的对象组成的列表，每个对象表示一个人脸框
                 Returns:
                   人脸索引
    """
    # 如果列表长度为1，则直接返回
    if len(dets) == 1:
        return 0
    # 计算每个人脸框的面积
    face_areas = [(det.right() - det.left()) * (det.bottom() - det.top()) for det in dets]
    import heapq
    # 找到面积最大的人脸框的索引
    largest_area = face_areas[0]
    largest_index = 0
    for index in range(1, len(dets)):
        if face_areas[index] > largest_area:
            largest_index = index
            largest_area = face_areas[index]
    # 打印最大人脸框的索引和总人脸数
    print("largest_face index is {} in {} faces".format(largest_index, len(dets)))
    return largest_index

def get_image_points_from_landmark_shape(landmark_shape):
    """
    @Desc    :   从dlib的检测结果抽取姿态估计需要的点坐标
                 Args:
                   landmark_shape:  所有的位置点
                 Returns:
                   void
    """

    if landmark_shape.num_parts != 68:
        print("ERROR:landmark_shape.num_parts-{}".format(landmark_shape.num_parts))
        return -1, None

    # 2D image points. If you change the image, you need to change vector

    image_points = np.array([
        (landmark_shape.part(17).x, landmark_shape.part(17).y),  # 17 left brow left corner
        (landmark_shape.part(21).x, landmark_shape.part(21).y),  # 21 left brow right corner
        (landmark_shape.part(22).x, landmark_shape.part(22).y),  # 22 right brow left corner
        (landmark_shape.part(26).x, landmark_shape.part(26).y),  # 26 right brow right corner
        (landmark_shape.part(36).x, landmark_shape.part(36).y),  # 36 left eye left corner
        (landmark_shape.part(39).x, landmark_shape.part(39).y),  # 39 left eye right corner
        (landmark_shape.part(42).x, landmark_shape.part(42).y),  # 42 right eye left corner
        (landmark_shape.part(45).x, landmark_shape.part(45).y),  # 45 right eye right corner
        (landmark_shape.part(31).x, landmark_shape.part(31).y),  # 31 nose left corner
        (landmark_shape.part(35).x, landmark_shape.part(35).y),  # 35 nose right corner
        (landmark_shape.part(48).x, landmark_shape.part(48).y),  # 48 mouth left corner
        (landmark_shape.part(54).x, landmark_shape.part(54).y),  # 54 mouth right corner
        (landmark_shape.part(57).x, landmark_shape.part(57).y),  # 57 mouth central bottom corner
        (landmark_shape.part(8).x, landmark_shape.part(8).y),  # 8 chin corner
    ], dtype="double")
    return 0, image_points


def get_pose_estimation(img_size, image_points):

    """
    @Desc    :   获取旋转向量和平移向量
                 Returns: void
    """

    # 3D model points.
    model_points = np.array([
        (6.825897, 6.760612, 4.402142),  # 33 left brow left corner
        (1.330353, 7.122144, 6.903745),  # 29 left brow right corner
        (-1.330353, 7.122144, 6.903745),  # 34 right brow left corner
        (-6.825897, 6.760612, 4.402142),  # 38 right brow right corner
        (5.311432, 5.485328, 3.987654),  # 13 left eye left corner
        (1.789930, 5.393625, 4.413414),  # 17 left eye right corner
        (-1.789930, 5.393625, 4.413414),  # 25 right eye left corner
        (-5.311432, 5.485328, 3.987654),  # 21 right eye right corner
        (2.005628, 1.409845, 6.165652),  # 55 nose left corner
        (-2.005628, 1.409845, 6.165652),  # 49 nose right corner
        (2.774015, -2.080775, 5.048531),  # 43 mouth left corner
        (-2.774015, -2.080775, 5.048531),  # 39 mouth right corner
        (0.000000, -3.116408, 6.097667),  # 45 mouth central bottom corner
        (0.000000, -7.415691, 4.070434)  # 6 chin corner
    ])
    # 相机参数，可变化

    focal_length = img_size[1]
    center = (img_size[1] / 2, img_size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.array([7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000],
                           dtype="double")  # Assuming no lens distortion

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs


def get_euler_angle(rotation_vector):
    """
    @Desc    :   从旋转向量转换为欧拉角
                 Args:

                 Returns:
                   void
    """

    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)

    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2) * rotation_vector[0][0] / theta
    y = math.sin(theta / 2) * rotation_vector[1][0] / theta
    z = math.sin(theta / 2) * rotation_vector[2][0] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)

    # print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)

    print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

    # 单位转换：将弧度转换为度
    pitch_degree = int((pitch / math.pi) * 180)
    yaw_degree = int((yaw / math.pi) * 180)
    roll_degree = int((roll / math.pi) * 180)

    return 0, pitch, yaw, roll, pitch_degree, yaw_degree, roll_degree

def get_pose_estimation_in_euler_angle(landmark_shape, im_szie):
    try:
        ret, image_points = get_image_points_from_landmark_shape(landmark_shape)
        if ret != 0:
            print('get_image_points failed')
            return -1, None, None, None

        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(im_szie,
                                                                                                   image_points)
        if ret != True:
            print('get_pose_estimation failed')
            return -1, None, None, None

        ret, pitch, yaw, roll = get_euler_angle(rotation_vector)
        if ret != 0:
            print('get_euler_angle failed')
            return -1, None, None, None

        euler_angle_str = 'Pitch:{}, Yaw:{}, Roll:{}'.format(pitch, yaw, roll)
        print(euler_angle_str)
        return 0, pitch, yaw, roll

    except Exception as e:
        print('get_pose_estimation_in_euler_angle exception:{}'.format(e))
        return -1, None, None, None


def get_image_points(img):
    """
    @Desc    :   用dlib检测关键点，返回姿态估计需要的几个点坐标
                 Args:
                 Returns:
                   void
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片调整为灰色

    dets = detector(gray, 1)  # 设置图片放大倍数为1，重要参数

    if 0 == len(dets):
        print("ERROR: found no face")
        return -1, None
    largest_index = _largest_face(dets)
    face_rectangle = dets[largest_index]

    landmark_shape = predictor(img, face_rectangle)

    return get_image_points_from_landmark_shape(landmark_shape)

"""

"""

# 从视频流循环帧
def detfatigue(frame):
    #frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用detector(gray, 0) 进行脸部位置检测
    rects = detector(gray, 0)
    eyear = 0.0
    mouthar = 0.0
    degree=[]
    # 循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
    for rect in rects:
        shape = predictor(gray, rect)

        # 将脸部特征信息转换为数组array的格式
        shape = face_utils.shape_to_np(shape)
        # 提取面部点位
        ret, image_points = get_image_points(frame)
        size = frame.shape
        # 返回姿态估计，获得欧拉角
        if ret != 0:
            print('no face')
        else:
            ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(size,
                                                                                           image_points)
            degree=get_euler_angle(rotation_vector)
        # 提取左眼和右眼坐标
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # 嘴巴坐标
        mouth = shape[mStart:mEnd]

        # 构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        eyear = (leftEAR + rightEAR) / 2.0
        # 打哈欠
        mouthar = mouth_aspect_ratio(mouth)

        # 标注识别结果
        # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # 画出眼睛、嘴巴竖直线
        cv2.line(frame,tuple(shape[38]),tuple(shape[40]),(0, 255, 0), 1)
        cv2.line(frame,tuple(shape[43]),tuple(shape[47]),(0, 255, 0), 1)
        cv2.line(frame,tuple(shape[51]),tuple(shape[57]),(0, 255, 0), 1)
        cv2.line(frame,tuple(shape[48]),tuple(shape[54]),(0, 255, 0), 1)

    # 返回信息
    # frame已经标注出眼睛和嘴巴的框线
    # eyeae为眼睛的长宽比
    # mouthar为嘴巴的长宽比
    # degree为返回的欧拉角

    return(frame,eyear,mouthar,degree)