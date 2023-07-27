# 主函数
import asyncio
import sys
import os
from glob import glob

import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PySide6.QtCore import QDir, QTimer, Slot
from PySide6.QtGui import QPixmap, QImage
from utils.rtsp_win import Window
from UIFunctions import *
from ui.home import Ui_MainWindow
from ui.CustomMessageBox import MessageBox
import cv2
import json
import myframe

import pyttsx3 as pyttsx
import re
# 定义变量
import asyncio
import websockets
import json

str_mes = ''


async def connect(mode, data):
    '''
    Websocket连接函数
    Args:
        mode: 选择时间序列模型
        data: 数据

    Returns:

    '''

    # WebSocket连接事件处理函数
    async def on_open(ws):
        print('WebSocket连接已建立')
        # 构造要发送的数据
        # 将数据转换为JSON字符串并发送给服务器
        if (mode == 'CNN-LSTM'):
            data.append(0)
        else:
            data.append(0)
        await ws.send(json.dumps(data))

    # WebSocket消息接收事件处理函数
    async def on_message(ws, message):
        global str_mes
        print('接收到服务器消息:', message)
        # list.append(message)
        str_mes = ''
        str_mes += str(message)

    # WebSocket关闭事件处理函数
    async def on_close(ws):
        print('WebSocket连接已关闭')

    async with websockets.connect('ws://47.94.57.223:8080') as ws:
        # 调用连接建立时的事件处理函数
        await on_open(ws)
        # 循环接收服务器消息
        message = await ws.recv()
        await on_message(ws, message)
        # 调用连接关闭时的事件处理函数
        await on_close(ws)


# 眼睛闭合判断
EYE_AR_THRESH = 0.15  # 眼睛长宽比
EYE_AR_CONSEC_FRAMES = 2  # 闪烁阈值

# 嘴巴开合判断
MAR_THRESH = 0.65  # 打哈欠长宽比
MOUTH_AR_CONSEC_FRAMES = 5  # 闪烁阈值

# 定义检测变量，并初始化
COUNTER = 0  # 眨眼帧计数器
TOTAL = 0  # 眨眼总数
mCOUNTER = 0  # 打哈欠帧计数器
mTOTAL = 0  # 打哈欠总数
ActionCOUNTER = 0  # 分心行为计数器器

# 疲劳判断变量
# Perclos模型
# perclos = (Rolleye/Roll) + (Rollmouth/Roll)*0.2
Roll = 0  # 整个循环内的帧技术
Rolleye = 0  # 循环内闭眼帧数
Rollmouth = 0  # 循环内打哈欠数
Percloslist = []
Rolls = 0
perclos = 0
degrees_last = []
degree_count = 0


####################################################

# 界面

####################################################
class MainWindow(QMainWindow, Ui_MainWindow):

    # main2yolo_begin_sgl = Signal()  # The main window sends an execution signal to the yolo instance
    def __init__(self, parent=None):
        super().__init__(parent)
        # basic interface
        self.setupUi(self)
        # 打开文件类型，用于类的定义
        self.f_type = None
        # 向服务器传递模型参数
        self.select_model = 'CNN-LSTM'
        # 原窗口初始化
        self.setGeometry(50, 20, 1150, 450)  # 设置窗口初始位置和大小
        self.setAttribute(Qt.WA_TranslucentBackground)  # rounded transparent
        self.setWindowFlags(Qt.FramelessWindowHint)  # Set window flag: hide window borders
        UIFuncitons.uiDefinitions(self)
        # Show module shadows
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))
        # 作者初始化
        self.logo.setToolTip("我们是作者")
        # 读取models文件
        new_items = ['CNN-LSTM', 'Transformer']
        self.model_box.addItems(new_items)

        # 模型参数，预留接口
        self.model_box.currentTextChanged.connect(self.change_model)
        self.iou_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'iou_spinbox'))  # iou box
        self.iou_slider.valueChanged.connect(lambda x: self.change_val(x, 'iou_slider'))  # iou scroll bar
        self.conf_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(lambda x: self.change_val(x, 'conf_slider'))  # conf scroll bar
        self.speed_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'speed_spinbox'))  # speed box
        self.speed_slider.valueChanged.connect(lambda x: self.change_val(x, 'speed_slider'))  # speed scroll bar

        # 初始化label
        self.Class_num.setText('清醒')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_name.setText(self.model_box.currentText())

        # 左侧功能狂
        self.src_file_button.clicked.connect(self.open_src_file)  # select local file
        # 摄像头授权
        self.src_cam_button.clicked.connect(self.empower)  # empower   CamConfig_init
        # 重新开始检测
        self.src_rtsp_button.clicked.connect(self.Restart_detection)  # chose_rtsp

        # 开始检测按钮
        self.run_button.clicked.connect(self.toggle_timer)  # pause/start

        # 左侧展开按钮
        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))  # left navigation button
        # 右侧设置按钮
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))  # top right settings button
        # 提示框初始化
        self.loading_box = MessageBox(title='Note', text='loading camera...', time=3000, auto=False, menus=False)
        # 模型重新加载的提示框初始化
        self.loading_box_model = MessageBox(title='Note', text='模型加载中...', time=3000, auto=False, menus=False)
        # 初始化
        self.load_config()

    # 为按钮绑定空格
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.run_button.click()
        else:
            super().keyPressEvent(event)

    # 重新开始检测的函数
    def Restart_detection(self):
        try:
            self.stop()
            restart = MessageBox(
                title='Restart_detection', text='确认重新开始检测？', time=2000, auto=False, menus=True).exec()
            if restart == QMessageBox.StandardButton.Ok:
                # 用户点击了“是”按钮
                # 执行摄像头授权相关的代码
                # 提示框初始化
                self.loading_box.show()
                # 摄像头类初始化
                CamConfig_init()
                # 初始化摄像头类之后，先暂停摄像头调取
                self.f_type.v_timer.stop()
                # 执行加载摄像头的代码
                self.run_button.setEnabled(True)  # 允许点击run_button
                self.run_button.clicked.connect(self.toggle_timer)  # 绑定toggle_timer函数
                self.run_button.setToolTip("开始检测")  # 设置提示信息
                # 初始化标题
                self.Class_num.setText('清醒')
                self.Target_num.setText('---')
                self.fps_label.setText('---')
                # 关闭加载提示框
                self.loading_box.close()
            elif restart == QMessageBox.StandardButton.Close:
                pass
        except Exception as e:
            self.show_status('%s' % e)
            print('%s' % e)

    # Select camera source----  have one bug
    def empower(self):
        try:
            self.stop()
            empower_box = MessageBox(
                title='授权摄像头', text='是否授权使用摄像头？', time=2000, auto=False, menus=True).exec()
            if empower_box == QMessageBox.StandardButton.Ok:
                # 用户点击了“是”按钮
                # 执行摄像头授权相关的代码
                self.loading_box.show()
                # 执行加载摄像头的代码
                self.run_button.setEnabled(True)  # 允许点击run_button
                self.run_button.clicked.connect(self.toggle_timer)  # 绑定toggle_timer函数
                self.run_button.setToolTip("开始检测")  # 设置提示信息
                CamConfig_init()
                # 初始化摄像头类之后，先暂停摄像头调取
                # self.f_type.v_timer.stop()
                # 设置重新检测按钮的功能可用
                self.src_rtsp_button.setEnabled(True)
                self.src_rtsp_button.setToolTip('重新开始检测...')
                # 关闭加载提示框
                self.loading_box.close()
                # 自动点击run_button按钮
                self.run_button.click()
            elif empower_box == QMessageBox.StandardButton.Close:
                # 用户点击了“否”按钮
                self.run_button.setEnabled(False)  # 禁止点击run_button
                self.run_button.setToolTip("没有授权")  # 设置提示信息
                self.src_rtsp_button.setEnabled(False)
                self.src_rtsp_button.setToolTip('摄像头未授权，无法进行重新检测！')
        except Exception as e:
            self.show_status('%s' % e)
            print('%s' % e)

    # 暂停按钮
    def toggle_timer(self):
        if self.run_button.isChecked():
            self.f_type.v_timer.start()  # 启动定时器
            self.show_status('正在检测...')
        else:
            self.f_type.v_timer.stop()  # 停止定时器
            self.show_status('已暂停')

    def run_or_continue(self):
        self.show_status('Please select a video source before starting detection...')

    # bottom status bar information
    def show_status(self, msg):
        self.status_bar.setText(msg)

    # # select local file
    def open_src_file(self):
        config_file = 'config/fold.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold,
                                              "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        # 如果存在文件则执行下列操作
        if name:
            print('打开了文件')

    # select network source
    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.2:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    # load network sources
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='提示', text='加载 rtsp...', time=1000, auto=True).exec()
            # self.yolo_predict.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.show_status('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.show_status('%s' % e)

    # Configuration initialization  ~~~wait to change~~~
    def load_config(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            save_res = 0
            save_txt = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "save_res": save_res,
                          "save_txt": save_txt
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                save_res = config['save_res']
                save_txt = config['save_txt']
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        # self.yolo_predict.save_res = (False if save_res==0 else True )
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt))
        # self.yolo_predict.save_txt = (False if save_txt==0 else True )
        self.run_button.setChecked(False)
        # self.show_status("Welcome~")

    # Terminate button and associated state
    def stop(self):
        self.run_button.setChecked(False)  # start key recovery
        self.save_res_button.setEnabled(True)  # Ability to use the save button
        self.save_txt_button.setEnabled(True)  # Ability to use the save button
        self.pre_video.clear()  # clear image display
        self.res_video.clear()  # clear image display
        # self.progress_bar.setValue(0)
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

    # Change detection parameters
    def change_val(self, x, flag):
        if flag == 'iou_spinbox':
            self.iou_slider.setValue(int(x * 100))  # The box value changes, changing the slider
        elif flag == 'iou_slider':
            self.iou_spinbox.setValue(x / 100)  # The slider value changes, changing the box
            self.show_status('IOU Threshold: %s' % str(x / 100))
            # self.yolo_predict.iou_thres = x/100
        elif flag == 'conf_spinbox':
            self.conf_slider.setValue(int(x * 100))
        elif flag == 'conf_slider':
            self.conf_spinbox.setValue(x / 100)
            self.show_status('Conf Threshold: %s' % str(x / 100))
            # self.yolo_predict.conf_thres = x/100
        elif flag == 'speed_spinbox':
            self.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.speed_spinbox.setValue(x)
            self.show_status('Delay: %s ms' % str(x))
            self.show_status('%s' % self.speed_slider.value())
            print(self.speed_slider.value(), '\n', self.speed_spinbox.value())
            # self.yolo_predict.speed_thres = x  # ms

    # change model
    def change_model(self, x):
        self.select_model = self.model_box.currentText()
        # 提示框初始化
        self.loading_box_model.show()
        # 摄像头类初始化
        CamConfig_init()
        # 初始化摄像头类之后，先暂停摄像头调取
        self.f_type.v_timer.stop()
        # 初始化标题
        self.Class_num.setText('清醒')
        self.Target_num.setText('---')
        self.fps_label.setText('---')
        # self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.show_status('Change Model：%s' % self.select_model)
        self.Model_name.setText(self.select_model)
        # 关闭加载提示框
        self.loading_box_model.close()

    # Get the mouse position (used to hold down the title bar and drag the window)记录鼠标位置
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize the adjustment when dragging the bottom and right edges of the window size调整窗口大小
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)

    def setdemo_window(self, msg):
        self.pre_video.clear()
        self.pre_video.setText(msg)

    def setStatusBar(self, statusbar):
        self.status_bar.clear()
        self.status_bar.setText(statusbar)

    def setTarget_num(self, msg):
        self.Target_num.clear()
        self.Target_num.setText(msg)

    def setClass_num(self, msg):
        self.Class_num.clear()
        self.Class_num.setText(msg)

    # 获得Delays滑条的值
    def getDelays_value(self):
        return self.speed_spinbox.value()

    # 获得Conf的值
    def getConf_value(self):
        return self.conf_spinbox.value()

    # 获得Iou的值
    def getIou_value(self):
        return self.iou_spinbox.value()


# 定义摄像头类
class CamConfig:
    def __init__(self, main_window):
        # 传入主窗口变量
        self.main_window = main_window
        # 用于更新帧率
        self.frame_count = 0
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)  # 每秒更新一次帧率
        # 打开摄像头
        self.main_window.setStatusBar('正在打开摄像头请稍后...')
        # 设置时钟
        self.v_timer = QTimer()
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap:
            self.main_window.setStatusBar("打开摄像头失败")
            return
        # 设置定时器周期，单位毫秒
        self.v_timer.start(10)
        # 连接定时器周期溢出的槽函数，用于显示一帧视频
        self.v_timer.timeout.connect(self.show_pic)

        # 在前端UI输出提示信息
        self.main_window.setStatusBar("载入成功，开始运行程序")
        engine = pyttsx.init()
        engine.say('载入成功，开始运行')
        engine.runAndWait()
        # self.main_window.setStatusBar("正在使用摄像头...")
        self.v_timer.start()

    def show_pic(self):
        # 全局变量
        # 在函数中引入定义的全局变量
        global EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, MAR_THRESH, MOUTH_AR_CONSEC_FRAMES, COUNTER, TOTAL, mCOUNTER, mTOTAL, ActionCOUNTER, Roll, Rolleye, Rollmouth, Percloslist, Rolls, perclos, degrees_last, degree_count

        # 读取摄像头的一帧画面
        success, frame = self.cap.read()
        show_init = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showImage_init = QImage(show_init.data, show_init.shape[1], show_init.shape[0], QImage.Format_RGB888)
        self.main_window.pre_video.setPixmap(QPixmap.fromImage(showImage_init))
        if success:
            # 检测
            # 将摄像头读到的frame传入检测函数myframe.frametest()
            ret, frame = myframe.frametest(frame)
            lab, eye, mouth, degree = ret
            # ret和frame，为函数返回
            # ret为检测结果，ret的格式为[lab,eye,mouth],lab为yolo的识别结果包含'phone' 'smoke' 'drink',eye为眼睛的开合程度（长宽比），mouth为嘴巴的开合程度
            # frame为标注了识别结果的帧画面，画上了标识框

            # 分心行为判断
            # 分心行为检测以15帧为一个循环
            ActionCOUNTER += 1

            # 如果检测到分心行为
            # 将信息返回到前端ui，使用红色字体来体现
            # 并加ActionCOUNTER减1，以延长循环时间
            for i in lab:
                if (i == "phone"):
                    # window.label_6.setText("<font color=red>正在用手机</font>")
                    # window.label_9.setText("<font color=red>请不要分心</font>")
                    self.main_window.setStatusBar('请不要分心')
                    self.main_window.setTarget_num('正在用手机')
                    engine = pyttsx.init()
                    engine.say('检测到使用手机，请不要分心')
                    engine.runAndWait()
                    if ActionCOUNTER > 0:
                        ActionCOUNTER -= 1
                elif (i == "smoke"):
                    # window.label_7.setText("<font color=red>正在抽烟</font>")
                    # window.label_9.setText("<font color=red>请不要分心</font>")
                    self.main_window.setStatusBar('请不要分心')
                    self.main_window.setTarget_num('正在抽烟')

                    engine = pyttsx.init()
                    engine.say('检测到正在抽烟，请不要分心')
                    engine.runAndWait()

                    if ActionCOUNTER > 0:
                        ActionCOUNTER -= 1
                elif (i == "drink"):
                    self.main_window.setStatusBar('请不要分心')
                    self.main_window.setTarget_num('正在用喝水')

                    engine = pyttsx.init()
                    engine.say('检测到正在喝水，请不要分心')
                    engine.runAndWait()
                    if ActionCOUNTER > 0:
                        ActionCOUNTER -= 1
                else:
                    self.main_window.setTarget_num('正常')
                    self.main_window.setStatusBar('正在检测...')

            # 如果超过15帧未检测到分心行为，将label修改为平时状态
            if ActionCOUNTER == 15:
                ActionCOUNTER = 0

            # 疲劳判断
            # 眨眼判断
            if eye < EYE_AR_THRESH:
                # 如果眼睛开合程度小于设定好的阈值
                # 则两个和眼睛相关的计数器加1
                COUNTER += 1
                Rolleye += 1
            else:
                # 如果连续2次都小于阈值，则表示进行了一次眨眼活动
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    # window.label_3.setText("眨眼次数：" + str(TOTAL))
                    # 重置眼帧计数器
                    COUNTER = 0

            # 哈欠判断，同上
            if mouth > MAR_THRESH:
                mCOUNTER += 1
                Rollmouth += 1
            else:
                # 如果连续3次都小于阈值，则表示打了一次哈欠
                if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                    mTOTAL += 1
                    # window.label_4.setText("哈欠次数：" + str(mTOTAL))
                    # 重置嘴帧计数器
                    mCOUNTER = 0

            # 将画面显示在前端UI上
            show_train = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage_train = QImage(show_train.data, show_train.shape[1], show_train.shape[0], QImage.Format_RGB888)
            self.main_window.res_video.setPixmap(QPixmap.fromImage(showImage_train))

            # 帧率计数加1
            self.frame_count += 1

            # 疲劳模型
            # 疲劳模型以150帧为一个循环
            # 每一帧Roll加1
            Roll += 1
            # 当检测满150帧时，计算模型得分
            if Roll == 10:
                # 计算Perclos模型得分
                perclos = (Rolleye / Roll) + (Rollmouth / Roll) * 0.2
                Percloslist.append(perclos)
                # 归零
                # 重新开始新一轮的检测
                Roll = 0
                Rolleye = 0
                Rollmouth = 0
                print(Percloslist)

            PERCLO_LEN = 20
            if len(Percloslist) == PERCLO_LEN:
                Percloslist.append(Rolls)
                asyncio.get_event_loop().run_until_complete(connect(self.main_window.select_model, Percloslist))
                print(str_mes)
                numbers = re.findall(r'[-+]?\d*\.\d+|\d+', str_mes)
                float_list = [float(x) for x in numbers]
                Rolls += 1
                p = sum(i > 0.38 for i in float_list)

                if p >= len(float_list) / 3:
                    self.main_window.setStatusBar('即将进入疲劳状态')
                    self.main_window.setClass_num('疲劳预警！')
                else:
                    self.main_window.setClass_num('清醒')
                Percloslist = []
                self.main_window.setStatusBar('重新开始执行疲劳检测...')
            elif perclos > 0.38:
                self.main_window.setClass_num("当前疲劳")
                if Roll == 0:
                    engine = pyttsx.init()
                    engine.say('当前处于疲劳状态，请注意休息')
                    engine.runAndWait()
            else:
                self.main_window.setClass_num("清醒")
            # 通过欧拉角判定姿态进行提示
            if degree_count == 0:
                if len(degree) != 0:
                    degrees_last = [degree[4], degree[5], degree[6]]
                    degree_count = degree_count + 1
            else:
                if len(degree) != 0:
                    degrees = [degree[4], degree[5], degree[6]]
                    degrees_diff = np.array(degrees) - np.array(degrees_last)
                    if max(degrees_diff) > 30:
                        self.main_window.setTarget_num('颈部姿势不当')
                        engine = pyttsx.init()
                        engine.say('当前姿势不当，请注意休息')
                        engine.runAndWait()
                    degrees_last = degrees

    # 刷新画面帧率
    def update_fps(self):
        # 计算帧率并更新到fps_label
        fps = self.frame_count
        self.main_window.fps_label.setText(str(fps))
        self.frame_count = 0


def CamConfig_init():
    window.f_type = CamConfig(window)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
