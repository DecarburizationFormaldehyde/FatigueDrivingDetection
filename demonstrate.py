# 主函数
import sys
import os
from glob import glob
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PySide6.QtCore import QDir, QTimer, Slot
from PySide6.QtGui import QPixmap, QImage
from ui.demo import Ui_MainWindow
from UIFunctions import *
import cv2
import json

import myframe
# 定义变量

# 眼睛闭合判断
EYE_AR_THRESH = 0.15  # 眼睛长宽比
EYE_AR_CONSEC_FRAMES = 2  # 闪烁阈值

# 嘴巴开合判断
MAR_THRESH = 0.65  # 打哈欠长宽比
MOUTH_AR_CONSEC_FRAMES = 3  # 闪烁阈值

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


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # 打开文件类型，用于类的定义
        self.f_type = 0

        # 原窗口初始化
        self.setAttribute(Qt.WA_TranslucentBackground)  # rounded transparent
        self.setWindowFlags(Qt.FramelessWindowHint)  # Set window flag: hide window borders
        UIFuncitons.uiDefinitions(self)
        # Show module shadows
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))
        # 读取models文件
        # read model folder
        self.pt_list = os.listdir('./models')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))  # sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        # 计时器用于刷新模型combobox
        self.Qtimer_ModelBox = QTimer(self)  # Timer: Monitor model file changes every 2 seconds
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # Model parameters
        self.model_box.currentTextChanged.connect(self.change_model)
        self.iou_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'iou_spinbox'))  # iou box
        self.iou_slider.valueChanged.connect(lambda x: self.change_val(x, 'iou_slider'))  # iou scroll bar
        self.conf_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(lambda x: self.change_val(x, 'conf_slider'))  # conf scroll bar
        self.speed_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'speed_spinbox'))  # speed box
        self.speed_slider.valueChanged.connect(lambda x: self.change_val(x, 'speed_slider'))  # speed scroll bar

        # Prompt window initialization
        # 初始label
        self.Class_num.setText('清醒')
        self.Target_num.setText('正常')
        self.fps_label.setText('--')
        self.Model_name.setText(self.model_box.currentText())

        # Select detection source
        # 左侧功能框暂时不用
        self.src_file_button.clicked.connect(self.open_src_file)  # select local file
        self.src_cam_button.clicked.connect(CamConfig_init)  # chose_cam
        self.src_rtsp_button.clicked.connect(self.chose_rtsp)  # chose_rtsp
        # self.show_status("The function has not yet been implemented.")

        # start testing button
        # 开始检测按钮
        self.run_button.clicked.connect(self.run_or_continue)  # pause/start
        self.stop_button.clicked.connect(self.stop)  # termination

        # Other function buttons
        # self.save_res_button.toggled.connect(self.is_save_res)  # save image option
        # self.save_txt_button.toggled.connect(self.is_save_txt)  # Save label option
        # 左侧展开按钮
        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))  # left navigation button
        # 右侧设置按钮
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))  # top right settings button

        # initialization
        # 初始化完成
        self.load_config()
        # self.window_init()

    def run_or_continue(self):
        self.show_status('Please select a video source before starting detection...')
        # if self.yolo_predict.source == '':
        #     self.show_status('Please select a video source before starting detection...')
        #     self.run_button.setChecked(False)
        # else:
        #     self.yolo_predict.stop_dtc = False
        #     if self.run_button.isChecked():
        #         self.run_button.setChecked(True)    # start button
        #         self.save_txt_button.setEnabled(False)  # It is forbidden to check and save after starting the detection
        #         self.save_res_button.setEnabled(False)
        #         self.show_status('Detecting...')
        #         self.yolo_predict.continue_dtc = True   # Control whether Yolo is paused
        #         if not self.yolo_thread.isRunning():
        #             self.yolo_thread.start()
        #             self.main2yolo_begin_sgl.emit()
        #
        #     else:
        #         self.yolo_predict.continue_dtc = False
        #         self.show_status("Pause...")
        #         self.run_button.setChecked(False)    # start button

    # bottom status bar information
    def show_status(self, msg):
        self.status_bar.setText(msg)
        if msg == 'Detection completed' or msg == '检测完成':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            # 终止正在运行的YOLOv程序
            # if self.yolo_thread.isRunning():
            #     self.yolo_thread.quit()         # end process
        elif msg == 'Detection terminated!' or msg == '检测终止':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            # if self.yolo_thread.isRunning():
            #     self.yolo_thread.quit()         # end process
            self.pre_video.clear()  # clear image display
            # 删除了右边的窗体
            # self.res_video.clear()
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

    #
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
            # self.yolo_predict.source = name
            # self.show_status('Load File：{}'.format(os.path.basename(name)))
            # config['open_fold'] = os.path.dirname(name)
            # config_json = json.dumps(config, ensure_ascii=False, indent=2)
            # with open(config_file, 'w', encoding='utf-8') as f:
            #     f.write(config_json)
            # self.stop()
            print('打开了文件')

    # Select camera source----  have one bug
    def chose_cam(self):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='Note', text='loading camera...', time=2000, auto=True).exec()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.src_cam_button.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).x()
            y = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).y()
            y = y + self.src_cam_button.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec(pos)
            if action:
                self.yolo_predict.source = action.text()
                self.show_status('Loading camera：{}'.format(action.text()))

        except Exception as e:
            self.show_status('%s' % e)

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
        # self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()), self.rtsp_window.close())
        # 与之前一样的含义，只是在提交url地址之后关闭confirm窗口
        self.rtsp_window.rtspButton.clicked.connect(
            lambda ip=self.rtsp_window.rtspEdit.text(): (self.load_rtsp(ip), self.rtsp_window.close()))

    # load network sources
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='提示', text='加载 rtsp...', time=1000, auto=True).exec()
            self.yolo_predict.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.show_status('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.show_status('%s' % e)

    # Save test result button--picture/video
    # def is_save_res(self):
    #     if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
    #         self.show_status('NOTE: Run image results are not saved.')
    #         self.yolo_predict.save_res = False
    #     elif self.save_res_button.checkState() == Qt.CheckState.Checked:
    #         self.show_status('NOTE: Run image results will be saved.')
    #         self.yolo_predict.save_res = True
    #
    # # Save test result button -- label (txt)
    # def is_save_txt(self):
    #     if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
    #         self.show_status('NOTE: Labels results are not saved.')
    #         self.yolo_predict.save_txt = False
    #     elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
    #         self.show_status('NOTE: Labels results will be saved.')
    #         self.yolo_predict.save_txt = True

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
        # 终止yolov程序
        # if self.yolo_thread.isRunning():
        #     self.yolo_thread.quit()         # end thread
        # self.yolo_predict.stop_dtc = True
        self.run_button.setChecked(False)  # start key recovery
        self.save_res_button.setEnabled(True)  # Ability to use the save button
        self.save_txt_button.setEnabled(True)  # Ability to use the save button
        self.pre_video.clear()  # clear image display
        # 删除了右边的窗体
        self.res_video.clear()  # clear image display
        self.progress_bar.setValue(0)
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
            # self.yolo_predict.speed_thres = x  # ms

    # change model
    def change_model(self, x):
        self.select_model = self.model_box.currentText()
        # self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        # self.show_status('Change Model：%s' % self.select_model)
        self.Model_name.setText(self.select_model)

    # label result
    # def show_labels(self, labels_dic):
    #     try:
    #         self.result_label.clear()
    #         labels_dic = sorted(labels_dic.items(), key=lambda x: x[1], reverse=True)
    #         labels_dic = [i for i in labels_dic if i[1]>0]
    #         result = [' '+str(i[0]) + '：' + str(i[1]) for i in labels_dic]
    #         self.result_label.addItems(result)
    #     except Exception as e:
    #         self.show_status(e)

    # Cycle monitoring model file changes 实时更新模型库的情况
    def ModelBoxRefre(self):
        pt_list = os.listdir('./models')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))
        # It must be sorted before comparing, otherwise the list will be refreshed all the time
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    # Get the mouse position (used to hold down the title bar and drag the window)记录鼠标位置
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize the adjustment when dragging the bottom and right edges of the window size调整窗口大小
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)


    # def window_init(self):
    #     # 设置控件属性
    #     # 设置label的初始值
    #     # self.label.setText("请打开摄像头")
    #     # self.label_2.setText("疲劳检测：")
    #     # self.label_3.setText("眨眼次数：0")
    #     # self.label_4.setText("哈欠次数：0")
    #     # self.label_5.setText("行为检测：")
    #     # self.label_6.setText("手机")
    #     # self.label_7.setText("抽烟")
    #     # self.label_8.setText("喝水")
    #     # self.label_9.setText("是否存在分心行为")
    #     # self.label_10.setText("是否为疲劳状态")
    #     # self.menu.setTitle("打开")
    #     # self.actionOpen_camera.setText("打开摄像头")
    #     # # 菜单按钮 槽连接 到函数
    #     # self.actionOpen_camera.triggered.connect(CamConfig_init)
    #     # 自适应窗口缩放
    #     self.label.setScaledContents(True)
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

# 定义摄像头类
class CamConfig:
    def __init__(self):
        # Ui_MainWindow.printf(window, "正在打开摄像头请稍后...")
        MainWindow.setdemo_window(window, "正在打开摄像头请稍后...")
        # 设置时钟
        self.v_timer = QTimer()
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap:
            # Ui_MainWindow.printf(window, "打开摄像头失败")
            MainWindow.setdemo_window(window, "打开摄像头失败")
            return
        # 设置定时器周期，单位毫秒
        self.v_timer.start(20)
        # 连接定时器周期溢出的槽函数，用于显示一帧视频
        self.v_timer.timeout.connect(self.show_pic)
        # 在前端UI输出提示信息
        # Ui_MainWindow.printf(window, "载入成功，开始运行程序")
        # Ui_MainWindow.printf(window, "")
        # Ui_MainWindow.printf(window, "开始执行疲劳检测...")
        # window.statusbar.showMessage("正在使用摄像头...")
        MainWindow.setdemo_window(window, "载入成功，开始运行程序")
        MainWindow.setStatusBar(window, "正在使用摄像头...")

    def show_pic(self):
        # 全局变量
        # 在函数中引入定义的全局变量
        global EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, MAR_THRESH, MOUTH_AR_CONSEC_FRAMES, COUNTER, TOTAL, mCOUNTER, mTOTAL, ActionCOUNTER, Roll, Rolleye, Rollmouth

        # 读取摄像头的一帧画面
        success, frame = self.cap.read()
        if success:
            # 检测
            # 将摄像头读到的frame传入检测函数myframe.frametest()
            ret, frame = myframe.frametest(frame)
            lab, eye, mouth = ret
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
                    MainWindow.setTarget_num(window, '正在用手机')
                    if ActionCOUNTER > 0:
                        ActionCOUNTER -= 1
                elif (i == "smoke"):
                    # window.label_7.setText("<font color=red>正在抽烟</font>")
                    # window.label_9.setText("<font color=red>请不要分心</font>")
                    MainWindow.setTarget_num(window, '正在抽烟')
                    if ActionCOUNTER > 0:
                        ActionCOUNTER -= 1
                elif (i == "drink"):
                    # window.label_8.setText("<font color=red>正在用喝水</font>")
                    # window.label_9.setText("<font color=red>请不要分心</font>")
                    MainWindow.setTarget_num(window, '正在用喝水')
                    if ActionCOUNTER > 0:
                        ActionCOUNTER -= 1
                else:
                    MainWindow.setTarget_num(window, '正常')


            # 如果超过15帧未检测到分心行为，将label修改为平时状态
            if ActionCOUNTER == 15:
                # MainWindow.setTarget_num(window, '正常')
                # window.label_6.setText("手机")
                # window.label_7.setText("抽烟")
                # window.label_8.setText("喝水")
                # window.label_9.setText("")
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
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            window.label.setPixmap(QPixmap.fromImage(showImage))

            # 疲劳模型
            # 疲劳模型以150帧为一个循环
            # 每一帧Roll加1
            Roll += 1
            # 当检测满150帧时，计算模型得分
            if Roll == 150:
                # 计算Perclos模型得分
                perclos = (Rolleye / Roll) + (Rollmouth / Roll) * 0.2
                # 在前端UI输出perclos值
                # Ui_MainWindow.printf(window, "过去150帧中，Perclos得分为" + str(round(perclos, 3)))
                # 当过去的150帧中，Perclos模型得分超过0.38时，判断为疲劳状态
                if perclos > 0.38:
                    # Ui_MainWindow.printf(window, "当前处于疲劳状态")
                    # window.label_10.setText("<font color=red>疲劳！！！</font>")
                    # Ui_MainWindow.printf(window, "")
                    MainWindow.setClass_num(window, '疲劳！')
                else:
                    # Ui_MainWindow.printf(window, "当前处于清醒状态")
                    # window.label_10.setText("清醒")
                    # Ui_MainWindow.printf(window, "")
                    MainWindow.setClass_num(window, '疲劳！')

                # 归零
                # 将三个计数器归零
                # 重新开始新一轮的检测
                Roll = 0
                Rolleye = 0
                Rollmouth = 0
                # Ui_MainWindow.printf(window, "重新开始执行疲劳检测...")


def CamConfig_init():
    window.f_type = CamConfig()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()