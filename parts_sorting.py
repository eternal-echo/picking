import cv2
import numpy as np
import logging
import os
from move.parts_moving import NozzleMoving, NozzleSetting, PartInfo, Point
from detect.parts_segment import BackgroundModel

class PartsSortingSystem:
    def __init__(self, camera: str, results_dir = 'run'):
        self.camera = cv2.VideoCapture(camera)
        # self.part_map = {}
        # self.nozzle_moving = NozzleMoving(self)
        # 标注工具
        self.marking = SystemMark()
        # 背景建模
        self.back_model = BackgroundModel(algo='MOG2', history=500, varThreshold=50, detectShadows=True)
        # 保存处理结果
        self.is_save = True
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.results_dir = results_dir
        # 显示中间过程
        self.is_show = True
        # 初始化日志系统
        self.__logger = logging.getLogger('system')
        self.__logger.setLevel(logging.DEBUG)
        self.__console_handler = logging.StreamHandler()
        self.__console_handler.setLevel(logging.DEBUG)
        self.__logger.addHandler(self.__console_handler)

    def start(self):
        # 打开摄像头
        if not self.camera.isOpened():
            self.__logger.error("Failed to open camera")
            return
        
        # 框选传送带检测范围
        self.__logger.info("Please select conveyor")
        self.__wait_for_marking()
        _ , self.conveyor_rect = self.marking.mark_conveyor(self.camera.read()[1])

        # 框选最大零件，用于连通域筛选，去除光线干扰
        self.__logger.info("Please select the biggest part")
        self.__wait_for_marking()
        _ , self.max_part_rect = self.marking.mark_part(self.camera.read()[1])
        
        # 框选最小零件，用于连通域筛选，去除传送带噪声
        self.__logger.info("Please select the smallest part")
        self.__wait_for_marking()
        _ , self.min_part_rect = self.marking.mark_part(self.camera.read()[1])

    def run(self):
        # 如果指定了保存路径，则创建VideoWriter对象保存背景消融的结果
        if self.is_save is not None:
            # TODO: 背景mask暂时不保存为视频
            # fps = int(self.camera.get(cv2.CAP_PROP_FPS))
            # frame_size = (int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            # self.bg_video = cv2.VideoWriter(os.path.join(self.results_dir, 'bg.mp4'), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frame_size)
            pass

        while True:
            ret, frame = self.camera.read()
            if frame is None:
                break

            # 选择传送带检测范围
            conveyor = frame[int(self.conveyor_rect[1]):int(self.conveyor_rect[1] + self.conveyor_rect[3]), int(self.conveyor_rect[0]):int(self.conveyor_rect[0] + self.conveyor_rect[2])]

            # 背景消融
            self.bg_mask = self.back_model.apply(conveyor)
            if self.is_save:
                # self.bg_video.write(self.bg_mask)
                pass
            if self.is_show:
                cv2.namedWindow("Background Subtraction", cv2.WINDOW_NORMAL)
                cv2.imshow("Background Subtraction", self.bg_mask)

            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break
        
        self.__logger.info("Finished")
        if self.is_save:
            self.__logger.info("Save results to {}".format(self.results_dir))
            # self.bg_video.release()
        self.camera.release()
        cv2.destroyAllWindows()

    # 等待用户开始标注
    def __wait_for_marking(self):
        self.__logger.info("Press space to start marking")
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            cv2.namedWindow("wait for marking", cv2.WINDOW_NORMAL)
            cv2.imshow("wait for marking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break


class SystemMark:
    def __init__(self):
        self.conveyor_points = []

    def mark_conveyor(self, frame: cv2.Mat):
        """在指定帧中通过两条直线标记出传送带的位置

        Args:
            frame (cv2.Mat): 传送带所在的帧

        Returns:
            frame (cv2.Mat): 标记好传送带位置的帧
            conveyor_lines (list): 两条直线的起点和终点坐标
        """
        # 显示原始图像，窗口大小可调整
        cv2.namedWindow("Mark Conveyor", cv2.WINDOW_NORMAL)
        cv2.imshow("Mark Conveyor", frame)

        # ROI选择传送带范围
        rect = cv2.selectROI("Mark Conveyor", frame, False, False)
        conveyor = frame[int(rect[1]):int(rect[1] + rect[3]), int(rect[0]):int(rect[0] + rect[2])]

        return conveyor, rect

    def mark_part(self, frame: cv2.Mat):
        """在指定帧中通过ROI框选零件，获取零件的外接矩形

        Args:
            frame (cv2.Mat): 零件所在的帧

        Returns:
            part (cv2.Mat): ROI框选的零件图像
            rect (tuple): 零件的外接矩形坐标
        """
        # 显示原始图像
        cv2.namedWindow("Mark Part", cv2.WINDOW_NORMAL)
        cv2.imshow("Mark Part", frame)

        # ROI选择零件
        rect = cv2.selectROI("Mark Part", frame, fromCenter=False, showCrosshair=True)
        part = frame[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])] # 获取ROI区域的图像

        return part, rect