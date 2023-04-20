import cv2
import numpy as np
from collections import deque
import logging
import os
import json
from dataclasses import dataclass
# from move.parts_moving import NozzleMoving, NozzleSetting, PartInfo, Point
from detect.parts_segment import BackgroundModel, ConnectedComponents
from detect.parts_tracker import Tracker, Point, Rectangle, TargetInfo, TargetTrack

class PartsSortingSystem:
    def __init__(self, camera: str, config_file, results_dir = 'run'):
        self.camera = cv2.VideoCapture(camera)
        # self.part_map = {}
        # self.nozzle_moving = NozzleMoving(self)
        # 标注工具
        self.marking = SystemMark()
        # 保存处理结果
        self.is_save = True
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.results_dir = results_dir
        self.config_dir = 'config'
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
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
        
        # 读取配置文件
        if os.path.exists(os.path.join(self.config_dir, 'config.json')):
            with open(os.path.join(self.config_dir, 'config.json'), 'r') as f:
                data = json.load(f)
            self.bbox_belt = (data['belt']['x'], data['belt']['y'], data['belt']['width'], data['belt']['height'])
            self.size_max = (data['max_part']['width'], data['max_part']['height'])
            self.size_min = (data['min_part']['width'], data['min_part']['height'])
        else:
        
            # 框选传送带检测范围
            self.__logger.info("Please select conveyor")
            self.__wait_for_marking()
            _ , self.bbox_belt = self.marking.mark_belt(self.camera.read()[1])

            # 框选最大零件，用于连通域筛选，去除光线干扰
            self.__logger.info("Please select the biggest part")
            self.__wait_for_marking()
            _ , bbox_max = self.marking.mark_part(self.camera.read()[1])
            self.size_max = (bbox_max[2], bbox_max[3])
            
            # 框选最小零件，用于连通域筛选，去除传送带噪声
            self.__logger.info("Please select the smallest part")
            self.__wait_for_marking()
            _ , bbox_min = self.marking.mark_part(self.camera.read()[1])
            self.size_min = (bbox_min[2], bbox_min[3])

            # 构建JSON对象
            data = {
                'belt': {
                    'x': int(self.bbox_belt[0]),
                    'y': int(self.bbox_belt[1]),
                    'width': int(self.bbox_belt[2]),
                    'height': int(self.bbox_belt[3])
                },
                'max_part': {
                    'width': int(self.size_max[0]),
                    'height': int(self.size_max[1])
                },
                'min_part': {
                    'width': int(self.size_min[0]),
                    'height': int(self.size_min[1])
                }
            }

            # 保存传送带区域和最大最小零件尺寸的json配置文件
            with open(os.path.join(self.config_dir, 'config.json'), 'w') as f:
                json.dump(data, f, indent=4)

        # 背景建模
        self.back_model = BackgroundModel(algo='MOG2', history=500, varThreshold=50, detectShadows=False)
        # 连通域分析
        self.connected_components = ConnectedComponents(min_area=self.size_min[0]*self.size_min[1], max_area=self.size_max[0]*self.size_max[1])
        #  # 创建光流对象
        # self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # self.prev_frame = None

        # 卡尔曼滤波
        # self.kalman = cv2.KalmanFilter(dynamParams=4, measureParams=2)
        # # 时间步长根据帧率调整
        # dt = 1.0 / self.camera.get(cv2.CAP_PROP_FPS)
        # # 状态转移矩阵 F. input
        # self.kalman.transitionMatrix = np.array([[1, 0, dt, 0], 
        #                                          [0, 1, 0, dt], 
        #                                          [0, 0, 1, 0], 
        #                                          [0, 0, 0, 1]], dtype=np.float32)
        # # 测量矩阵 H. input
        # self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
        #                                             [0, 1, 0, 0]], dtype=np.float32)
        # # 过程噪声协方差矩阵 Q. input
        # self.kalman.processNoiseCov = np.array([[0.25*dt**4, 0, 0.5*dt**3, 0],
        #                                         [0, 0.25*dt**4, 0, 0.5*dt**3],
        #                                         [0.5*dt**3, 0, dt**2, 0],
        #                                         [0, 0.5*dt**3, 0, dt**2]], dtype=np.float32)
        # # 测量噪声协方差矩阵 R. input
        # self.kalman.measurementNoiseCov = np.array([[1, 0],
        #                                             [0, 1]], dtype=np.float32)
        # # 误差协方差矩阵 P._k|k  KF state var
        # self.kalman.errorCovPost = np.eye(4) * 1000
        # # 初始化状态向量 x^_k|k  KF state var
        # self.kalman.statePost = np.array([[self.bbox_belt[0] + self.bbox_belt[2] / 2], [self.bbox_belt[1]], [0], [0]], dtype=np.float32)

        # 候选目标列表，节点类型为TargetTrack
        self.candidates = list()


    def run(self):
        # 如果指定了保存路径，则创建VideoWriter对象保存连通域分析后的框选结果
        if self.is_save is not None:
            self.belt_video = cv2.VideoWriter(os.path.join(self.results_dir, 'belt.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (int(self.bbox_belt[2]), int(self.bbox_belt[3])))
            self.cc_video = cv2.VideoWriter(os.path.join(self.results_dir, 'cc.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (int(self.bbox_belt[2]), int(self.bbox_belt[3])))

        while True:
            ret, frame = self.camera.read()
            frame_id = self.camera.get(cv2.CAP_PROP_POS_FRAMES)
            if frame is None:
                break

            # 选择传送带检测范围
            belt = frame[int(self.bbox_belt[1]):int(self.bbox_belt[1] + self.bbox_belt[3]), int(self.bbox_belt[0]):int(self.bbox_belt[0] + self.bbox_belt[2])]
            selected_belt = belt.copy()
            candidates_belt = belt.copy()

            # 中值滤波
            belt = cv2.medianBlur(belt, 5)

            # 背景消融
            self.bg_mask = self.back_model.apply(belt)
            if self.is_save:
                # self.bg_video.write(self.bg_mask)
                pass
            if self.is_show:
                cv2.namedWindow("Background Subtraction", cv2.WINDOW_NORMAL)
                cv2.imshow("Background Subtraction", self.bg_mask)

            # 形态学处理去除传送带上的噪点
            pre_proc = self.connected_components.pre_process(self.bg_mask)

            # 连通域分析
            components_num_labels, components_labels, components_stats, components_centroids \
                = self.connected_components.process(pre_proc)
            
            # 框选零件
            # if components_num_labels > 0:
            #     self.components_stats = components_stats
            #     self.components_centroids = components_centroids

            # TODO: 对零件特征点使用光流法来跟踪，目前直接对连通域的坐标进行卡尔曼滤波
            for candidate in self.candidates:
                # 处理物体消失
                if frame_id - candidate.last.timestamp > 10 or candidate.last.center.y < 30:
                    self.candidates.remove(candidate)
            # 遍历当前连通域，与候选目标队列的连通域进行匹配
            for i in range(components_num_labels):
                x, y, w, h, area = components_stats[i]
                centroid = components_centroids[i]
                cur_info = TargetInfo(Rectangle(x, y, w, h), Point(centroid[0], centroid[1]), area, frame_id)

                # 判断是否属于已有物体
                matched = False
                for candidate in self.candidates:
                    # 求中心点位移
                    (dx, dy) = (centroid[0] - candidate.last.center.x, centroid[1] - candidate.last.center.y)
                    # 计算两个矩形的重叠面积
                    intersection = max(0, min(x+w, candidate.last.rect.x+candidate.last.rect.w) - max(x, candidate.last.rect.x)) * \
                                   max(0, min(y+h, candidate.last.rect.y+candidate.last.rect.h) - max(y, candidate.last.rect.y))
                    min_area = min(w*h, candidate.last.rect.w*candidate.last.rect.h)
                    # 重叠比例
                    overlap = intersection / min_area

                    # 当矩形选框的重叠面积大于0.8时，认为是同一个物体
                    if candidate.last.timestamp < frame_id and overlap > 0.8:
                        matched = True
                        # 当x坐标基本不变，y坐标在减小，并且面积基本不变时，认为跟踪成功
                        if abs(dx) < 3 and dy < 0 and abs(candidate.last.area - area) < 0.1 * area:
                            candidate.update(cur_info)
                            break

                # 如果不属于已有物体，则加入候选目标队列
                if not matched:
                    self.candidates.append(TargetTrack(cur_info))

            # TODO: 卡尔曼滤波
            if self.is_show:
                # 形态学处理的结果
                cv2.namedWindow("Pre", cv2.WINDOW_NORMAL)
                cv2.imshow("Pre", pre_proc)
                # 连通域分析的结果
                components = cv2.convertScaleAbs(components_labels)
                components[components > 0] = 255
                cv2.namedWindow("Components", cv2.WINDOW_NORMAL)
                cv2.imshow("Components", components)
                # 框选零件
                if components_num_labels > 0:
                    for i in range(components_num_labels):
                        x, y, w, h, area = components_stats[i]
                        if w > self.size_min[0] and h > self.size_min[1]:
                            cv2.rectangle(selected_belt, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # cv2.putText(selected_belt, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(selected_belt, str(i) + ": (" + str(x) + ", " + str(y) + ")", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(selected_belt, str(w) + "x" + str(h) + ", " + str(area), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.namedWindow("Selected Belt", cv2.WINDOW_NORMAL)
                cv2.imshow("Selected Belt", selected_belt)

                # 显示候选目标队列
                for candidate in self.candidates:
                    # 绘制矩形
                    cv2.rectangle(candidates_belt, (candidate.last.rect.x, candidate.last.rect.y), (candidate.last.rect.x + candidate.last.rect.w, candidate.last.rect.y + candidate.last.rect.h), (0, 255, 0), 2)
                    # 显示连通域编号，中心点坐标，面积，时间戳
                    # cv2.putText(candidates_belt, str(candidate.last.timestamp) + ": (" + str(candidate.last.center.x) + ", " + str(candidate.last.center.y) + ")", (candidate.last.rect.x, candidate.last.rect.y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(candidates_belt, "{0}: ({1:.2f}, {2:.2f})".format(candidate.last.timestamp, candidate.last.center.x, candidate.last.center.y), (candidate.last.rect.x, candidate.last.rect.y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.namedWindow("Candidates Belt", cv2.WINDOW_NORMAL)
                cv2.imshow("Candidates Belt", candidates_belt)

            if self.is_save:
                self.belt_video.write(belt)
                self.cc_video.write(selected_belt)
    

            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break
        
        self.__logger.info("Finished")
        if self.is_save:
            self.__logger.info("Save results to {}".format(self.results_dir))
            self.cc_video.release()
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

    def mark_belt(self, frame: cv2.Mat):
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