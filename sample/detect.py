import cv2
import numpy as np
import os
import random
import csv
import time
from config.config import Configurator
from track_layer.sort.sort import Sort
import detect_layer.YoloDetectAPI.yolo_detectAPI as yolo_detectAPI



class App:
    def __init__(self, camera_name='1', config_dir='config', cache_dir='run'):
        if camera_name.isdigit():
            camera_name = int(camera_name)
        self.camera = cv2.VideoCapture(camera_name)
        # 配置
        self.configurator = Configurator(self.camera, config_dir, cache_dir)
        self.config_dir = config_dir
        self.cache_dir = cache_dir
        # 获取传送带区域和最大最小零件等配置信息
        config = self.configurator.get_config()
        self.bbox_belt = config['bbox_belt']
        self.size_max = config['size_max']
        self.size_min = config['size_min']
        self.max_area = self.size_max[0]*self.size_max[1]
        self.min_area = self.size_min[0]*self.size_min[1]

        # 背景建模
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        # SORT多目标跟踪
        self.tracker = Sort(max_age=2, min_hits=3, iou_threshold=0.3)
        # YOLOv5检测
        self.yolo_detect = yolo_detectAPI.DetectAPI(weights='num4.pt', conf_thres=0.5, iou_thres=0.5)

        self.save_idx = 0

    def start(self):
        # 打开摄像头
        if not self.camera.isOpened():
            print("Failed to open camera")
            return -1
        # TODO: 喷嘴控制
        return 0

    def run(self):
        times = []
        while True:
            ret, frame = self.camera.read()
            frame_id: int = self.camera.get(cv2.CAP_PROP_POS_FRAMES)
            if frame is None or not ret:
                break
            if frame_id % 30 != 0:
                continue
            # [预处理模块]
            # - 裁剪检测区域
            belt = frame[int(self.bbox_belt[1]):int(self.bbox_belt[1] + self.bbox_belt[3]),
                    int(self.bbox_belt[0]):int(self.bbox_belt[0] + self.bbox_belt[2])].copy()
            # - 中值滤波
            belt = cv2.medianBlur(belt, 5)

            # [检测模块]
            start_time = time.time()
            detect_res, detect_names = self.yolo_detect.detect([belt])
            # print('res', detect_res[0][1])
            # print('name', detect_names)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print('detect time:', elapsed_time)
            times.append(elapsed_time)

            # [显示]
            selected_belt = belt.copy()
            for res in detect_res[0][1]:
                # - 画框
                xmin, ymin, xmax, ymax = res[1]
                cv2.rectangle(selected_belt, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # - 画标签
                label = res[0]
                conf = res[2]
                cv2.putText(selected_belt, str(label), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(selected_belt, str(conf), (xmin, ymin+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.namedWindow("Candidates Belt", cv2.WINDOW_NORMAL)
            cv2.imshow("Candidates Belt", selected_belt)

            keyboard = cv2.waitKey(1)
            if keyboard == 'q' or keyboard == 27:
                break

        self.camera.release()
        cv2.destroyAllWindows()
        with open(os.path.join(self.cache_dir, 'detect.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(times)

if __name__ == '__main__':
    system = App(camera_name=r'data\test\num.mp4', config_dir='config', cache_dir='run')
    if system.start() >= 0:
        system.run()
    else:
        print("Failed to start system")