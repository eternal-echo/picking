import cv2
import numpy as np
import os
import random
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
        while True:
            ret, frame = self.camera.read()
            frame_id: int = self.camera.get(cv2.CAP_PROP_POS_FRAMES)
            if frame is None or not ret:
                break
            # [预处理模块]
            # - 裁剪检测区域
            belt = frame[int(self.bbox_belt[1]):int(self.bbox_belt[1] + self.bbox_belt[3]),
                    int(self.bbox_belt[0]):int(self.bbox_belt[0] + self.bbox_belt[2])].copy()
            # - 中值滤波
            belt = cv2.medianBlur(belt, 5)
            # [查找模块]
            # - 背景建模
            bg_mask = self.bg_subtractor.apply(belt)
            pre_proc = cv2.medianBlur(bg_mask, 5)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            pre_proc = cv2.morphologyEx(pre_proc, cv2.MORPH_OPEN, kernel)
            # - 连通域分析
            origin_nums, origin_labels, origin_stats, origin_centroids = cv2.connectedComponentsWithStats(pre_proc)
            # 对连通域按面积从大到小排序
            areas = origin_stats[:, 4]
            sorted_areas_indices = np.argsort(areas)[::-1]
            selected_areas_indices = []
            for i in range(len(sorted_areas_indices)):
                if (areas[sorted_areas_indices[i]] < self.max_area
                        # TODO: 边长小于最大尺寸
                        and max(origin_stats[sorted_areas_indices[i], 2], origin_stats[sorted_areas_indices[i], 3]) < max(self.size_max)
                        # 区域的边缘处不再选择
                        and origin_stats[sorted_areas_indices[i], 1] + origin_stats[sorted_areas_indices[i], 3] < pre_proc.shape[0] - 10
                        and origin_stats[sorted_areas_indices[i], 1] > 10
                        and origin_stats[sorted_areas_indices[i], 0] > 5
                        and origin_stats[sorted_areas_indices[i], 0] + origin_stats[sorted_areas_indices[i], 2] < pre_proc.shape[1] - 5):
                    if areas[sorted_areas_indices[i]] < self.min_area:
                        break
                    selected_areas_indices.append(sorted_areas_indices[i])
            obj_num = len(selected_areas_indices)
            obj_stats = origin_stats[selected_areas_indices]
            obj_centroids = origin_centroids[selected_areas_indices]
            assert obj_num == len(obj_stats) == len(obj_centroids)

            trackers = []
            # 连通域分析后存在候选目标
            if obj_num > 0:
            # [检测模块]
                detect_res, detect_names = self.yolo_detect.detect([belt])
                # 获取第一章图片的检测结果
                detect_res = detect_res[0]
                # detect_res[1]: list
                if detect_res is not None:
                    detections = np.zeros((len(detect_res[1]), 5))
                    for i, res in enumerate(detect_res[1]):
                        detect_xmin, detect_ymin, detect_xmax, detect_ymax = res[1]
                        detect_name = res[0]
                        detect_conf = res[2]
                        detection = [detect_xmin, detect_ymin, detect_xmax, detect_ymax, detect_conf]
                        detections[i, :] = detection
                        # [跟踪模块]
                        trackers = self.tracker.update(detections)

                # detections = np.zeros((obj_num, 5))
                # for i in range(obj_num):
                #     detection = [obj_stats[i, 0], obj_stats[i, 1], obj_stats[i, 0] + obj_stats[i, 2],
                #                 obj_stats[i, 1] + obj_stats[i, 3], 1]
                #     detections[i, :] = detection
                # trackers = self.tracker.update(detections)

            # [显示]
            selected_belt = belt.copy()
            connected = np.zeros_like(pre_proc)
            if trackers is not None:
                for d in trackers:
                    # 绘制跟踪目标
                    cv2.rectangle(selected_belt, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 255, 0), 2)
                    # id
                    cv2.putText(selected_belt, str(int(d[4])), (int(d[0]), int(d[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 255, 0), 2)
                
                # 将连通域labels绘制到图像上
                for i in range(obj_num):
                    color = 255 * i / max((obj_num - 1), 1)
                    connected[origin_labels == selected_areas_indices[i]] = color

                # [保存]
                os.makedirs(os.path.join(self.cache_dir, 'track_layer'), exist_ok=True)
                cv2.imwrite(os.path.join(self.cache_dir, 'track_layer', '{}_src.jpg'.format(int(frame_id))),
                            selected_belt)
                cv2.imwrite(os.path.join(self.cache_dir, 'track_layer', '{}_bg.jpg'.format(int(frame_id))),
                            bg_mask)
                cv2.imwrite(os.path.join(self.cache_dir, 'track_layer', '{}_connected.jpg'.format(int(frame_id))),
                    connected)
                
            cv2.namedWindow("Candidates Belt", cv2.WINDOW_NORMAL)
            cv2.namedWindow("bg_mask", cv2.WINDOW_NORMAL)
            cv2.namedWindow("pre_proc", cv2.WINDOW_NORMAL)
            cv2.imshow("Candidates Belt", selected_belt)
            cv2.imshow("bg_mask", bg_mask)
            cv2.imshow("pre_proc", pre_proc)       
            cv2.namedWindow("connected", cv2.WINDOW_NORMAL)
            cv2.imshow("connected", connected)
                

            keyboard = cv2.waitKey(1)
            if keyboard == 'q' or keyboard == 27:
                break

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    system = App(camera_name=r'data\test\num.mp4', config_dir='config', cache_dir='run')
    if system.start() >= 0:
        system.run()
    else:
        print("Failed to start system")