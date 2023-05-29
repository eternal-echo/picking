import cv2
import numpy as np
import os
import time
import concurrent.futures
from config.config import Configurator
from track_layer.sort.sort import Sort
import detect_layer.YoloDetectAPI.yolo_detectAPI as yolo_detectAPI


class Part_Object:
    def __init__(self, id, bbox, frame_id, timestamp):
        self.id = id
        self.bbox = bbox
        self.first_bbox = bbox
        self.frame_id = frame_id
        self.first_frame_id = frame_id
        self.timestamp = timestamp
        self.first_timestamp = timestamp

    def update(self, id, bbox, frame_id, timestamp):
        if self.id == id:
            self.bbox = bbox
            self.frame_id = frame_id
            self.timestamp = timestamp
            return True
        else:
            return False


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
        self.max_area = self.size_max[0] * self.size_max[1]
        self.min_area = self.size_min[0] * self.size_min[1]
        # 区域和待分类零件的映射关系
        self.parts_map = config['parts_map']
        self.nozzle_y0 = config['nozzle_y0']
        self.speed = config['speed']
        self.nozzle_dy = config['spacing']
        self.nozzle_t0 = self.nozzle_y0 / self.speed
        # 开始区域的y坐标
        self.start_y = int(self.bbox_belt[3] * (1 - 1 / 2))
        # 离开区域的y坐标
        self.end_y = int(self.bbox_belt[3] / 10)
        # 中央区域的y坐标
        # self.center_y = int(self.bbox_belt[3] / 2)
        self.center_y = int((self.start_y + self.end_y) / 2)

        print('start_y:', self.start_y)
        print('center_y:', self.center_y)
        print('end_y:', self.end_y)
        print('nozzle_y0:', self.nozzle_y0)
        print('nozzle_dy:', self.nozzle_dy)
        print('nozzle_t0:', self.nozzle_t0)
        print('speed:', self.speed)
        print('max_area:', self.max_area)
        print('min_area:', self.min_area)


        # 背景建模
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        # SORT多目标跟踪
        self.tracker = Sort(max_age=2, min_hits=3, iou_threshold=0.3)
        # YOLOv5检测
        self.yolo_detect = yolo_detectAPI.DetectAPI(weights='num4.pt', conf_thres=0.5, iou_thres=0.5)

        # 线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        self.obj_map = {}

        self.part_cnt = 0

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
            # TODO: 对连通域按面积从大到小排序是否必要，可以直接遍历
            areas = origin_stats[:, 4]
            sorted_areas_indices = np.argsort(areas)[::-1]
            selected_areas_indices = []
            for i in range(len(sorted_areas_indices)):
                if (areas[sorted_areas_indices[i]] < self.max_area
                        # TODO: 边长小于最大尺寸
                        and max(origin_stats[sorted_areas_indices[i], 2],
                                origin_stats[sorted_areas_indices[i], 3]) < max(self.size_max)
                        # 区域的边缘处不再选择
                        and origin_stats[sorted_areas_indices[i], 1] + origin_stats[sorted_areas_indices[i], 3] <
                        pre_proc.shape[0] - 10
                        and origin_stats[sorted_areas_indices[i], 1] > 10
                        and origin_stats[sorted_areas_indices[i], 0] > 5
                        and origin_stats[sorted_areas_indices[i], 0] + origin_stats[sorted_areas_indices[i], 2] <
                        pre_proc.shape[1] - 5):
                    if areas[sorted_areas_indices[i]] < self.min_area:
                        break
                    selected_areas_indices.append(sorted_areas_indices[i])
            obj_num = len(selected_areas_indices)
            obj_stats = origin_stats[selected_areas_indices]
            obj_centroids = origin_centroids[selected_areas_indices]
            assert obj_num == len(obj_stats) == len(obj_centroids)

            trackers = []
            # [跟踪模块]
            trackers = []
            if obj_num > 0:
                detections = np.zeros((obj_num, 5))
                for i in range(obj_num):
                    detection = [obj_stats[i, 0], obj_stats[i, 1], obj_stats[i, 0] + obj_stats[i, 2],
                                 obj_stats[i, 1] + obj_stats[i, 3], 1]
                    detections[i, :] = detection
                trackers = self.tracker.update(detections)
            # [运动轨迹检测]
            if len(trackers):
                for d in trackers:
                    xmin, ymin, xmax, ymax, obj_id = d
                    # 属于已有目标
                    if obj_id in self.obj_map.keys():
                        # 零件移动到中央区域 且 之前不在中央区域(ymax/2)
                        if ymin < self.center_y and self.start_y > self.obj_map[obj_id].bbox[1] > self.center_y:
                            # [Event]: 目标进入中央区域
                            print("[{}] 目标{}进入中央区域".format(time.time(), obj_id))
                            detect_future = self.executor.submit(self.__detect_move_task, belt, obj_id)
                        # 零件移动到结束区域 且 之前不在结束区域(0)
                        if ymin < self.end_y and self.center_y > self.obj_map[obj_id].bbox[1] > self.end_y:
                            self.part_cnt += 1
                            # [Event]: 目标进入结束区域
                            print("[{}] 目标{}进入结束区域".format(time.time(), obj_id))
                        # 更新目标信息
                        self.obj_map[obj_id].update(id=obj_id, bbox=[xmin, ymin, xmax, ymax],
                                                    frame_id=frame_id, timestamp=time.time())
                        # print("目标{}更新{}".format(obj_id, self.obj_map[obj_id].bbox))
                    # 属于新目标
                    else:
                        # 新目标的起始位置大于设置的下边界阈值start_y时，才认为是零件目标(ymax)
                        if ymin > self.start_y:
                            # 添加新目标
                            new_obj = Part_Object(id=obj_id, bbox=[xmin, ymin, xmax, ymax],
                                                  frame_id=frame_id, timestamp=time.time())
                            self.obj_map[obj_id] = new_obj
                            # [Event]: 目标出现
                            print("[{}] 目标{}出现".format(time.time(), obj_id))
                        # else: 认为是光斑

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
                    color = 255 * (i + 1) / (obj_num)
                    connected[origin_labels == selected_areas_indices[i]] = color

                # 绘制起始、中央和结束线
                cv2.line(selected_belt, (0, self.start_y), (selected_belt.shape[1], self.start_y), (0, 0, 255), 2)
                cv2.line(selected_belt, (0, self.center_y), (selected_belt.shape[1], self.center_y), (0, 0, 255), 2)
                cv2.line(selected_belt, (0, self.end_y), (selected_belt.shape[1], self.end_y), (0, 0, 255), 2)

                # 零件计数
                cv2.putText(selected_belt, "Part Count: {}".format(self.part_cnt), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 255, 0), 2)

                # # [保存]
                # os.makedirs(os.path.join(self.cache_dir, 'track_layer'), exist_ok=True)
                # cv2.imwrite(os.path.join(self.cache_dir, 'track_layer', '{}_src.jpg'.format(int(frame_id))),
                #             selected_belt)
                # cv2.imwrite(os.path.join(self.cache_dir, 'track_layer', '{}_bg.jpg'.format(int(frame_id))),
                #             bg_mask)
                # cv2.imwrite(os.path.join(self.cache_dir, 'track_layer', '{}_connected.jpg'.format(int(frame_id))),
                #     connected)

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

    def __detect_move_task(self, belt, tracker_id):
        detect_res, detect_names = self.yolo_detect.detect([belt])
        detect_res = detect_res[0]
        if detect_res:
            print("目标{}的检测结果为{}".format(tracker_id, detect_res[1]))
            detection_name = detect_res[1][0][0]
            # 查找零件的对应分拣区域为喷嘴 i
            area_id: int = -1
            for area, parts in self.parts_map.items():
                if detection_name in parts:
                    area_id = int(area)
                    print("Part {} in area {}".format(detection_name, area_id))
                    break
            if area_id <= -1:
                print("Part {} not in areas".format(detection_name))
                return

            # 零件当前坐标 y
            obj_y = abs(self.bbox_belt[3] - (self.obj_map[tracker_id].bbox[1] + self.obj_map[tracker_id].bbox[3]) / 2)
            # 零件在当前位置时的时间戳
            obj_t = self.obj_map[tracker_id].timestamp
            # 零件首次坐标 y'
            obj_y_ = abs(self.bbox_belt[3] - (self.obj_map[tracker_id].first_bbox[1] + self.obj_map[tracker_id].first_bbox[3]) / 2)
            # 零件在首次出现的时间戳
            obj_t_ = self.obj_map[tracker_id].first_timestamp
            # 零件的速度 px/frame
            obj_v = abs(obj_y - obj_y_) / abs(obj_t - obj_t_)
            # 零件的运动时间
            obj_t_move = abs(obj_t - obj_t_)
            # 更新当前位置
            obj_yy = abs(self.bbox_belt[3] - (self.obj_map[tracker_id].bbox[1] + self.obj_map[tracker_id].bbox[3]) / 2)
            # 零件从y=0移动到当前位置的时间
            obj_t_move0 = obj_yy / obj_v
            print("[{}]: y': {}".format(obj_t_, obj_y_))
            print("[{}]: y: {}".format(obj_t, obj_y))
            print("t0: {}, t_move0: {}, t_move: {}, t: {}, v: {}".format(self.nozzle_t0, obj_t_move0, obj_t_move, obj_t, obj_v))

            # 定时时间
            delay = self.nozzle_t0 - obj_t_move0
            # 喷嘴运行时间
            duration = self.nozzle_dy / self.speed / 2
            print("[{}] delay: {} duration: {}".format(time.time(), delay, duration))

            # 延时
            time.sleep(delay - duration / 2)
            # 喷嘴运行
            print("[{}] 喷嘴运行".format(time.time()))
            time.sleep(duration)
            # 喷嘴停止
            print("[{}] 喷嘴停止".format(time.time()))


if __name__ == '__main__':
    system = App(camera_name=r'data\test\multi_track.mp4', config_dir='config', cache_dir='run')
    # system = App(camera_name='1', config_dir='config', cache_dir='run')
    if system.start() >= 0:
        system.run()
    else:
        print("Failed to start system")