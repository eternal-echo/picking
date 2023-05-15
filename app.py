import cv2
import os
import logging
import time
import concurrent.futures
import asyncio
from config.config import Configurator
from detection_layer.detection_layer import DetectionLayer
from tracking_layer.tracking_layer import BackgroundModel, ConnectedComponents, Point, Rectangle, TargetInfo, TargetTrack
from move_layer.nozzle_controller import NozzleController

class ApplicationLayer:
    def __init__(self, camera_name='1', config_dir='config', cache_dir='run'):
        if camera_name.isdigit():
            camera_name = int(camera_name)
        self.camera = cv2.VideoCapture(camera_name)
        # 配置
        self.configurator = Configurator(self.camera, config_dir, cache_dir)
        self.config_dir = config_dir
        self.cache_dir = cache_dir
        # 初始化日志系统
        self.__logger = logging.getLogger('system')
        self.__logger.setLevel(logging.DEBUG)
        self.__console_handler = logging.StreamHandler()
        self.__console_handler.setLevel(logging.DEBUG)
        self.__logger.addHandler(self.__console_handler)
        # 保存处理结果
        self.is_save = True
        # 显示中间过程
        self.is_show = True
        # 创建检测层对象
        self.detector = DetectionLayer()
        # 创建喷嘴控制器对象
        self.nozzle_controller = NozzleController(port="COM12")

    def start(self) -> int:
        # 打开摄像头
        if not self.camera.isOpened():
            self.__logger.error("Failed to open camera")
            return -1
        if not self.nozzle_controller.connect():
            self.__logger.error("Failed to connect nozzle controller")
            return -1
        # 获取传送带区域和最大最小零件等配置信息
        config = self.configurator.get_config()
        self.bbox_belt = config['bbox_belt']
        self.size_max = config['size_max']
        self.size_min = config['size_min']
        self.time0 = config['time0']
        self.speed = config['speed']
        self.spacing = config['spacing']
        # 区域和待分类零件的映射关系
        self.parts_map = config['parts_map']
        # 背景建模
        self.back_model = BackgroundModel(algo='MOG2', history=500, varThreshold=50, detectShadows=False)
        # 连通域分析器
        self.connected_components_analyzer = ConnectedComponents(min_area=self.size_min[0]*self.size_min[1], max_area=self.size_max[0]*self.size_max[1])
        # 候选目标列表，节点类型为TargetTrack
        self.candidates = list()
        # 离开区域的y坐标
        self.end_y = int(self.bbox_belt[3] / 15)
        # 开始区域的y坐标
        self.start_y = int(self.bbox_belt[3] * (1 - 1 / 4))
        return 0

    def run(self):
        shot_idx = 4139
        # 如果指定了保存路径，则创建VideoWriter对象保存连通域分析后的框选结果
        if self.is_save:
            self.belt_video = cv2.VideoWriter(os.path.join(self.cache_dir, 'belt.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (int(self.bbox_belt[2]), int(self.bbox_belt[3])))
            self.track_video = cv2.VideoWriter(os.path.join(self.cache_dir, 'track.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (int(self.bbox_belt[2]), int(self.bbox_belt[3])))

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            while True:
                ret, frame = self.camera.read()
                frame_id = self.camera.get(cv2.CAP_PROP_POS_FRAMES)
                if frame is None:
                    break

                # 选择传送带检测范围
                belt = frame[int(self.bbox_belt[1]):int(self.bbox_belt[1] + self.bbox_belt[3]), int(self.bbox_belt[0]):int(self.bbox_belt[0] + self.bbox_belt[2])]
                selected_belt = belt.copy()

                # 中值滤波
                belt = cv2.medianBlur(belt, 5)

                # 背景消融
                self.bg_mask = self.back_model.apply(belt)

                # 形态学处理去除传送带上的噪点
                pre_proc = self.connected_components_analyzer.pre_process(self.bg_mask)

                # 连通域分析
                components_num_labels, components_labels, components_stats, components_centroids \
                    = self.connected_components_analyzer.process(pre_proc)

                # 遍历当前连通域，与候选目标队列的连通域进行匹配
                for i in range(components_num_labels):
                    x, y, w, h, area = components_stats[i]
                    centroid = components_centroids[i]
                    cur_info = TargetInfo(Rectangle(x, y, w, h), Point(centroid[0], centroid[1]), area, time.time())

                    # 判断是否属于已有物体
                    matched = False
                    for candidate in self.candidates:
                        # 求中心点位移
                        (dx, dy) = (centroid[0] - candidate.last.center.x, centroid[1] - candidate.last.center.y)
                        # 计算两个矩形的重叠面积
                        intersection = max(0, min(x+w, candidate.last.rect.x+candidate.last.rect.w) - max(x, candidate.last.rect.x)) * \
                                        max(0, min(y+h, candidate.last.rect.y+candidate.last.rect.h) - max(y, candidate.last.rect.y))
                        min_area = min(w*h, candidate.last.rect.w*candidate.last.rect.h)
                        # 计算重叠比例
                        overlap = intersection / min_area

                        # 当矩形选框的重叠面积大于0.8时，认为是同一个物体
                        if candidate.last.timestamp < time.time() and overlap > 0.75:
                            matched = True
                            # 当x坐标基本不变，y坐标在减小，并且面积基本不变时，认为跟踪成功
                            if abs(dx) < 3 and dy < 0:
                                candidate.update(cur_info)
                                break
                    
                    # 当没有匹配到帧间目标，并且起始位置大于设置的下边界阈值start_y时，认为是新目标
                    if (not matched) and y > self.start_y:
                        tracker = TargetTrack(cur_info)
                        self.candidates.append(tracker)
                        # [obj] 目标出现
                        detect_future = executor.submit(self.__detect_move_task, belt, tracker)


                # 过期检查
                for candidate in self.candidates:
                    # 目标离开上边界
                    if candidate.last.rect.y < self.end_y:
                        # [obj] 目标离开
                        candidate.is_expired = True
                        # src_obj = selected_belt[y1:y2, x1:x2]
                        # # 将src_obj保存到缓存文件夹
                        # cv2.imwrite(os.path.join(self.cache_dir, "part{}.jpg".format(frame_id)), src_obj)
                        # result = self.detector.detect(src_obj)
                        continue
                    # # 处理过期物体
                    # if time.time() - candidate.last.timestamp > 0.5:
                    #     self.candidates.remove(candidate)
                
                if self.is_show:
                    # 显示候选目标队列
                    for candidate in self.candidates:
                        # 绘制矩形
                        cv2.rectangle(selected_belt, (candidate.last.rect.x, candidate.last.rect.y), (candidate.last.rect.x + candidate.last.rect.w, candidate.last.rect.y + candidate.last.rect.h), (0, 255, 0), 2)
                        # 显示连通域编号，中心点坐标，面积，时间戳
                        cv2.putText(selected_belt, "{0}: ({1:.2f}, {2:.2f})".format(candidate.last.timestamp, candidate.last.center.x, candidate.last.center.y), (candidate.last.rect.x, candidate.last.rect.y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                        cv2.putText(selected_belt, "{0}x{1}, {2}".format(candidate.last.rect.w, candidate.last.rect.h, candidate.last.area), (int(candidate.last.center.x), int(candidate.last.center.y)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    # 绘制开始线
                    cv2.line(selected_belt, (0, self.start_y), (selected_belt.shape[1], self.start_y), (0, 0, 255), 2)
                    # 绘制结束线
                    cv2.line(selected_belt, (0, self.end_y), (selected_belt.shape[1], self.end_y), (0, 0, 255), 2)
                    cv2.namedWindow("Candidates Belt", cv2.WINDOW_NORMAL)
                    cv2.imshow("Candidates Belt", selected_belt)
                
                if self.is_save:
                    self.belt_video.write(belt)
                    self.track_video.write(selected_belt)
                    # # 存在目标时，保存目标图片
                    # if len(self.candidates) > 0 and not self.candidates[0].is_expired:
                    #     # shot 00001.jpg
                    #     cv2.imwrite(os.path.join(self.cache_dir, "shot", "{:05d}.jpg".format(shot_idx)), belt)
                    #     shot_idx += 1

                keyboard = cv2.waitKey(1)
                if keyboard == 'q' or keyboard == 27:
                    break

        self.__logger.info("Finished")
        if self.is_save:
            self.__logger.info("Save results to {}".format(self.cache_dir))
            self.belt_video.release()
            self.track_video.release()
        self.camera.release()
        cv2.destroyAllWindows()

    def __detect_move_task(self, image, tracker):
        try:
            # 异步检测
            detection_name, detection_class = self.detector.detect(image)
            # 检测到结果后执行演示启动对应坐标的喷嘴
            if detection_class is not None:
                self.__logger.info("Part {} detected".format(detection_name))
                v_pixel = tracker.get_velocity()
                self.__logger.debug("Velocity: {}".format(v_pixel))
                # 根据目标类型求出区域id
                area_id: int = -1
                for area, parts in self.parts_map.items():
                    if detection_class in parts:
                        area_id = int(area)
                        self.__logger.debug("Part {} in area {}".format(detection_name, area_id))
                        break
                if area_id <= -1:
                    self.__logger.info("Part {} not in areas".format(detection_name))
                    return

                # 计算喷嘴启动的延时时间
                t_0 = (tracker.first.timestamp - tracker.first.center.y / v_pixel)
                delta_t_0_area = self.time0 + area_id * self.spacing / self.speed
                delay = delta_t_0_area - (time.time() - t_0)
                # 喷嘴运行的时间
                duration = self.spacing / self.speed
                self.__logger.debug("Time: {}, Delay: {}(t_0: {}, t_0_area: {}), Duration: {}".format(time.time(), delay, t_0, delta_t_0_area, duration))
                # 延时
                time.sleep(delay - duration / 2)
                # 启动喷嘴
                self.nozzle_controller.open_valve(area_id)
                # 延时
                time.sleep(duration)
                # 关闭喷嘴
                self.nozzle_controller.close_valve(area_id)
        finally:
            # 删除目标
            self.candidates.remove(tracker)