import cv2
import os
import json
import logging


class Configurator:
    def __init__(self, camera, config_dir='config', cache_dir='run'):
        # camera对象
        self.camera = camera
        # 标注工具
        self.marking = SystemMark()
        # 运行缓存路径
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        # 配置路径
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        self.config_dir = config_dir
        # 初始化日志系统
        self.__logger = logging.getLogger('config')
        self.__logger.setLevel(logging.DEBUG)
        self.__console_handler = logging.StreamHandler()
        self.__console_handler.setLevel(logging.DEBUG)
        self.__logger.addHandler(self.__console_handler)

    def get_config(self):
        # 读取配置文件，获取传送带区域、最大最小零件尺寸信息
        if os.path.exists(os.path.join(self.config_dir, 'config.json')):
            with open(os.path.join(self.config_dir, 'config.json'), 'r') as f:
                config = json.load(f)
            if 'areas' not in config:
                self.__logger.error("Failed to get areas")
                return None
            else:
                # 移动相关的参数
                self.parts_map = config['areas']['parts_map']
                # y=0 到喷嘴0的距离 60 cm 中央到喷嘴0的距离 53.3cm
                self.nozzle_y0 = config['areas']['nozzle_y0']
                # 传送带速度 m/min -> cm/s
                self.speed = config['areas']['speed'] * 100 / 60
                # 喷嘴间距 cm
                self.spacing = config['areas']['spacing']

            # 未标注传送带区域
            if 'tracking' not in config or not config['tracking']:
                # 框选传送带检测范围
                self.__logger.info("Please select conveyor")
                self.__wait_for_marking()
                _, self.bbox_belt = self.marking.mark_belt(self.camera.read()[1])

                # 框选最大零件，用于连通域筛选，去除光线干扰
                self.__logger.info("Please select the biggest part")
                self.__wait_for_marking()
                _, bbox_max = self.marking.mark_part(self.camera.read()[1])
                self.size_max = (bbox_max[2], bbox_max[3])

                # 框选最小零件，用于连通域筛选，去除传送带噪声
                self.__logger.info("Please select the smallest part")
                self.__wait_for_marking()
                _, bbox_min = self.marking.mark_part(self.camera.read()[1])
                self.size_min = (bbox_min[2], bbox_min[3])

                # 销毁窗口
                cv2.destroyAllWindows()

                # 构建JSON对象
                tracking_config = {
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
                config['tracking'] = tracking_config
                # 保存传送带区域和最大最小零件尺寸的json配置文件
                with open(os.path.join(self.config_dir, 'config.json'), 'w') as f:
                    json.dump(config, f, indent=4)
            # 已标注过传送带区域
            else:
                # 跟踪的参数
                self.bbox_belt = (
                config['tracking']['belt']['x'], config['tracking']['belt']['y'], config['tracking']['belt']['width'],
                config['tracking']['belt']['height'])
                self.size_max = (config['tracking']['max_part']['width'], config['tracking']['max_part']['height'])
                self.size_min = (config['tracking']['min_part']['width'], config['tracking']['min_part']['height'])

        else:
            self.__logger.error("Config file not found, please check config/config.json")
            return None

        return {
            'bbox_belt': self.bbox_belt,
            'size_max': self.size_max,
            'size_min': self.size_min,
            'parts_map': self.parts_map,
            'nozzle_y0': self.nozzle_y0,
            'speed': self.speed,
            'spacing': self.spacing
        }

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

        # 销毁窗口
        cv2.destroyWindow("Mark Conveyor")

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
        part = frame[int(rect[1]):int(rect[1] + rect[3]), int(rect[0]):int(rect[0] + rect[2])]  # 获取ROI区域的图像

        # 销毁窗口
        cv2.destroyWindow("Mark Part")

        return part, rect