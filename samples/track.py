import cv2
import os
from tracking_layer.tracking_layer import BackgroundModel, ConnectedComponents, Point, Rectangle, TargetInfo, TargetTrack
from app import TrackModule, PreProcessModule
from config.config import Configurator
import argparse

class FindModule:
    def __init__(self, config):
        self.size_max = config['size_max']
        self.size_min = config['size_min']

        # 背景建模
        self.back_model = BackgroundModel(algo='MOG2', history=500, varThreshold=50, detectShadows=False)
        # 连通域分析器
        self.connected_components_analyzer = ConnectedComponents(min_area=self.size_min[0] * self.size_min[1],
                                                                 max_area=self.size_max[0] * self.size_max[1])

    def find(self, belt):
        # 背景消融
        self.bg_mask = self.back_model.apply(belt)

        # 形态学处理去除传送带上的噪点
        pre_proc = self.connected_components_analyzer.pre_process(self.bg_mask)

        # 连通域分析
        obj_nums, obj_labels, obj_stats, obj_centroids \
            = self.connected_components_analyzer.process(pre_proc)
        bin = pre_proc.copy()
        return obj_nums, obj_labels, obj_stats, obj_centroids, bin

class ApplicationLayer:
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
        self.preprocessor = PreProcessModule(config)
        self.finder = FindModule(config)
        self.tracker = TrackModule(config)

        self.find_callback = None
        self.center_callback = None
        self.lost_callback = None

        self.save_idx = 0

    def save_callback(self, belt, tracker, bin, name):
        self.save_idx = self.save_idx + 1
        # 创建'track/bg'文件夹
        os.makedirs(os.path.join(self.cache_dir, 'track'), exist_ok=True)
        # 保存背景图像
        cv2.imwrite(os.path.join(self.cache_dir, 'track', 'src_{}_{}.jpg'.format(name, self.save_idx)), belt)
        # 保存二值图像
        cv2.imwrite(os.path.join(self.cache_dir, 'track', 'bin_{}_{}.jpg'.format(name, self.save_idx)), bin)


    def start(self) -> int:
        # 打开摄像头
        if not self.camera.isOpened():
            print("Failed to open camera")
            return -1
        find_save_lambda = lambda belt, tracker, bin: self.save_callback(belt, tracker, bin, 'first')
        center_save_lambda = lambda belt, tracker, bin: self.save_callback(belt, tracker, bin, 'center')
        lost_save_lambda = lambda belt, tracker, bin: self.tracker.candidates.remove(tracker)
        self.tracker.set_callback(find_save_lambda, center_save_lambda, lost_save_lambda)
        return 0

    def run(self):
        while True:
            ret, frame = self.camera.read()
            frame_id = self.camera.get(cv2.CAP_PROP_POS_FRAMES)
            # frame不为空 empty()返回false
            if frame is None or not ret:
                break
            # 预处理模块：裁剪检测区域；图像预处理
            belt = self.preprocessor.pre_process(frame)
            # 查找模块：背景建模；连通域分析
            obj_nums, obj_labels, obj_stats, obj_centroids, bin = self.finder.find(belt)
            # 跟踪模块：
            self.tracker.track(belt, obj_nums, obj_stats, obj_centroids, bin)

            selected_belt = belt.copy()
            # 显示候选目标队列
            for candidate in self.tracker.candidates:
                # 绘制矩形
                cv2.rectangle(selected_belt, (candidate.last.rect.x, candidate.last.rect.y), (
                candidate.last.rect.x + candidate.last.rect.w, candidate.last.rect.y + candidate.last.rect.h),
                              (0, 255, 0), 2)
            # 绘制开始线
            cv2.line(selected_belt, (0, self.tracker.start_y), (selected_belt.shape[1], self.tracker.start_y),
                     (0, 0, 255), 2)
            # 绘制结束线
            cv2.line(selected_belt, (0, self.tracker.end_y), (selected_belt.shape[1], self.tracker.end_y),
                     (0, 0, 255), 2)
            cv2.namedWindow("Candidates Belt", cv2.WINDOW_NORMAL)
            cv2.namedWindow("bg_pre", cv2.WINDOW_NORMAL)
            cv2.imshow("Candidates Belt", selected_belt)
            cv2.imshow("bg_pre", bin)

            keyboard = cv2.waitKey(1)
            if keyboard == 'q' or keyboard == 27:
                break

        self.camera.release()
        cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description="Tarcking Layer")
# 手动框选
parser.add_argument('--config_dir', type=str, help='The config file path of the selected area.', default='config')
parser.add_argument('--input', type=str, help='Camera index or video file path.', default=r'data\dataset\test_video\num.mp4')
parser.add_argument('--output_dir', type=str, help='Results path.', default='run')
args = parser.parse_args()

system = ApplicationLayer(camera_name=args.input, config_dir=args.config_dir, cache_dir=args.output_dir)

if system.start() >= 0:
    system.run()
else:
    print("Failed to start system")