import cv2
import requests
from dataclasses import dataclass
import numpy as np


@dataclass
class Point:
    x: float
    y: float

@dataclass
class Rectangle:
    x: float
    y: float
    w: float
    h: float

@dataclass
class TargetInfo:
    rect: Rectangle
    center: Point
    area: float
    timestamp: float

class TrackingLayer:
    def __init__(self):
        """
        初始化跟踪层
        """
        pass

    def start_tracking(self):
        """
        开始目标跟踪
        """
        pass

    def stop_tracking(self):
        """
        停止目标跟踪
        """
        pass

    def get_tracking_result(self):
        """
        获取跟踪结果
        :return: 跟踪结果
        """
        pass

# TODO: kalman跟踪
class TargetTrack:
    def __init__(self, first: TargetInfo):
        self.first = first
        self.last = first
        self.is_expired = False
        self.is_center = False
        # self.tracker = Tracker(first.center)

    def update(self, target: TargetInfo):
        self.last = target
    #     self.tracker.correct(target.center)

    # def get_position(self):
    #     return self.tracker.get_position()

    def get_velocity(self):
        # return self.tracker.get_velocity()
        return (self.first.center.y - self.last.center.y) / (self.last.timestamp - self.first.timestamp)

# 背景建模
class BackgroundModel:
    def __init__(self, algo='MOG2', history=500, varThreshold=16, detectShadows=True):
        if algo == 'MOG2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=varThreshold, detectShadows=detectShadows)

    def apply(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        return fg_mask
    
# 连通域分析
class ConnectedComponents:
    def __init__(self, min_area=100, max_area=10000):
        self.min_area = min_area
        self.max_area = max_area

    # 基于腐蚀的形态学重建
    def pre_process(self, img, kernel_size=(9, 9)):
        # 中值滤波
        img = cv2.medianBlur(img, 5)
        # 开操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        return opening

    def process(self, img):
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        # 获取连通域面积
        areas = stats[:, 4]
        # 对连通域按面积从大到小排序
        sorted_areas_indices = np.argsort(areas)[::-1]
        # 选择面积在 self.min_area 到 self.max_area 像素之间的连通域
        # selected_areas_indices = sorted_areas_indices[(areas[sorted_areas_indices] > self.min_area) & (areas[sorted_areas_indices] < self.max_area)]
        selected_areas_indices = []
        for i in range(len(sorted_areas_indices)):
            if (areas[sorted_areas_indices[i]] < self.max_area
                # 区域的边缘处不再选择
                and stats[sorted_areas_indices[i], 1] + stats[sorted_areas_indices[i], 3] < img.shape[0] - 10
                and stats[sorted_areas_indices[i], 1] > 10
                and stats[sorted_areas_indices[i], 0] > 5
                and stats[sorted_areas_indices[i], 0] + stats[sorted_areas_indices[i], 2] < img.shape[1] - 5):
                if areas[sorted_areas_indices[i]] < self.min_area:
                    break
                selected_areas_indices.append(sorted_areas_indices[i])
        
        num_labels = len(selected_areas_indices)
        stats = stats[selected_areas_indices]
        centroids = centroids[selected_areas_indices]

        assert num_labels == len(stats) == len(centroids)

        # 返回图像和连通域标签
        return num_labels, labels, stats, centroids