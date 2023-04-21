import cv2
import numpy as np
from dataclasses import dataclass


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

class TargetTrack:
    def __init__(self, first: TargetInfo):
        self.first = first
        self.last = first
        self.tracker = Tracker(first.center)

    def update(self, target: TargetInfo):
        self.last = target
        self.tracker.correct(target.center)

    def get_position(self):
        return self.tracker.get_position()

    def get_velocity(self):
        return self.tracker.get_velocity()

class Tracker:
    def __init__(self, center: Point):
        # 初始化 Kalman 滤波器
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], dtype=np.float32) * 0.0003
        
        # 初始化状态向量
        self.state = np.array([[center.x], [center.y], [0], [0]], dtype=np.float32)
        
        # 进行预测
        self.predict()
        
        # 记录初始位置
        self.start_pos = (center.x, center.y)
        
        # 记录跟踪的帧数
        self.frames_tracked = 0
        
    def predict(self):
        # 进行预测
        self.state = self.kalman.predict()
        
    def correct(self, measurement: Point):
        # 进行更新
        self.state = self.kalman.correct(np.array([[measurement.x], [measurement.y]], dtype=np.float32))
        
    def get_position(self):
        # 获取位置
        x = int(self.state[0][0])
        y = int(self.state[1][0])
        return (x, y)

    def get_velocity(self):
        # 获取速度
        vx = int(self.state[2][0])
        vy = int(self.state[3][0])
        return (vx, vy)

