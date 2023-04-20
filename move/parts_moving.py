from move.nozzle.nozzle_controller import NozzleController
import logging
from collections import deque
import time
import threading
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, List, Deque, Callable

_logger = logging.getLogger('moving')
_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
_logger.addHandler(console_handler)

# base
class PartsMoving:
    def __init__(self) -> None:
        pass

    def move(self, part_id, pos) -> bool:
        pass

# TODO: 和sys中的数据结构冲突，后续需要去除
@dataclass
class Point:
    x: float
    y: float

@dataclass
class PartInfo:
    part_id: int
    pos: Point
    timestamp: float

@dataclass
class NozzleSetting:
    nozzle_id: int
    part_id: int
    move_time: float


class NozzleMoving(PartsMoving):
    def __init__(
            self, 
            nozzle_controller: NozzleController, # 控制喷嘴电磁阀的控制器句柄
            nozzle_settings: Dict[int, NozzleSetting], # 键为喷嘴id，值为喷嘴id、设置的零件id、移动到对应喷嘴位置需要的时间。
            time_range: float, # 喷嘴吹送一次的工作时长
            part_y_tol: int = 10 # 检测到的零件y轴坐标与过去的坐标需小于容差
        ):
        self.nozzle_controller = nozzle_controller
        self.time_range = time_range
        # 检查nozzle_settings中喷嘴id的键值对是否匹配
        for nozzle_id in nozzle_settings.keys():
            assert nozzle_id == nozzle_settings[nozzle_id].nozzle_id
        self.nozzle_settings: Dict[int, NozzleSetting] = nozzle_settings
        # 键为零件id，值为喷嘴id
        self.part_map: Dict[int, int] = {}
        for nozzle_setting in nozzle_settings.values():
            self.part_map[nozzle_setting.part_id] = nozzle_setting.nozzle_id
        self.move_requests = self.MoveRequests(self)
        self.part_y_tolerance = part_y_tol

    def move(self, part_info: PartInfo) -> bool:
        """
        Move the part to the specified position by controlling the nozzle valves
        :param part_info: The information of the part to be moved
        :return: True if the part was moved successfully
        """
        # TODO: 此处添加滤波器，连续三次在非边界位置出现才更新零件信息
        return self.move_requests.update(part_info)

    class MoveRequests:
        def __init__(self, moving_instance: 'NozzleMoving'):
            # 按照零件x坐标降序排列
            self.requests: Deque[self.MoveRequest] = deque()
            self.moving_instance: NozzleMoving = moving_instance
        
        def update(self, detected_part: PartInfo) -> bool:
            """
            更新零件信息。从头部开始遍历，找到第一个小于等于零件x坐标的移动请求并替换，如果没有找到则添加到队列尾部
            :param detected_part: 检测到的零件信息
            :return: True 如果更新成功
            """
            # 检查零件类型是否支持分类
            if detected_part.part_id not in self.moving_instance.part_map:
                return False
            # 当requests为空时，直接添加到队列尾部
            if len(self.requests) == 0:
                move_request = self.MoveRequest(detected_part, detected_part, self)
                self.requests.append(move_request)
                return True
            pos_x = detected_part.pos.x
            # 从头部开始遍历，找到第一个小于等于零件x坐标的移动请求的idx
            for idx, request in enumerate(self.requests):
                # 找到了第一个小于等于零件x坐标的移动请求的idx，更新零件信息的end_info
                if pos_x >= request.end_info.pos.x:
                    # 当检测到时间戳小于end_info的时间戳
                    if (detected_part.timestamp <= request.end_info.timestamp or 
                        # 或detected_part的y轴坐标与end_info的y轴坐标间的误差超过容差时
                        abs(detected_part.pos.y - request.end_info.pos.y) > self.moving_instance.part_y_tolerance):
                        return False
                    # 当零件的id与检测到的一致时
                    elif detected_part.part_id == request.start_info.part_id == request.end_info.part_id:
                        request.end_info = detected_part
                        return True
                    # TODO: 当零件的id与检测到的不一致时：
                    # - 当前输入的零件匹配错误或之前这个零件检测错误
                    # - 这个零件时间戳太久没更新了，可能是后一个零件的
                    else:
                    #     # TODO: 如果连续3次在这个位置请求替换另一个零件，那么就更新deque中的零件end_info
                    #     # TODO: 更新零件信息后，还需要计算更新后的定时器时间
                        return False
            # 没有找到第一个小于等于零件x坐标的移动请求的idx，添加到队列尾部
            if idx == len(self.requests) - 1:
                move_request = self.MoveRequest(detected_part, detected_part, self)
                self.requests.append(move_request)
            return True
        
        def remove(self, request: 'MoveRequest') -> bool:
            """
            从队列中移除一个移动请求
            :param request: 要移除的移动请求
            :return: True if the request was removed successfully
            """
            if request in self.requests:
                self.requests.remove(request)
                return True
            return False
        
        
        class MoveRequest:
            def __init__(self, start_info: PartInfo, end_info: PartInfo, requets: 'NozzleMoving.MoveRequests'):
                """
                :param start_info: 同一个零件第一次被检测到的信息
                :param end_info: 同一个零件最后一次被检测到的信息
                :param requets: 移动请求队列的实例。用于在定时器回调函数中移除自己。
                """
                self.start_info: PartInfo = start_info
                self.end_info: PartInfo = end_info
                self.requets_instance: NozzleMoving.MoveRequests = requets
                self.nozzle_id = self.requets_instance.moving_instance.part_map[self.start_info.part_id]
                self.timer = threading.Timer(self.requets_instance.moving_instance.nozzle_settings[self.nozzle_id].move_time, self.move_task)

            def move_task(self):
                nozzle_id = self.requets_instance.moving_instance.part_map[self.start_info.part_id]
                self.requets_instance.moving_instance.nozzle_controller.open_valve(nozzle_id)
                time.sleep(self.requets_instance.moving_instance.time_range)
                self.requets_instance.moving_instance.nozzle_controller.close_valve(nozzle_id)
                # when the timer is waiting to be executed
                if self.timer.is_alive():
                    self.requets_instance.remove(self)
                # when the timer is already executed
                else:
                    # when the timer is executed successfully
                    if self.timer.finished.is_set():
                        pass
                    # when the timer is executing
                    else:
                        # TODO: cancel the nozzle_work
                        pass
                        # self.move_requests.remove(self)

            def start(self):
                self.timer.start()
