import pytest
import threading
from unittest.mock import MagicMock
from move.parts_moving import NozzleMoving, NozzleSetting, PartInfo, Point
from dataclasses import dataclass
from typing import Dict, List, Deque, Callable

class TestNozzleMoving:

    @pytest.fixture(scope="class")
    def mock_nozzle_moving(self):
        nozzle_controller = MagicMock()
        nozzle_settings: Dict[int, NozzleSetting] = {
            0: NozzleSetting(0, 5, 0.5),
            1: NozzleSetting(1, 6, 0.5),
        }
        time_range = 1
        nozzle_moving = NozzleMoving(nozzle_controller, nozzle_settings, time_range)
        return nozzle_moving

    @pytest.mark.unit
    def test_move_init(self, mock_nozzle_moving: NozzleMoving):
        # class NozzleMoving(PartsMoving):
        assert mock_nozzle_moving.part_map is not None
        for nozzle_setting in mock_nozzle_moving.nozzle_settings.values():
            assert mock_nozzle_moving.part_map[nozzle_setting.part_id] == nozzle_setting.nozzle_id

        # class MoveRequests:
        assert mock_nozzle_moving.move_requests.moving_instance == mock_nozzle_moving


    @pytest.mark.unit
    @pytest.mark.parametrize(
        "input_part, expected_part_infos, expected_result, request_idx",
        [
            # 首次检测到了零件5
            (
                PartInfo(5, Point(2, 3), 0), 
                (PartInfo(5, Point(2, 3), 0), PartInfo(5, Point(2, 3), 0)), 
                True, 
                0
            ),
            # 零件5移动后，需要更新end_info
            (
                PartInfo(5, Point(3, 3), 0.5), 
                (PartInfo(5, Point(2, 3), 0), PartInfo(5, Point(3, 3), 0.5)), 
                True, 
                0
            ),
            # 首次检测到零件6
            (
                PartInfo(6, Point(2, 3), 0.5), 
                (PartInfo(6, Point(2, 3), 0.5), PartInfo(6, Point(2, 3), 0.5)), 
                True, 
                1
            ),
            # 零件5移动后更新end_info
            (
                PartInfo(5, Point(4, 3), 1), 
                (PartInfo(5, Point(2, 3), 0), PartInfo(5, Point(4, 3), 1)), 
                True, 
                0
            ),
            # 零件5的同一时刻的多次请求，时间戳必须大于上一次请求的时间戳
            (
                PartInfo(5, Point(4, 3), 1), 
                (PartInfo(5, Point(2, 3), 0), PartInfo(5, Point(4, 3), 1)), 
                False, 
                0
            ),
            # 零件6移动后，需要更新end_info
            (
                PartInfo(6, Point(3, 3), 1.5), 
                (PartInfo(6, Point(2, 3), 0.5), PartInfo(6, Point(3, 3), 1.5)), 
                True, 
                1
            ),
            # 来了一个新的零件5，需要添加到请求队列后面
            (
                PartInfo(5, Point(1, 3), 1.6), 
                (PartInfo(5, Point(1, 3), 1.6), PartInfo(5, Point(1, 3), 1.6)), 
                True, 
                2
            ),
            # 新零件5移动
            (
                PartInfo(5, Point(4, 3), 1.8), 
                (PartInfo(5, Point(1, 3), 1.6), PartInfo(5, Point(3, 3), 1.8)), 
                True, 
                2
            ),
            
        ],
    )
    def test_move_requests(self, input_part, expected_part_infos, expected_result, request_idx, mock_nozzle_moving: NozzleMoving):
        assert mock_nozzle_moving.move(input_part) == expected_result
        assert mock_nozzle_moving.move_requests.requests[request_idx].start_info == expected_part_infos[0]
        assert mock_nozzle_moving.move_requests.requests[request_idx].end_info == expected_part_infos[1]