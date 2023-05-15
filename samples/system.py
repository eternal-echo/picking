from app import ApplicationLayer
import asyncio
import argparse
from unittest.mock import MagicMock

parser = argparse.ArgumentParser(description="Parts Sorting System")
# 手动框选
parser.add_argument('--config_dir', type=str, help='The config file path of the selected area.', default='config')
parser.add_argument('--input', type=str, help='Camera index or video file path.', default='1')
parser.add_argument('--output_dir', type=str, help='Results path.', default='run')
args = parser.parse_args()

system = ApplicationLayer(camera_name=args.input, config_dir=args.config_dir, cache_dir=args.output_dir)
# nozzle_client = MagicMock()
# # 定义close_valve打桩函数，打印函数名和输入参数，返回True
# nozzle_client.close_valve = MagicMock(side_effect=lambda x: print("close_valve", x) or True)
# # 定义open_valve打桩函数，打印函数名和输入参数，返回True
# nozzle_client.open_valve = MagicMock(side_effect=lambda x: print("open_valve", x) or True)
# # 定义connect打桩函数, 打印函数名和输入参数，返回True
# nozzle_client.connect = MagicMock(side_effect=lambda: print("connect") or True)
# # 定义disconnect打桩函数, 打印函数名和输入参数，返回True
# nozzle_client.disconnect = MagicMock(side_effect=lambda: print("disconnect") or True)
# # 替换系统中的喷嘴控制器对象
# system.nozzle_controller = nozzle_client
if system.start() >= 0:
    system.run()
else:
    print("Failed to start system")
