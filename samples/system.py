from parts_sorting import PartsSortingSystem
import argparse

parser = argparse.ArgumentParser(description="Parts Sorting System")
# 手动框选
parser.add_argument('--select', type=str, help='The config file path of the selected area.', default='config/select.json')
parser.add_argument('--input', type=str, help='Camera index or video file path.', default='1')
parser.add_argument('--output', type=str, help='Results path.', default='run')
args = parser.parse_args()

system = PartsSortingSystem(camera=args.input, config_file=args.select, results_dir=args.output)
system.start()
system.run()
