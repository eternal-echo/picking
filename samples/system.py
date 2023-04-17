from parts_sorting import PartsSortingSystem
import argparse

parser = argparse.ArgumentParser(description="Parts Sorting System")
parser.add_argument('--input', type=str, help='Camera index or video file path.', default='1')
parser.add_argument('--output', type=str, help='Results path.', default='run')
args = parser.parse_args()

system = PartsSortingSystem(args.input, args.output)
system.start()
system.run()
