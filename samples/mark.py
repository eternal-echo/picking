from parts_sorting import SystemMark
import argparse
import cv2

parser = argparse.ArgumentParser(description="Mark Conveyor and Part")
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
args = parser.parse_args()

system_mark = SystemMark()

# 打开视频流
capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

# 等待用户空格开始标注
print("Press space to start marking")
while True:
    ret, frame = capture.read()
    if not ret:
        break
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        break

# 开始标注传送带
print("Marking conveyor")
ret, frame = capture.read()
conveyor_frame, conveyor_rect = system_mark.mark_conveyor(frame)
cv2.imshow("Conveyor Frame", conveyor_frame)
print("Conveyor:", conveyor_rect)

# 等待用户空格开始标注
print("Press space to start marking")
while True:
    ret, frame = capture.read()
    if not ret:
        break
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        break

# 开始标注最大零件
print("Marking the largest part")
ret, frame = capture.read()
max_part_frame, max_part_rect = system_mark.mark_part(frame)
cv2.imshow("Marked Frames", max_part_frame)
print("Largest part:", max_part_rect)

# 等待用户空格开始标注
print("Press space to start marking")
while True:
    ret, frame = capture.read()
    if not ret:
        break
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        break

# 开始标注最小零件
print("Marking the smallest part")
ret, frame = capture.read()
min_part_frame, min_part_rect = system_mark.mark_part(frame)
cv2.imshow("Marked Frames", min_part_frame)
print("Largest part:", min_part_rect)
