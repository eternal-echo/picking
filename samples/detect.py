import cv2
import time
from detection_layer.detection_layer import DetectionLayer

detector = DetectionLayer()
# 计时
img = [
    cv2.imread(r'D:\embeded\project\graduation\picking\Software\system\data\dataset\partn_classes\1\1_000000.jpg'), 
]
start = time.time()
res = [
    detector.detect(img[0]), 
]
end = time.time()

print(res[0][0])
print(end - start)