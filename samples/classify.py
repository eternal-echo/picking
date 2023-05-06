import cv2
import os
import argparse
from detect.parts_classify import ExtractTemplate, MatchTemplate

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_path', type=str, default='data\\dataset\\images', help='path to dataset images')
# parser.add_argument('--config_path', type=str, default='config', help='path to cache directory')
# args = parser.parse_args()
#
# dataset_path = args.dataset_path
# config_path = args.config_path

# extractor = ExtractTemplate()
# templates = extractor.get_config()
# templates = extractor.get_templates()
# extractor.make_templates()
# template = extractor.get_binary()

img = cv2.imread(r"D:\embeded\project\graduation\picking\Software\system\run\templates\2_positive_th.jpg", 0)
# # 为bin添加黑色的边框，扩充图片的尺寸
# img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=0)
matcher = MatchTemplate()
# matcher.make_mask()
best_match, best_score, _ = matcher.match(img)
print(best_match, best_score)
cv2.imshow('img', img)
cv2.waitKey(0)
