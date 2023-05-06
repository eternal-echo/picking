# 根据data/images中的字模图片和标签，制作字模的模板，用于后续的模板匹配

import cv2
import numpy as np
import os
import json
import yaml

class ExtractTemplate:
    def __init__(self, dataset_path='data\\dataset\\images', binary_path='data\\dataset\\binary', config_path='config', type_path='parts.yaml', temp_path='run\\templates'):
        self.dataset_path = dataset_path
        self.binary_path = binary_path
        self.config_path = config_path
        self.type_path = type_path
        self.temp_path = temp_path
        self.is_save = True

    # 获取配置信息，写入json中
    def get_config(self):
        templates = {}
        with open(self.type_path, 'r') as f:
            type_names = yaml.load(f, Loader=yaml.FullLoader)['names']
            # Create a reverse lookup dictionary
            type_index_dict = {v: int(k) for k, v in type_names.items()}
        for part_name in os.listdir(self.dataset_path):
            part_id = type_index_dict[part_name]
            part_path = os.path.join(self.dataset_path, part_name)
            labels_path = os.path.join(part_path, 'labels')
            label_names = os.listdir(labels_path)
            positive_label_name = label_names[0]
            negative_label_name = label_names[-1]
            positive_label_path = os.path.join(labels_path, positive_label_name)
            negative_label_path = os.path.join(labels_path, negative_label_name)
            positive_image_name = positive_label_name.split('.')[0] + '.jpg'
            negative_image_name = negative_label_name.split('.')[0] + '.jpg'
            positive_src_path = os.path.join(part_path, positive_image_name)
            negative_src_path = os.path.join(part_path, negative_image_name)
            template = {
                "name": part_name,
                "binary": {
                    "path": "",
                },
                "positive": {
                    "src_path": "",
                    "img_path": "",
                    "bbox": [],
                },
                "negative": {
                    "src_path": "",
                    "img_path": "",
                    "bbox": [],
                }
            }
            # 读取二值化的模板，作为正面的bbox
            bin_path = os.path.join(self.binary_path, part_name, '0.png')
            bin_img = cv2.imread(bin_path, cv2.IMREAD_GRAYSCALE)
             # 二值化
            _, bin_img = cv2.threshold(bin_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # 开操作
            kernel = np.ones((9, 9), np.uint8)
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
            # 寻找最大轮廓
            contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 找到最大轮廓
            max_contour = contours[0]
            max_area = cv2.contourArea(max_contour)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour
            # 绘制最大轮廓
            bin_img = np.zeros_like(bin_img)
            bin_img = cv2.drawContours(bin_img, [max_contour], -1, 255, -1)

            # 将最大轮廓旋转为正
            rect = cv2.minAreaRect(max_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # 获取旋转矩阵
            center = rect[0]
            size = rect[1]
            angle = rect[2]
            if size[0] < size[1]:
                angle = 90 + angle
            M = cv2.getRotationMatrix2D(center, angle, 1)

            bin_img = cv2.warpAffine(bin_img, M, bin_img.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
            # 连通域分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
            # 找到最大连通域
            max_label = 1
            max_size = stats[1, cv2.CC_STAT_AREA]
            for i in range(2, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > max_size:
                    max_label = i
                    max_size = stats[i, cv2.CC_STAT_AREA]
            # 获取最大连通域的bbox
            x, y, w, h, area = stats[max_label]
            # 截取最大连通域
            bin_img = bin_img[y:y+h, x:x+w]
            template["binary"]["path"] = os.path.join(self.config_path, 'templates', f'{part_id}_bin.jpg')
            # template["binary"]["img"] = bin_img
            cv2.imwrite(template["binary"]["path"], bin_img)

            template["positive"]["src_path"] = positive_src_path
            template["negative"]["src_path"] = negative_src_path
            template["positive"]["img_path"] = os.path.join(self.config_path, 'templates', f'{part_id}_positive.jpg')
            template["negative"]["img_path"] = os.path.join(self.config_path, 'templates', f'{part_id}_negative.jpg')
            positive_img = cv2.imread(positive_src_path)

            # template['positive']['img'] = positive_img

            # Load the label file for the positive object type to get its position
            with open(positive_label_path, 'r') as f:
                label_data = f.readlines()[0].strip().split(' ')
                label_type, x_center, y_center, w, h = int(float(label_data[0])), float(label_data[1]), float(label_data[2]), float(label_data[3]), float(label_data[4])
                x, y, w, h = int(x_center * positive_img.shape[1]), int(y_center * positive_img.shape[0]), int(w * positive_img.shape[1]), int(h * positive_img.shape[0])
                x = x - w // 2
                y = y - h // 2
                template["positive"]["bbox"] = [x, y, w, h]

            positive_img = positive_img[y:y+h, x:x+w]
            cv2.imwrite(template["positive"]["img_path"], positive_img)

            negative_img = cv2.imread(negative_src_path)
            # Load the label file for the negative object type to get its position
            with open(negative_label_path, 'r') as f:
                label_data = f.readlines()[0].strip().split(' ')
                label_type, x_center, y_center, w, h = int(float(label_data[0])), float(label_data[1]), float(label_data[2]), float(label_data[3]), float(label_data[4])
                x, y, w, h = int(x_center * negative_img.shape[1]), int(y_center * negative_img.shape[0]), int(w * negative_img.shape[1]), int(h * negative_img.shape[0])
                x = x - w // 2
                y = y - h // 2
                template["negative"]["bbox"] = [x, y, w, h]

            negative_img = negative_img[y:y+h, x:x+w]
            cv2.imwrite(template["negative"]["img_path"], negative_img)
            # template['negative']['img'] = negative_img

            templates[part_id] = template

        # sort the templates by key
        templates = dict(sorted(templates.items(), key=lambda x: x[0]))
        # Save the template information for both positive and negative images
        with open(os.path.join(self.config_path, 'templates.json'), 'w') as f:
            json.dump(templates, f, ensure_ascii=False, indent=4)

        return templates

    # 根据json配置文件，读取并截取有效图片，保存到templates文件夹中
    def get_templates(self):
        with open(os.path.join(self.config_path, 'templates.json'), 'r') as f:
            templates = json.load(f)
        # 创建图片模板文件夹
        templates_path = os.path.join(self.config_path, 'templates')
        if not os.path.exists(os.path.join(self.config_path, 'templates')):
            os.mkdir(templates_path)
        # 创建零件截取的图片
        for id, template in templates.items():
            positive_img = cv2.imread(template["positive"]["src_path"])
            negative_img = cv2.imread(template["negative"]["src_path"])
            positive_bbox = template["positive"]["bbox"]
            negative_bbox = template["negative"]["bbox"]
            positive_crop = positive_img[positive_bbox[1]:positive_bbox[1]+positive_bbox[3], positive_bbox[0]:positive_bbox[0]+positive_bbox[2]]
            negative_crop = negative_img[negative_bbox[1]:negative_bbox[1]+negative_bbox[3], negative_bbox[0]:negative_bbox[0]+negative_bbox[2]]
            positive_path = template['positive']['img_path']
            negative_path = template['negative']['img_path']
            cv2.imwrite(positive_path, positive_crop)
            cv2.imwrite(negative_path, negative_crop)
        with open(os.path.join(self.config_path, 'templates.json'), 'w') as f:
            json.dump(templates, f, ensure_ascii=False, indent=4)
        return templates

class MatchTemplate:
    def __init__(self, config_path='config', temp_path='run\\match'):
        self.config_path = config_path
        self.is_save = True
        self.temp_path = temp_path
        if not os.path.exists(self.temp_path):
            os.mkdir(self.temp_path)

        self.templates = self.load_templates()
    # 读取模板
    def load_templates(self):
        with open(os.path.join(self.config_path, 'templates.json'), 'r') as f:
            templates = json.load(f)
        for id, template in templates.items():
            template['binary']['img'] = cv2.imread(template['binary']['path'], 0)
            template['positive']['img'] = cv2.imread(template['positive']['img_path'])
            template['mask'] = {}
            template['mask']['img'] = template['binary']['img'].copy()
            template['mask']['contours'], hierarchy = cv2.findContours(template['mask']['img'], cv2.RETR_LIST,
                                                                        cv2.CHAIN_APPROX_SIMPLE)
            # 去除小轮廓
            min_area = 100
            template['mask']['contours'] = [c for c in template['mask']['contours'] if cv2.contourArea(c) > min_area]

            # 颜色直方图
            hist = cv2.calcHist([template['positive']['img']], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            # 对直方图进行归一化
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            template['mask']['hist'] = hist

        print('load templates success')
        return templates

    # 匹配模板
    def match(self, img, binary):
        best_match = None
        best_score = 0
        # 求颜色直方图
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        # 对直方图进行归一化
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        # # 灰度图
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # # 二值化
        # bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # 求轮廓
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        # macthed_list = []
        for id, template in self.templates.items():
            # 形状匹配
            shape_score = cv2.matchShapes(template['mask']['contours'][0], max_contour, cv2.CONTOURS_MATCH_I1, 0)
            # 将score归一化到0-1之间，并且变为越大越好
            shape_score = 1 - shape_score * 5

            # 颜色直方图匹配
            hist_score = cv2.compareHist(template['mask']['hist'], hist, cv2.HISTCMP_CORREL)
            score = shape_score * 0.2 + hist_score * 0.8
            if score > best_score:
                best_score = score
                best_match = id
        name = self.templates[best_match]['name']
        # 缓存匹配结果
        if self.is_save:
            # 绘制模板的轮廓
            # for id, score in macthed_list:
            #     img = cv2.drawContours(img, [self.templates[id]['mask']['contours'][0]], -1, (0, 0, 255), 2)
            img = cv2.drawContours(img, [self.templates[best_match]['mask']['contours'][0]], -1, 122, 2)
            # 绘制匹配的轮廓
            img = cv2.drawContours(img, [max_contour], -1, 64, 2)
            # 保存匹配结果
            cv2.imwrite(os.path.join(self.temp_path, f'{best_match}_match.jpg'), img)
        return best_match, best_score, name
