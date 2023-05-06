"""Perform test request"""
import cv2
import requests

class Detect:
    def __init__(self):
        self.detection_url = "http://localhost:5000/v1/object-detection/yolov5"

    def detect(self, image):
        # 将OpenCV的Mat转换为JPEG格式
        _, img_encoded = cv2.imencode(".jpg", image)
        image_bytes = img_encoded.tobytes()

        # 发送POST请求并获取响应
        response = requests.post(self.detection_url, files={"image": image_bytes}).json()

        return response

        # # 解析响应并返回结果
        # results = []
        # for obj in response["predictions"]:
        #     class_name = obj["label"]
        #     confidence = obj["confidence"]
        #     bbox = obj["bbox"]
        #     x1, y1, x2, y2 = bbox
        #     results.append({
        #         "class": class_name,
        #         "confidence": confidence,
        #         "bbox": [x1, y1, x2, y2]
        #     })
        # return results