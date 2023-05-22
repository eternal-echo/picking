import cv2
import torch
import detection_layer.YoloDetectAPI.yolo_detectAPI as yolo_detectAPI
class DetectionLayer:
    def __init__(self, weights='num4.pt'):
        """
        初始化检测层
        """
        # self.detection_url = "http://localhost:5000/v1/object-detection/yolov5"
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='num4.pt')
        
        self.api = yolo_detectAPI.DetectAPI(weights=weights, conf_thres=0.5, iou_thres=0.5) 

    def detect(self, img):
        """
        对传入的视频帧进行目标检测
        :param source: OpenCV的Mat
        :return: 目标检测结果，每个结果包括目标类别和目标框坐标等信息
        """
        result, names = self.api.detect([img])
        return result[0][1][0]