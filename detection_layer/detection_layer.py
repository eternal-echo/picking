import cv2
import requests

class DetectionLayer:
    def __init__(self):
        """
        初始化检测层
        """
        self.detection_url = "http://localhost:5000/v1/object-detection/yolov5"

    def detect(self, image):
        """
        对传入的视频帧进行目标检测
        :param image: OpenCV的Mat
        :return: 目标检测结果，每个结果包括目标类别和目标框坐标等信息
        """
        # 将OpenCV的Mat转换为JPEG格式
        _, img_encoded = cv2.imencode(".jpg", image)
        image_bytes = img_encoded.tobytes()

        # 发送POST请求并获取响应
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(self.detection_url, data={"image": image_bytes}) as resp:
        #         response = await resp.json()
        response = requests.post(self.detection_url, files={"image": image_bytes}).json()

        return response[0]['name'], response[0]['class']