import cv2
from move.parts_moving import NozzleMoving, NozzleSetting, PartInfo, Point

class PartsSortingSystem:
    def __init__(self, camera: cv2.VideoCapture, nozzle_setting: NozzleSetting):
        self.camera = camera
        self.part_map = {}
        self.nozzle_moving = NozzleMoving(self)


class SystemMark:
    def __init__(self):
        self.conveyor_lines = []

    def mark_conveyor(self, frame: cv2.Mat):
        """在指定帧中通过两条直线标记出传送带的位置

        Args:
            frame (cv2.Mat): 传送带所在的帧

        Returns:
            frame (cv2.Mat): 标记好传送带位置的帧
            conveyor_lines (list): 两条直线的起点和终点坐标
        """
        # 显示原始图像
        cv2.imshow("Mark Conveyor", frame)

        # 让用户用鼠标画两条直线，并记录直线的起点和终点坐标
        def draw_line(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.conveyor_lines.append((x, y))
            elif event == cv2.EVENT_LBUTTONUP:
                self.conveyor_lines.append((x, y))
                cv2.line(frame, self.conveyor_lines[-2], self.conveyor_lines[-1], (0, 255, 0), thickness=2)
        
        cv2.setMouseCallback("Mark Conveyor", draw_line)

        # 等待用户按下“Enter”键
        while True:
            cv2.imshow("Mark Conveyor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 13: # Enter键的ASCII码为13
                break

        # 关闭窗口并返回标记好传送带位置的图像和直线坐标信息
        cv2.destroyAllWindows()
        return frame, self.conveyor_lines[-4:], # 返回最后记录的两条直线的起点和终点坐标

    def mark_part(self, frame: cv2.Mat):
        """在指定帧中通过ROI框选零件，获取零件的外接矩形

        Args:
            frame (cv2.Mat): 零件所在的帧

        Returns:
            part (cv2.Mat): ROI框选的零件图像
            rect (tuple): 零件的外接矩形坐标
        """
        # 显示原始图像
        cv2.imshow("Mark Part", frame)

        # ROI选择零件
        rect = cv2.selectROI("Mark Part", frame, fromCenter=False, showCrosshair=True)
        part = frame[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])] # 获取ROI区域的图像

        cv2.destroyAllWindows()

        return part, rect