import cv2

# 背景建模
class BackgroundModel:
    def __init__(self, algo='MOG2', history=500, varThreshold=16, detectShadows=True):
        if algo == 'MOG2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=varThreshold, detectShadows=detectShadows)

    def apply(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        return fg_mask