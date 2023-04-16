import pytest
import cv2
from detect.parts_segment import BackgroundModel

class TestBackgroundModel:
    def test_apply(self):
        bg_model = BackgroundModel()
        frame = cv2.imread(r'test\images\frame_0.jpg')
        fg_mask = bg_model.apply(frame)
        assert fg_mask is not None

    def test_get_background(self):
        bg_model = BackgroundModel()
        frame = cv2.imread('test\images\frame_0.jpg')
        bg_model.apply(frame)
        bg = bg_model.get_background()
        assert bg is not None