import pytest
import cv2
import os
import pickle
from detect.parts_segment import BackgroundModel
from detect.parts_classify import ExtractTemplate

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


class TestExtractTemplate:
    def test_extract(self):
        # Create an instance of the ExtractTemplate class
        dataset_path = 'path/to/dataset'
        cache_path = 'path/to/cache'
        extractor = ExtractTemplate(dataset_path, cache_path)

        # Call the extract method and get the templates
        templates = extractor.extract()

        # Check that the templates were cached to a file
        assert os.path.exists(cache_path)

        # Check that the templates were cached correctly
        with open(cache_path, 'rb') as f:
            cached_templates = pickle.load(f)
        assert templates == cached_templates