import tempfile
import unittest
import urllib.request
from pathlib import Path

import cv2

from vision.detector import OpenVocabularyDetector
import vision.local_object as local_object_module
from vision.local_object import LocalTeachEngine


BUS_IMAGE_URL = 'https://ultralytics.com/images/bus.jpg'
NEGATIVE_IMAGE_URL = 'https://ultralytics.com/images/zidane.jpg'


def download_image(url, path):
    urllib.request.urlretrieve(url, path)
    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        raise RuntimeError(f'failed to read image from {url}')
    return image


class ModelPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = OpenVocabularyDetector()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp_dir.name)
        cls.bus_image = download_image(BUS_IMAGE_URL, cls.base / 'bus.jpg')
        cls.negative_image = download_image(NEGATIVE_IMAGE_URL, cls.base / 'negative.jpg')

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        self.profile_dir = self.base / 'profiles'
        if self.profile_dir.exists():
            for item in self.profile_dir.rglob('*'):
                if item.is_file():
                    item.unlink()
            for item in sorted(self.profile_dir.rglob('*'), reverse=True):
                if item.is_dir():
                    item.rmdir()
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        local_object_module.PROFILES_DIR = self.profile_dir
        self.engine = LocalTeachEngine()

    def _register_demo_bus(self, expected_color=''):
        detections = self.detector.detect(self.bus_image, ['bus'])
        self.assertTrue(detections, 'YOLOWorld failed to detect a bus in the sample image')
        best = max(detections, key=lambda d: d['confidence'])
        ok, created = self.engine.add_profile(
            'demo_bus',
            'object',
            self.bus_image,
            best['rect'],
            {
                'label': 'Demo Bus',
                'category': 'bus',
                'brand': '',
                'expected_color': expected_color,
                'detect_as': 'bus',
                'aliases': 'coach, transit bus',
                'notes': 'automated test profile',
            },
        )
        self.assertTrue(ok)
        self.assertTrue(created)

    def test_model_matches_on_source_image(self):
        self._register_demo_bus()
        prompts, prompt_to_profiles = self.engine.build_detection_prompts()
        detections = self.detector.detect(self.bus_image, prompts)
        self.engine.update(self.bus_image, detections, prompt_to_profiles, bus=_SilentBus())
        profile = self.engine.get_profile('demo_bus')
        self.assertTrue(profile.visible)
        self.assertGreater(profile.match_score, 0.0)
        self.assertIn(profile.prompt_used, {'bus', 'coach', 'transit bus'})

    def test_model_rejects_negative_image(self):
        self._register_demo_bus()
        prompts, prompt_to_profiles = self.engine.build_detection_prompts()
        detections = self.detector.detect(self.negative_image, prompts)
        self.engine.update(self.negative_image, detections, prompt_to_profiles, bus=_SilentBus())
        profile = self.engine.get_profile('demo_bus')
        self.assertFalse(profile.visible)

    def test_expected_color_gate_can_reject(self):
        self._register_demo_bus(expected_color='purple')
        prompts, prompt_to_profiles = self.engine.build_detection_prompts()
        detections = self.detector.detect(self.bus_image, prompts)
        self.engine.update(self.bus_image, detections, prompt_to_profiles, bus=_SilentBus())
        profile = self.engine.get_profile('demo_bus')
        self.assertFalse(profile.visible)


class _SilentBus:
    def emit(self, *args, **kwargs):
        return None


if __name__ == '__main__':
    unittest.main()
