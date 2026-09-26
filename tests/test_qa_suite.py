import unittest
import os
import shutil
import tempfile
import json
import subprocess
import sys
import cv2
import numpy as np
from pathlib import Path

from models import BBox, Region, TextBox
from postprocess import Postprocessor
from classifier import Classifier
from detector import TextDetector
from visualization import Visualizer
from preprocessing import detect_page, preprocess_image
from inference import process_image

VALID_LABELS = {"header", "footer", "main_text", "side_text", "filler"}

def verify_json_regions(test_case: unittest.TestCase, json_path: str):
    test_case.assertTrue(os.path.exists(json_path), f"JSON file does not exist: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    test_case.assertIn("image", data)
    test_case.assertIn("image_size", data)
    img_w = data["image_size"]["width"]
    img_h = data["image_size"]["height"]
    test_case.assertGreater(img_w, 0)
    test_case.assertGreater(img_h, 0)

    for idx, region in enumerate(data.get("regions", [])):
        label = region.get("label")
        test_case.assertIn(label, VALID_LABELS, f"Region {idx} has invalid label: {label}")
        
        conf = region.get("confidence")
        test_case.assertIsInstance(conf, (int, float), f"Region {idx} confidence is not a number")
        test_case.assertTrue(0.0 <= conf <= 1.0, f"Region {idx} confidence {conf} out of [0, 1]")

        bbox = region.get("bbox", {})
        x1 = bbox.get("x1", bbox.get("x"))
        y1 = bbox.get("y1", bbox.get("y"))
        x2 = bbox.get("x2", bbox.get("x", 0) + bbox.get("w", 0))
        y2 = bbox.get("y2", bbox.get("y", 0) + bbox.get("h", 0))

        test_case.assertGreaterEqual(x1, 0, f"Region {idx} x1 ({x1}) < 0")
        test_case.assertGreaterEqual(y1, 0, f"Region {idx} y1 ({y1}) < 0")
        test_case.assertLessEqual(x2, img_w, f"Region {idx} x2 ({x2}) > image_width ({img_w})")
        test_case.assertLessEqual(y2, img_h, f"Region {idx} y2 ({y2}) > image_height ({img_h})")
        test_case.assertLess(x1, x2, f"Region {idx} x1 ({x1}) >= x2 ({x2})")
        test_case.assertLess(y1, y2, f"Region {idx} y1 ({y1}) >= y2 ({y2})")

class TestQAScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="manuscript_qa_")
        cls.detector = TextDetector()
        cls.visualizer = Visualizer()
        cls.python_exe = sys.executable

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_single_image_input(self):
        # Scenario 1: Single image input
        sample_img = "data/test_images/1.png"
        out_dir = os.path.join(self.test_dir, "out_single")
        os.makedirs(out_dir, exist_ok=True)
        
        process_image(sample_img, out_dir, self.detector, self.visualizer)
        json_path = os.path.join(out_dir, "json", "1.json")
        verify_json_regions(self, json_path)

    def test_02_folder_input(self):
        # Scenario 2: Folder input
        in_dir = "data/test_images"
        out_dir = os.path.join(self.test_dir, "out_folder")
        
        cmd = [self.python_exe, "inference.py", "--input", in_dir, "--output", out_dir]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"CLI folder run failed: {res.stderr}")

        for stem in ["1", "2"]:
            json_path = os.path.join(out_dir, "json", f"{stem}.json")
            verify_json_regions(self, json_path)

    def test_03_jpg_input(self):
        # Scenario 3: JPG
        jpg_path = os.path.join(self.test_dir, "test_manuscript.jpg")
        src = cv2.imread("data/test_images/1.png")
        cv2.imwrite(jpg_path, src)

        out_dir = os.path.join(self.test_dir, "out_jpg")
        cmd = [self.python_exe, "inference.py", "--input", jpg_path, "--output", out_dir]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"CLI JPG run failed: {res.stderr}")

        json_path = os.path.join(out_dir, "json", "test_manuscript.json")
        verify_json_regions(self, json_path)

    def test_04_png_input(self):
        # Scenario 4: PNG
        png_path = "data/test_images/2.png"
        out_dir = os.path.join(self.test_dir, "out_png")
        cmd = [self.python_exe, "inference.py", "--input", png_path, "--output", out_dir]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"CLI PNG run failed: {res.stderr}")

        json_path = os.path.join(out_dir, "json", "2.json")
        verify_json_regions(self, json_path)

    def test_05_invalid_image(self):
        # Scenario 5: Invalid image (empty, corrupted, non-image)
        empty_img = os.path.join(self.test_dir, "empty.png")
        Path(empty_img).touch()

        corrupt_img = os.path.join(self.test_dir, "corrupt.jpg")
        with open(corrupt_img, "wb") as f:
            f.write(b"not a valid image content at all")

        out_dir = os.path.join(self.test_dir, "out_invalid")
        # Should gracefully handle without raising unhandled exception
        process_image(empty_img, out_dir, self.detector, self.visualizer)
        process_image(corrupt_img, out_dir, self.detector, self.visualizer)

        cmd = [self.python_exe, "inference.py", "--input", corrupt_img, "--output", out_dir]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)

    def test_06_empty_directory(self):
        # Scenario 6: Empty directory
        empty_dir = os.path.join(self.test_dir, "empty_dir")
        os.makedirs(empty_dir, exist_ok=True)

        out_dir = os.path.join(self.test_dir, "out_empty_dir")
        cmd = [self.python_exe, "inference.py", "--input", empty_dir, "--output", out_dir]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("No valid images found", res.stdout)

    def test_07_ocr_returning_no_detections(self):
        # Scenario 7: OCR returning no detections
        blank_path = os.path.join(self.test_dir, "blank.png")
        blank_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.imwrite(blank_path, blank_img)

        out_dir = os.path.join(self.test_dir, "out_no_ocr")
        process_image(blank_path, out_dir, self.detector, self.visualizer)

        json_path = os.path.join(out_dir, "json", "blank.json")
        verify_json_regions(self, json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.assertEqual(len(data["regions"]), 0)

    def test_08_bounding_boxes_outside_boundaries(self):
        # Scenario 8: Bounding boxes outside boundaries
        pp = Postprocessor(100, 100)
        test_boxes = [
            Region(BBox(-50, -50, 40, 40), "main_text", 0.9),    # completely outside top-left
            Region(BBox(-10, -10, 50, 50), "main_text", 0.9),    # partially outside top-left
            Region(BBox(80, 80, 50, 50), "main_text", 0.8),      # partially outside bottom-right
            Region(BBox(150, 150, 50, 50), "main_text", 0.7),    # completely outside bottom-right
            Region(BBox(-20, 20, 150, 30), "main_text", 0.8),    # wider than page, negative x
        ]
        refined = pp.refine_regions(test_boxes)
        for r in refined:
            x1 = r.bbox.x
            y1 = r.bbox.y
            x2 = r.bbox.x + r.bbox.w
            y2 = r.bbox.y + r.bbox.h
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 100)
            self.assertLessEqual(y2, 100)
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)

    def test_09_very_small_image(self):
        # Scenario 9: Very small image
        for size in [(1, 1), (5, 5), (10, 10), (25, 25)]:
            small_path = os.path.join(self.test_dir, f"small_{size[0]}x{size[1]}.png")
            small_img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 200
            cv2.imwrite(small_path, small_img)

            out_dir = os.path.join(self.test_dir, f"out_small_{size[0]}x{size[1]}")
            process_image(small_path, out_dir, self.detector, self.visualizer)

            json_path = os.path.join(out_dir, "json", f"small_{size[0]}x{size[1]}.json")
            verify_json_regions(self, json_path)

    def test_10_large_image(self):
        # Scenario 10: Large image
        large_path = os.path.join(self.test_dir, "large_4000x3000.png")
        large_img = np.ones((3000, 4000, 3), dtype=np.uint8) * 240
        # Add some text-like lines
        cv2.putText(large_img, "Header Line Text", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
        cv2.putText(large_img, "Main Paragraph Body Text Line 1", (500, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
        cv2.putText(large_img, "Main Paragraph Body Text Line 2", (500, 1250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
        cv2.putText(large_img, "Footer Annotation Text", (500, 2800), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
        cv2.imwrite(large_path, large_img)

        out_dir = os.path.join(self.test_dir, "out_large")
        cmd = [self.python_exe, "inference.py", "--input", large_path, "--output", out_dir]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"CLI Large image run failed: {res.stderr}")

        json_path = os.path.join(out_dir, "json", "large_4000x3000.json")
        verify_json_regions(self, json_path)

if __name__ == "__main__":
    unittest.main()
