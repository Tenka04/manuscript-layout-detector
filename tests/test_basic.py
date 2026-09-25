import unittest
from models import BBox
from postprocess import Postprocessor
from classifier import Classifier
import config
import numpy as np

class TestBasicPipeline(unittest.TestCase):
    def test_postprocessor_clamping(self):
        postprocessor = Postprocessor(100, 100)
        regions = [
            # A region outside boundaries
            __import__('models').Region(BBox(-10, -10, 50, 50), "main_text", 0.9)
        ]
        refined = postprocessor.refine_regions(regions)
        self.assertEqual(refined[0].bbox.x, 0)
        self.assertEqual(refined[0].bbox.y, 0)
        self.assertEqual(refined[0].bbox.w, 40)
        self.assertEqual(refined[0].bbox.h, 40)

    def test_classifier_zone(self):
        classifier = Classifier(1000, 1000)
        # Header zone
        bbox = BBox(100, 50, 500, 50)
        binary = np.ones((1000, 1000), dtype=np.uint8) * 255 # full density
        region = classifier.classify_region(bbox, binary, None)
        self.assertEqual(region.label, "header")
        
        # Footer zone
        bbox_footer = BBox(100, 900, 500, 50)
        region_footer = classifier.classify_region(bbox_footer, binary, None)
        self.assertEqual(region_footer.label, "footer")

if __name__ == "__main__":
    unittest.main()
