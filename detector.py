import cv2
import numpy as np
from typing import List
from models import BBox, TextBox
import config

try:
    from craft_text_detector import Craft
    HAS_CRAFT = True
except ImportError:
    HAS_CRAFT = False

class TextDetector:
    def __init__(self):
        self.craft = None
        if HAS_CRAFT:
            try:
                self.craft = Craft(
                    output_dir=None, 
                    crop_type="box", 
                    cuda=False
                )
            except Exception as e:
                print(f"Failed to initialize CRAFT: {e}")
                self.craft = None

    def detect(self, image: np.ndarray, binary: np.ndarray) -> List[TextBox]:
        """
        Detect text boxes. If CRAFT is available, use it on grayscale/color image.
        Otherwise, use morphological fallback on binary image.
        """
        if self.craft is not None:
            return self._detect_craft(image)
        else:
            return self._detect_fallback(binary)

    def _detect_craft(self, image: np.ndarray) -> List[TextBox]:
        # CRAFT works better on RGB, if grayscale, convert to BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        prediction_result = self.craft.detect_text(
            image,
            text_threshold=config.CRAFT_TEXT_THRESHOLD,
            link_threshold=config.CRAFT_LINK_THRESHOLD,
            low_text=config.CRAFT_LOW_TEXT
        )
        
        boxes = prediction_result["boxes"]
        text_boxes = []
        
        for poly in boxes:
            # poly is 4x2 array
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            text_boxes.append(TextBox(BBox(x, y, w, h), confidence=1.0)) # Ideally average heatmap
            
        return text_boxes

    def _detect_fallback(self, binary: np.ndarray) -> List[TextBox]:
        """
        Fallback method using morphological operations.
        Binary should have text as white on black background.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.FALLBACK_DILATE_KERNEL)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < config.FALLBACK_MIN_AREA:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            # Basic aspect ratio filter: width > height (usually words are wider)
            if w > h:
                text_boxes.append(TextBox(BBox(x, y, w, h), confidence=0.5))
                
        return text_boxes
