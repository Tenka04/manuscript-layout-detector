import cv2
import numpy as np
from models import BBox, Region
import config

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

class Classifier:
    def __init__(self, page_width: int, page_height: int):
        self.page_w = page_width
        self.page_h = page_height

    def classify_region(self, bbox: BBox, binary_image: np.ndarray, crop_image: np.ndarray = None) -> Region:
        """
        Classify a bounding box into one of the 5 categories.
        binary_image is the full page binary image (white text on black).
        crop_image is the original/grayscale crop of the region (for OCR).
        """
        center_x = bbox.x + bbox.w / 2
        center_y = bbox.y + bbox.h / 2
        
        norm_x = center_x / self.page_w
        norm_y = center_y / self.page_h
        norm_w = bbox.w / self.page_w
        norm_h = bbox.h / self.page_h
        
        # Text density
        region_binary = binary_image[max(0, bbox.y):min(self.page_h, bbox.y+bbox.h), 
                                     max(0, bbox.x):min(self.page_w, bbox.x+bbox.w)]
        if region_binary.size > 0:
            text_density = np.count_nonzero(region_binary) / region_binary.size
        else:
            text_density = 0
            
        # Is English (Script Detection)
        is_english = False
        if HAS_TESSERACT and crop_image is not None and crop_image.size > 0:
            # Only run if in header/footer zones
            if norm_y < config.HEADER_ZONE_RATIO or norm_y > config.FOOTER_ZONE_RATIO:
                try:
                    osd = pytesseract.image_to_osd(crop_image, config='--psm 0')
                    if "Script: Latin" in osd:
                        is_english = True
                except Exception:
                    pass
                    
        # Apply rules
        # 1. English filler
        if is_english:
            return Region(bbox, "filler", 0.85, "english")
            
        # 3. Low density filler
        if text_density < config.FILLER_TEXT_DENSITY_THRESHOLD and bbox.w * bbox.h > config.FALLBACK_MIN_AREA:
            return Region(bbox, "filler", 0.60, "low_density")
            
        # 4. Header
        if norm_y < config.HEADER_ZONE_RATIO and norm_w > config.MIN_REGION_WIDTH_RATIO:
            return Region(bbox, "header", 0.80)
            
        # 5. Footer
        if norm_y > config.FOOTER_ZONE_RATIO and norm_w > config.MIN_REGION_WIDTH_RATIO:
            return Region(bbox, "footer", 0.80)
            
        # 6. Side text
        if (norm_x < config.SIDE_ZONE_RATIO or norm_x > 1 - config.SIDE_ZONE_RATIO) and norm_h > 0.1 and norm_w < 0.25:
            return Region(bbox, "side_text", 0.80)
            
        # 7. Main text
        return Region(bbox, "main_text", 0.70)
