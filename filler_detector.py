import cv2
import numpy as np
from typing import List
from models import BBox, Region
import config
import math

class FillerDetector:
    def detect_non_text(self, binary: np.ndarray, page_width: int, page_height: int) -> List[Region]:
        """
        Detect decorative elements and holes.
        binary: white text/elements on black background.
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        non_text_regions = []
        page_area = page_width * page_height
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area == 0:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            is_hole = False
            is_decorative = False
            
            # Check for hole
            if circularity > config.CIRCULARITY_THRESHOLD:
                if config.HOLE_AREA_MIN_RATIO * page_area < area < config.HOLE_AREA_MAX_RATIO * page_area:
                    # Also check if it's near the center horizontally
                    center_x = x + w / 2
                    if abs(center_x - page_width / 2) < page_width * 0.2:
                        is_hole = True
                        
            # Check for decorative
            if not is_hole:
                if solidity < config.SOLIDITY_THRESHOLD:
                    is_decorative = True
                elif area > config.MIN_DECORATIVE_AREA_RATIO * page_area:
                    is_decorative = True
                elif abs(w - h) < max(w, h) * 0.1 and area > config.FALLBACK_MIN_AREA * 5: # Roughly square and large
                    # Could be an ornament
                    is_decorative = True
                    
            if is_hole:
                non_text_regions.append(Region(BBox(x, y, w, h), "filler", 0.9, "hole"))
            elif is_decorative:
                non_text_regions.append(Region(BBox(x, y, w, h), "filler", 0.75, "decorative"))
                
        return non_text_regions
