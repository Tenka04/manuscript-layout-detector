import cv2
import numpy as np
import json
import os
from typing import List, Dict
from models import Region, DetectionResult
import config

class Visualizer:
    def draw_regions(self, image: np.ndarray, regions: List[Region], original_bbox=None) -> np.ndarray:
        """
        Draw regions on the original image.
        If original_bbox is provided, offset the region coordinates by the page crop offset.
        """
        annotated = image.copy()
        
        offset_x = original_bbox.x if original_bbox else 0
        offset_y = original_bbox.y if original_bbox else 0
        
        for r in regions:
            color = config.BBOX_COLORS.get(r.label, (255, 255, 255))
            # Convert RGB to BGR for OpenCV
            color = (color[2], color[1], color[0])
            
            x = r.bbox.x + offset_x
            y = r.bbox.y + offset_y
            w = r.bbox.w
            h = r.bbox.h
            
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, config.BBOX_THICKNESS)
            
            label_text = f"{r.label} ({r.confidence:.0%})"
            if r.sub_type:
                label_text += f" [{r.sub_type}]"
                
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, config.LABEL_FONT_SCALE, 1
            )
            
            cv2.rectangle(annotated, (x, y - text_h - 5), (x + text_w, y), color, -1)
            cv2.putText(
                annotated, label_text, (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, config.LABEL_FONT_SCALE, (0, 0, 0), 1
            )
            
        return annotated

    def save_json(self, result: DetectionResult, out_path: str):
        data = {
            "image": result.image_path,
            "image_size": {"width": result.image_size[0], "height": result.image_size[1]},
            "regions": []
        }
        for r in result.regions:
            r_dict = {
                "label": r.label,
                "bbox": {"x": r.bbox.x, "y": r.bbox.y, "w": r.bbox.w, "h": r.bbox.h},
                "confidence": r.confidence
            }
            if r.sub_type:
                r_dict["sub_type"] = r.sub_type
            data["regions"].append(r_dict)
            
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
