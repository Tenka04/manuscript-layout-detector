import numpy as np
from typing import List
from models import BBox, Region, TextBox
import config

class Postprocessor:
    def __init__(self, page_width: int, page_height: int):
        self.page_w = page_width
        self.page_h = page_height

    def merge_text_boxes(self, text_boxes: List[TextBox]) -> List[BBox]:
        """
        Merge word-level bounding boxes into block-level bounding boxes.
        """
        if not text_boxes:
            return []
            
        # 1. Line-level merging
        # Sort by y-center
        boxes = sorted(text_boxes, key=lambda tb: tb.bbox.y + tb.bbox.h / 2)
        
        lines = []
        if boxes:
            median_h = np.median([tb.bbox.h for tb in boxes])
            current_line = [boxes[0]]
            
            for tb in boxes[1:]:
                y_center_curr = tb.bbox.y + tb.bbox.h / 2
                y_center_prev = current_line[-1].bbox.y + current_line[-1].bbox.h / 2
                
                if abs(y_center_curr - y_center_prev) < config.LINE_MERGE_Y_TOLERANCE_FACTOR * median_h:
                    current_line.append(tb)
                else:
                    lines.append(self._bbox_union([b.bbox for b in current_line]))
                    current_line = [tb]
            if current_line:
                lines.append(self._bbox_union([b.bbox for b in current_line]))
                
        # 2. Block-level merging
        # Sort lines by y_top
        lines = sorted(lines, key=lambda b: b.y)
        
        blocks = []
        if lines:
            median_line_h = np.median([b.h for b in lines])
            current_block = [lines[0]]
            
            for line in lines[1:]:
                prev_line = current_block[-1]
                gap = line.y - (prev_line.y + prev_line.h)
                
                # Check horizontal overlap
                min_x = max(line.x, prev_line.x)
                max_x = min(line.x + line.w, prev_line.x + prev_line.w)
                overlap_w = max(0, max_x - min_x)
                overlap_ratio = overlap_w / min(line.w, prev_line.w) if min(line.w, prev_line.w) > 0 else 0
                
                if gap < config.BLOCK_MERGE_GAP_FACTOR * median_line_h and overlap_ratio > config.BLOCK_MERGE_X_OVERLAP_THRESHOLD:
                    current_block.append(line)
                else:
                    blocks.append(self._bbox_union(current_block))
                    current_block = [line]
            if current_block:
                blocks.append(self._bbox_union(current_block))
                
        return blocks

    def refine_regions(self, regions: List[Region]) -> List[Region]:
        """
        Conflict resolution, clamping, and reclassification logic.
        """
        refined = []
        for r in regions:
            # Clamping
            right = min(self.page_w, r.bbox.x + r.bbox.w)
            bottom = min(self.page_h, r.bbox.y + r.bbox.h)
            x = max(0, r.bbox.x)
            y = max(0, r.bbox.y)
            w = max(0, right - x)
            h = max(0, bottom - y)
            
            # Reclassify based on size
            label = r.label
            if label == "header" and (h / self.page_h) > config.MAX_HEADER_HEIGHT_RATIO:
                label = "main_text"
            if label == "footer" and (h / self.page_h) > config.MAX_HEADER_HEIGHT_RATIO:
                label = "main_text"
                
            refined.append(Region(BBox(x, y, w, h), label, r.confidence, r.sub_type))
            
        # Conflict resolution: remove fully subsumed regions
        final_refined = []
        for i, r in enumerate(refined):
            subsumed = False
            for j, other in enumerate(refined):
                if i != j:
                    if r.bbox.x >= other.bbox.x and r.bbox.y >= other.bbox.y and \
                       r.bbox.x + r.bbox.w <= other.bbox.x + other.bbox.w and \
                       r.bbox.y + r.bbox.h <= other.bbox.y + other.bbox.h:
                        if other.bbox.w * other.bbox.h > r.bbox.w * r.bbox.h:
                            subsumed = True
                            break
            if not subsumed:
                final_refined.append(r)
                
        return final_refined

    def _bbox_union(self, bboxes: List[BBox]) -> BBox:
        min_x = min(b.x for b in bboxes)
        min_y = min(b.y for b in bboxes)
        max_x = max(b.x + b.w for b in bboxes)
        max_y = max(b.y + b.h for b in bboxes)
        return BBox(min_x, min_y, max_x - min_x, max_y - min_y)
