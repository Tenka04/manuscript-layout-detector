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

        # 0. Split oversized detections at zone boundaries.
        #    The fallback dilate+contour detector can produce a single contour
        #    spanning the entire page.  Splitting it ensures that header/footer
        #    text get their own lines and can be classified independently.
        #    Only split when OTHER (non-oversized) detections exist in the
        #    header/footer zones, providing evidence of distinct content there.
        header_y = int(config.HEADER_ZONE_RATIO * self.page_h)
        footer_y = int(config.FOOTER_ZONE_RATIO * self.page_h)
        has_hf_evidence = False
        for tb in text_boxes:
            b = tb.bbox
            if b.h > self.page_h * 0.5:
                continue  # skip oversized boxes themselves
            cy = b.y + b.h / 2
            if (cy < header_y or cy > footer_y) and b.w > config.MIN_REGION_WIDTH_RATIO * self.page_w:
                has_hf_evidence = True
                break

        split_boxes = []
        for tb in text_boxes:
            b = tb.bbox
            if has_hf_evidence and b.h > self.page_h * 0.5:
                # Box is oversized and H/F content exists — split at zone boundaries
                cuts = sorted(set([b.y, header_y, footer_y, b.y + b.h]))
                for i in range(len(cuts) - 1):
                    y_top = max(b.y, cuts[i])
                    y_bot = min(b.y + b.h, cuts[i + 1])
                    if y_bot > y_top:
                        split_boxes.append(TextBox(
                            BBox(b.x, y_top, b.w, y_bot - y_top),
                            tb.confidence
                        ))
            else:
                split_boxes.append(tb)
        text_boxes = split_boxes

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
                
        # 2. Block-level merging: fixed-point agglomerative clustering
        #    Each pass freezes group extents, finds valid merges via union-find,
        #    applies them, and repeats until no pass produces new merges.
        #    Hard zone-barrier: never merge groups whose y-centers fall in
        #    different zones (header / body / footer).
        blocks = []
        if lines:
            median_line_h = np.median([b.h for b in lines])
            gap_threshold = config.BLOCK_MERGE_GAP_FACTOR * median_line_h
            overlap_threshold = config.BLOCK_MERGE_X_OVERLAP_THRESHOLD
            header_boundary = config.HEADER_ZONE_RATIO * self.page_h
            footer_boundary = config.FOOTER_ZONE_RATIO * self.page_h

            def _zone(y_top, y_bot):
                """Classify a group's zone by its y-center."""
                cy = (y_top + y_bot) / 2
                if cy < header_boundary:
                    return 0  # header
                elif cy > footer_boundary:
                    return 2  # footer
                return 1      # body

            # Each group is a list of line indices into `lines`
            groups = [[i] for i in range(len(lines))]

            while True:
                # --- Freeze group extents for this pass ---
                group_extents = []
                for group in groups:
                    members = [lines[i] for i in group]
                    y_top = min(b.y for b in members)
                    y_bot = max(b.y + b.h for b in members)
                    # Representative x-range: IQR-based for 4+ lines,
                    # median for smaller groups, to prevent one wide
                    # outlier from inflating the merge zone.
                    x_starts = [b.x for b in members]
                    x_ends = [b.x + b.w for b in members]
                    n = len(members)
                    if n >= 4:
                        repr_x0 = float(np.percentile(x_starts, 25))
                        repr_x1 = float(np.percentile(x_ends, 75))
                    else:
                        repr_x0 = float(np.median(x_starts))
                        repr_x1 = float(np.median(x_ends))
                    repr_w = max(repr_x1 - repr_x0, 1)
                    group_extents.append((y_top, y_bot, repr_x0, repr_x1, repr_w))

                # Sort group indices by y_top for early-break optimisation
                order = sorted(range(len(groups)), key=lambda k: group_extents[k][0])

                # Union-find scoped to this pass
                parent = list(range(len(groups)))

                def _find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                any_merged = False
                for idx_i in range(len(order)):
                    gi = order[idx_i]
                    y_top_i, y_bot_i, x0_i, x1_i, w_i = group_extents[gi]

                    for idx_j in range(idx_i + 1, len(order)):
                        gj = order[idx_j]
                        y_top_j, y_bot_j, x0_j, x1_j, w_j = group_extents[gj]

                        # Vertical gap (groups sorted by y_top, so gap
                        # increases monotonically → safe to break early)
                        gap = y_top_j - y_bot_i
                        if gap >= gap_threshold:
                            break

                        # Already in the same component this pass
                        if _find(gi) == _find(gj):
                            continue

                        # Zone barrier: block cross-zone merges when at least
                        # one group is wide enough to be a real region.  Only
                        # allow cross-zone merges between two narrow fragments
                        # (noise that would otherwise be stranded).
                        min_width_for_zone = config.MIN_REGION_WIDTH_RATIO * self.page_w
                        if (_zone(y_top_i, y_bot_i) != _zone(y_top_j, y_bot_j)
                                and (w_i >= min_width_for_zone
                                     or w_j >= min_width_for_zone)):
                            continue

                        # Horizontal overlap on representative x-ranges
                        ov_left = max(x0_i, x0_j)
                        ov_right = min(x1_i, x1_j)
                        ov_w = max(0, ov_right - ov_left)
                        ov_ratio = ov_w / min(w_i, w_j) if min(w_i, w_j) > 0 else 0

                        if ov_ratio > overlap_threshold:
                            ra, rb = _find(gi), _find(gj)
                            parent[rb] = ra
                            any_merged = True

                if not any_merged:
                    break

                # Rebuild groups from union-find results
                new_groups = {}
                for gi in range(len(groups)):
                    root = _find(gi)
                    if root not in new_groups:
                        new_groups[root] = []
                    new_groups[root].extend(groups[gi])
                groups = list(new_groups.values())

            # Final blocks from converged groups
            for group in groups:
                blocks.append(self._bbox_union([lines[i] for i in group]))
                
        return blocks

    def refine_regions(self, regions: List[Region]) -> List[Region]:
        """
        Conflict resolution, clamping, and reclassification logic.
        """
        refined = []
        for r in regions:
            # Clamping coordinates strictly within [0, page_w] and [0, page_h]
            x1 = max(0, min(self.page_w, r.bbox.x))
            y1 = max(0, min(self.page_h, r.bbox.y))
            x2 = max(0, min(self.page_w, r.bbox.x + r.bbox.w))
            y2 = max(0, min(self.page_h, r.bbox.y + r.bbox.h))
            
            # Discard regions that are degenerate or completely outside boundaries
            if x2 <= x1 or y2 <= y1:
                continue
                
            w = x2 - x1
            h = y2 - y1
            
            # Reclassify based on size
            label = r.label
            if self.page_h > 0:
                if label == "header" and (h / self.page_h) > config.MAX_HEADER_HEIGHT_RATIO:
                    label = "main_text"
                if label == "footer" and (h / self.page_h) > config.MAX_HEADER_HEIGHT_RATIO:
                    label = "main_text"
                
            conf = max(0.0, min(1.0, float(r.confidence)))
            refined.append(Region(BBox(x1, y1, w, h), label, conf, r.sub_type))
            
        # Conflict resolution: remove fully subsumed regions
        final_refined = []
        for i, r in enumerate(refined):
            subsumed = False
            for j, other in enumerate(refined):
                if i != j:
                    if r.bbox.x >= other.bbox.x and r.bbox.y >= other.bbox.y and \
                       r.bbox.x + r.bbox.w <= other.bbox.x + other.bbox.w and \
                       r.bbox.y + r.bbox.h <= other.bbox.y + other.bbox.h:
                        if (other.bbox.w * other.bbox.h > r.bbox.w * r.bbox.h) or \
                           (other.bbox.w * other.bbox.h == r.bbox.w * r.bbox.h and j < i):
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
