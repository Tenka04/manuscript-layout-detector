"""
Debug script: run the pipeline on 3.png and dump every intermediate
line bbox and every pairwise merge decision from block-level merging.
"""
import cv2
import numpy as np
import sys, os

# Ensure repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import detect_page, preprocess_image
from detector import TextDetector
from models import BBox, TextBox
import config

IMAGE_PATH = "data/test_images/3.png"

# --- Step 1-3: run the pipeline up to raw text boxes ---
image = cv2.imread(IMAGE_PATH)
assert image is not None, f"Cannot read {IMAGE_PATH}"
orig_h, orig_w = image.shape[:2]
print(f"Image size: {orig_w}x{orig_h}")

page_bbox, mask, cropped = detect_page(image)
if cropped is None or cropped.size == 0:
    cropped = image
    page_bbox = BBox(0, 0, orig_w, orig_h)
page_w, page_h = cropped.shape[1], cropped.shape[0]
print(f"Cropped page: {page_w}x{page_h}")

enhanced_gray, binary, skew_angle = preprocess_image(cropped)
print(f"Skew angle: {skew_angle:.2f}")

detector = TextDetector()
text_boxes = detector.detect(cropped, binary)
print(f"\n=== Raw text boxes from detector: {len(text_boxes)} ===")
for i, tb in enumerate(text_boxes):
    b = tb.bbox
    print(f"  [{i:3d}] x={b.x:5d} y={b.y:5d} w={b.w:5d} h={b.h:4d}  conf={tb.confidence:.2f}")

# --- Step 4: Line-level merging (replicated from postprocess.py) ---
boxes = sorted(text_boxes, key=lambda tb: tb.bbox.y + tb.bbox.h / 2)
lines = []
if boxes:
    median_h = np.median([tb.bbox.h for tb in boxes])
    print(f"\nMedian box height: {median_h:.1f}")
    print(f"LINE_MERGE_Y_TOLERANCE = {config.LINE_MERGE_Y_TOLERANCE_FACTOR} * {median_h:.1f} = {config.LINE_MERGE_Y_TOLERANCE_FACTOR * median_h:.1f}")
    current_line = [boxes[0]]
    for tb in boxes[1:]:
        y_center_curr = tb.bbox.y + tb.bbox.h / 2
        y_center_prev = current_line[-1].bbox.y + current_line[-1].bbox.h / 2
        if abs(y_center_curr - y_center_prev) < config.LINE_MERGE_Y_TOLERANCE_FACTOR * median_h:
            current_line.append(tb)
        else:
            def _bbox_union(bboxes):
                min_x = min(b.x for b in bboxes)
                min_y = min(b.y for b in bboxes)
                max_x = max(b.x + b.w for b in bboxes)
                max_y = max(b.y + b.h for b in bboxes)
                return BBox(min_x, min_y, max_x - min_x, max_y - min_y)
            lines.append(_bbox_union([b.bbox for b in current_line]))
            current_line = [tb]
    if current_line:
        def _bbox_union(bboxes):
            min_x = min(b.x for b in bboxes)
            min_y = min(b.y for b in bboxes)
            max_x = max(b.x + b.w for b in bboxes)
            max_y = max(b.y + b.h for b in bboxes)
            return BBox(min_x, min_y, max_x - min_x, max_y - min_y)
        lines.append(_bbox_union([b.bbox for b in current_line]))

print(f"\n=== Merged lines: {len(lines)} ===")
for i, b in enumerate(lines):
    norm_cy = (b.y + b.h / 2) / page_h
    zone = "HEADER" if norm_cy < config.HEADER_ZONE_RATIO else ("FOOTER" if norm_cy > config.FOOTER_ZONE_RATIO else "BODY")
    print(f"  line[{i:2d}] x={b.x:5d}..{b.x+b.w:5d}  y={b.y:5d}..{b.y+b.h:5d}  w={b.w:5d}  h={b.h:4d}  cy_norm={norm_cy:.3f}  zone={zone}")

# --- Step 5: Block-level merging with full logging ---
if lines:
    median_line_h = np.median([b.h for b in lines])
    gap_threshold = config.BLOCK_MERGE_GAP_FACTOR * median_line_h
    overlap_threshold = config.BLOCK_MERGE_X_OVERLAP_THRESHOLD
    header_boundary = config.HEADER_ZONE_RATIO * page_h
    footer_boundary = config.FOOTER_ZONE_RATIO * page_h

    print(f"\n=== Block merge parameters ===")
    print(f"  median_line_h = {median_line_h:.1f}")
    print(f"  gap_threshold = {config.BLOCK_MERGE_GAP_FACTOR} * {median_line_h:.1f} = {gap_threshold:.1f}")
    print(f"  overlap_threshold = {overlap_threshold}")
    print(f"  header_boundary = {header_boundary:.1f}  footer_boundary = {footer_boundary:.1f}")
    print(f"  page_h = {page_h}")

    def _zone(y_top, y_bot):
        cy = (y_top + y_bot) / 2
        if cy < header_boundary:
            return "HEADER"
        elif cy > footer_boundary:
            return "FOOTER"
        return "BODY"

    groups = [[i] for i in range(len(lines))]
    pass_num = 0

    while True:
        pass_num += 1
        print(f"\n--- Pass {pass_num} ({len(groups)} groups) ---")

        # Freeze extents
        group_extents = []
        for group in groups:
            members = [lines[i] for i in group]
            y_top = min(b.y for b in members)
            y_bot = max(b.y + b.h for b in members)
            x_starts = [b.x for b in members]
            x_ends   = [b.x + b.w for b in members]
            n = len(members)
            if n >= 4:
                repr_x0 = float(np.percentile(x_starts, 25))
                repr_x1 = float(np.percentile(x_ends, 75))
            else:
                repr_x0 = float(np.median(x_starts))
                repr_x1 = float(np.median(x_ends))
            repr_w = max(repr_x1 - repr_x0, 1)
            group_extents.append((y_top, y_bot, repr_x0, repr_x1, repr_w))

        for gi, (group, ext) in enumerate(zip(groups, group_extents)):
            y_top, y_bot, x0, x1, w = ext
            print(f"  group[{gi}] lines={group}  y=[{y_top:.0f}..{y_bot:.0f}]  repr_x=[{x0:.0f}..{x1:.0f}]  repr_w={w:.0f}  zone={_zone(y_top, y_bot)}")

        order = sorted(range(len(groups)), key=lambda k: group_extents[k][0])
        parent = list(range(len(groups)))

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        any_merged = False
        print(f"\n  Pairwise decisions:")
        for idx_i in range(len(order)):
            gi = order[idx_i]
            y_top_i, y_bot_i, x0_i, x1_i, w_i = group_extents[gi]

            for idx_j in range(idx_i + 1, len(order)):
                gj = order[idx_j]
                y_top_j, y_bot_j, x0_j, x1_j, w_j = group_extents[gj]

                gap = y_top_j - y_bot_i

                if gap >= gap_threshold:
                    print(f"    g{gi} vs g{gj}: gap={gap:.0f} >= {gap_threshold:.0f}  -> BREAK (too far)")
                    break

                if _find(gi) == _find(gj):
                    print(f"    g{gi} vs g{gj}: gap={gap:.0f}  -> SKIP (same component)")
                    continue

                zone_i = _zone(y_top_i, y_bot_i)
                zone_j = _zone(y_top_j, y_bot_j)
                if zone_i != zone_j:
                    print(f"    g{gi} vs g{gj}: gap={gap:.0f}  zone={zone_i}!={zone_j}  -> BLOCKED (cross-zone)")
                    continue

                ov_left = max(x0_i, x0_j)
                ov_right = min(x1_i, x1_j)
                ov_w = max(0, ov_right - ov_left)
                ov_ratio = ov_w / min(w_i, w_j) if min(w_i, w_j) > 0 else 0

                if ov_ratio > overlap_threshold:
                    verdict = "MERGE"
                    ra, rb = _find(gi), _find(gj)
                    parent[rb] = ra
                    any_merged = True
                else:
                    verdict = "FAIL (low overlap)"

                print(f"    g{gi} vs g{gj}: gap={gap:.0f}  ov_ratio={ov_ratio:.3f} (need>{overlap_threshold})  zone={zone_i}  -> {verdict}")

        if not any_merged:
            print(f"\n  No merges this pass -> CONVERGED")
            break

        new_groups = {}
        for gi in range(len(groups)):
            root = _find(gi)
            if root not in new_groups:
                new_groups[root] = []
            new_groups[root].extend(groups[gi])
        groups = list(new_groups.values())

    print(f"\n=== Final blocks: {len(groups)} ===")
    for gi, group in enumerate(groups):
        members = [lines[i] for i in group]
        min_x = min(b.x for b in members)
        min_y = min(b.y for b in members)
        max_x = max(b.x + b.w for b in members)
        max_y = max(b.y + b.h for b in members)
        print(f"  block[{gi}] lines={group}  bbox=({min_x},{min_y},{max_x-min_x},{max_y-min_y})")
