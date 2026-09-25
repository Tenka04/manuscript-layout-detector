"""Trace pairwise decisions for 6.png merge pass by pass."""
import cv2, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import detect_page, preprocess_image
from detector import TextDetector
from models import BBox, TextBox
import config

image = cv2.imread("data/test_images/6.png")
orig_h, orig_w = image.shape[:2]
page_bbox, mask, cropped = detect_page(image)
if cropped is None or cropped.size == 0:
    cropped = image
    page_bbox = BBox(0, 0, orig_w, orig_h)
page_w, page_h = cropped.shape[1], cropped.shape[0]
print(f"page: {page_w}x{page_h}")

enhanced_gray, binary, skew_angle = preprocess_image(cropped)
detector = TextDetector()
text_boxes = detector.detect(cropped, binary)

# Step 0: Split oversized
header_y = int(config.HEADER_ZONE_RATIO * page_h)
footer_y = int(config.FOOTER_ZONE_RATIO * page_h)
split_boxes = []
for tb in text_boxes:
    b = tb.bbox
    if b.h > page_h * 0.5:
        cuts = sorted(set([b.y, header_y, footer_y, b.y + b.h]))
        for k in range(len(cuts) - 1):
            y_top = max(b.y, cuts[k])
            y_bot = min(b.y + b.h, cuts[k + 1])
            if y_bot > y_top:
                split_boxes.append(TextBox(BBox(b.x, y_top, b.w, y_bot - y_top), tb.confidence))
    else:
        split_boxes.append(tb)
text_boxes = split_boxes

# Step 1: Line merge
boxes = sorted(text_boxes, key=lambda tb: tb.bbox.y + tb.bbox.h / 2)
lines = []
if boxes:
    median_h = np.median([tb.bbox.h for tb in boxes])
    current_line = [boxes[0]]
    for tb in boxes[1:]:
        yc = tb.bbox.y + tb.bbox.h / 2
        yp = current_line[-1].bbox.y + current_line[-1].bbox.h / 2
        if abs(yc - yp) < config.LINE_MERGE_Y_TOLERANCE_FACTOR * median_h:
            current_line.append(tb)
        else:
            def _bu(bbs):
                return BBox(min(b.x for b in bbs), min(b.y for b in bbs),
                            max(b.x+b.w for b in bbs)-min(b.x for b in bbs),
                            max(b.y+b.h for b in bbs)-min(b.y for b in bbs))
            lines.append(_bu([b.bbox for b in current_line]))
            current_line = [tb]
    if current_line:
        def _bu(bbs):
            return BBox(min(b.x for b in bbs), min(b.y for b in bbs),
                        max(b.x+b.w for b in bbs)-min(b.x for b in bbs),
                        max(b.y+b.h for b in bbs)-min(b.y for b in bbs))
        lines.append(_bu([b.bbox for b in current_line]))

print(f"Merged lines: {len(lines)}")
header_boundary = config.HEADER_ZONE_RATIO * page_h
footer_boundary = config.FOOTER_ZONE_RATIO * page_h
min_width_for_zone = config.MIN_REGION_WIDTH_RATIO * page_w

def _zone(y_top, y_bot):
    cy = (y_top + y_bot) / 2
    if cy < header_boundary: return "H"
    if cy > footer_boundary: return "F"
    return "B"

for i, b in enumerate(lines):
    z = _zone(b.y, b.y + b.h)
    print(f"  line[{i:2d}] y={b.y:4d}..{b.y+b.h:4d} x={b.x:4d}..{b.x+b.w:4d} w={b.w:4d} h={b.h:3d} zone={z}")

# Step 2: Block merge with logging
median_line_h = np.median([b.h for b in lines])
gap_threshold = config.BLOCK_MERGE_GAP_FACTOR * median_line_h
overlap_threshold = config.BLOCK_MERGE_X_OVERLAP_THRESHOLD
print(f"\ngap_threshold={gap_threshold:.1f}  overlap_threshold={overlap_threshold}  min_width_for_zone={min_width_for_zone:.0f}")

groups = [[i] for i in range(len(lines))]
pass_num = 0
while True:
    pass_num += 1
    group_extents = []
    for group in groups:
        members = [lines[i] for i in group]
        y_top = min(b.y for b in members)
        y_bot = max(b.y + b.h for b in members)
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

    order = sorted(range(len(groups)), key=lambda k: group_extents[k][0])
    parent = list(range(len(groups)))
    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    any_merged = False
    print(f"\n--- Pass {pass_num} ({len(groups)} groups) ---")

    # Only print merges involving the header or body split portions
    for idx_i in range(len(order)):
        gi = order[idx_i]
        y_top_i, y_bot_i, x0_i, x1_i, w_i = group_extents[gi]
        for idx_j in range(idx_i + 1, len(order)):
            gj = order[idx_j]
            y_top_j, y_bot_j, x0_j, x1_j, w_j = group_extents[gj]
            gap = y_top_j - y_bot_i
            if gap >= gap_threshold:
                break
            if _find(gi) == _find(gj):
                continue

            zone_i = _zone(y_top_i, y_bot_i)
            zone_j = _zone(y_top_j, y_bot_j)
            cross_zone = zone_i != zone_j
            both_wide = w_i >= min_width_for_zone and w_j >= min_width_for_zone
            blocked = cross_zone and both_wide

            ov_left = max(x0_i, x0_j)
            ov_right = min(x1_i, x1_j)
            ov_w = max(0, ov_right - ov_left)
            ov_ratio = ov_w / min(w_i, w_j) if min(w_i, w_j) > 0 else 0
            would_merge = ov_ratio > overlap_threshold

            if blocked or would_merge or (cross_zone and not blocked):
                result = "BLOCKED" if blocked else ("MERGE" if would_merge else "FAIL")
                print(f"  g{gi}({groups[gi]}) vs g{gj}({groups[gj]}): gap={gap:.0f} zone={zone_i}/{zone_j} w_i={w_i:.0f} w_j={w_j:.0f} ov={ov_ratio:.3f} -> {result}")

            if blocked:
                continue
            if would_merge:
                ra, rb = _find(gi), _find(gj)
                parent[rb] = ra
                any_merged = True

    if not any_merged:
        print("CONVERGED")
        break
    new_groups = {}
    for gi in range(len(groups)):
        root = _find(gi)
        if root not in new_groups:
            new_groups[root] = []
        new_groups[root].extend(groups[gi])
    groups = list(new_groups.values())

print(f"\nFinal blocks: {len(groups)}")
for gi, group in enumerate(groups):
    members = [lines[i] for i in group]
    min_x = min(b.x for b in members)
    min_y = min(b.y for b in members)
    max_x = max(b.x + b.w for b in members)
    max_y = max(b.y + b.h for b in members)
    z = _zone(min_y, max_y)
    print(f"  block[{gi}] lines={group} bbox=({min_x},{min_y},{max_x-min_x},{max_y-min_y}) zone={z}")
