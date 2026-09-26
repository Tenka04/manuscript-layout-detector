import os
import cv2
import numpy as np
import argparse
import glob
from pathlib import Path
from typing import List

from preprocessing import detect_page, preprocess_image
from detector import TextDetector
from filler_detector import FillerDetector
from postprocess import Postprocessor
from classifier import Classifier
from visualization import Visualizer
from models import DetectionResult, Region, BBox
import config


def _detect_margin_text(image: np.ndarray, page_bbox: BBox, orig_w: int, orig_h: int) -> List[Region]:
    """Detect header/footer watermark text in the original-image margins that
    fall outside ``page_bbox``.

    Many manuscript scans contain printed metadata lines (e.g.
    "CC-0. In Public Domain …" or "CSU-Sringeri …") above or below the
    manuscript leaf.  When ``detect_page`` crops to the leaf these lines are
    lost.  This function recovers them by scanning the top and bottom
    margin strips of the **original** image.

    Returns Region objects whose coordinates are in *original-image* space.
    The caller must NOT apply page_bbox offsets to these regions.
    """
    regions: List[Region] = []

    # Minimum margin height to bother scanning (pixels)
    MIN_MARGIN_PX = 20
    # Minimum text-line width relative to image width
    MIN_LINE_WIDTH_RATIO = 0.15

    def _scan_strip(strip: np.ndarray, label: str, orig_y_offset: int):
        """Find text lines in a horizontal strip via projection profile."""
        if strip is None or strip.size == 0 or strip.shape[0] < 5:
            return

        if len(strip.shape) == 3:
            gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        else:
            gray = strip.copy()

        # Binarise — text is dark on light background for these watermarks
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Horizontal projection profile to find text rows
        proj = np.sum(bw, axis=1)
        threshold = proj.max() * 0.1 if proj.max() > 0 else 1

        # Find contiguous runs of rows with text
        in_text = False
        y_start = 0
        for row_idx in range(len(proj)):
            if not in_text and proj[row_idx] > threshold:
                in_text = True
                y_start = row_idx
            elif in_text and (proj[row_idx] <= threshold or row_idx == len(proj) - 1):
                in_text = False
                y_end = row_idx if proj[row_idx] <= threshold else row_idx + 1
                line_h = y_end - y_start
                if line_h < 3:
                    continue

                # Find x-extent of text in this row band
                row_band = bw[y_start:y_end, :]
                col_proj = np.sum(row_band, axis=0)
                text_cols = np.where(col_proj > 0)[0]
                if len(text_cols) == 0:
                    continue
                x_start = int(text_cols[0])
                x_end = int(text_cols[-1]) + 1
                line_w = x_end - x_start

                if line_w < MIN_LINE_WIDTH_RATIO * orig_w:
                    continue

                # Coordinates in original image space
                abs_x = x_start
                abs_y = orig_y_offset + y_start
                abs_w = line_w
                abs_h = line_h

                regions.append(Region(
                    BBox(abs_x, abs_y, abs_w, abs_h),
                    label, 0.80
                ))

    top_margin = page_bbox.y
    bottom_margin = orig_h - (page_bbox.y + page_bbox.h)

    if top_margin >= MIN_MARGIN_PX:
        strip = image[0:page_bbox.y, :]
        _scan_strip(strip, "header", 0)

    if bottom_margin >= MIN_MARGIN_PX:
        strip = image[page_bbox.y + page_bbox.h:orig_h, :]
        _scan_strip(strip, "footer", page_bbox.y + page_bbox.h)

    return regions


def process_image(image_path: str, out_dir: str, text_detector: TextDetector, visualizer: Visualizer):
    image = cv2.imread(image_path)
    if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        print(f"Failed to read {image_path}")
        return

    orig_h, orig_w = image.shape[:2]

    # 1. Page boundary detection
    page_bbox, mask, cropped = detect_page(image)
    if cropped is None or cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
        cropped = image
        page_bbox = BBox(0, 0, orig_w, orig_h)

    page_w, page_h = cropped.shape[1], cropped.shape[0]

    # 1b. Detect header/footer watermark text in margins outside page crop.
    # These regions are returned in *original-image* coordinates and bypass
    # the page-body refinement pipeline.
    margin_regions = _detect_margin_text(image, page_bbox, orig_w, orig_h)

    # 2. Preprocessing
    enhanced_gray, binary, skew_angle = preprocess_image(cropped)

    # 3. Text detection
    text_boxes = text_detector.detect(cropped, binary)

    # 4. Merging
    postprocessor = Postprocessor(page_w, page_h)
    merged_bboxes = postprocessor.merge_text_boxes(text_boxes)

    # 5. Zone classification
    classifier = Classifier(page_w, page_h)
    regions = []
    for bbox in merged_bboxes:
        crop_image = enhanced_gray[max(0, bbox.y):min(page_h, bbox.y+bbox.h), max(0, bbox.x):min(page_w, bbox.x+bbox.w)]
        region = classifier.classify_region(bbox, binary, crop_image)
        regions.append(region)

    # 6. Non-text detection
    filler_detector = FillerDetector()
    non_text_regions = filler_detector.detect_non_text(binary, page_w, page_h)
    regions.extend(non_text_regions)

    # 7. Postprocessing / refinement (page-body regions only)
    final_regions = postprocessor.refine_regions(regions)

    # 8. Output generation
    filename = Path(image_path).name
    stem = Path(image_path).stem


    annotated = visualizer.draw_regions(image, final_regions, page_bbox)
    # Draw margin regions separately (no offset — already in original coords)
    if margin_regions:
        annotated = visualizer.draw_regions(annotated, margin_regions, BBox(0, 0, 0, 0))

    annotated_dir = os.path.join(out_dir, "annotated")
    json_dir = os.path.join(out_dir, "json")
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    cv2.imwrite(os.path.join(annotated_dir, f"{stem}_annotated.jpg"), annotated)

    # Save original coordinates in JSON by adjusting with page_bbox offset
    offset_x = page_bbox.x if page_bbox else 0
    offset_y = page_bbox.y if page_bbox else 0

    result_regions = []
    for r in final_regions:
        x = r.bbox.x + offset_x
        y = r.bbox.y + offset_y
        w = r.bbox.w
        h = r.bbox.h
        
        # Strictly clamp against original image dimensions [0, orig_w] and [0, orig_h]
        x1 = max(0, min(orig_w, x))
        y1 = max(0, min(orig_h, y))
        x2 = max(0, min(orig_w, x + w))
        y2 = max(0, min(orig_h, y + h))
        
        if x2 > x1 and y2 > y1:
            conf = max(0.0, min(1.0, float(r.confidence)))
            result_regions.append(Region(BBox(x1, y1, x2 - x1, y2 - y1), r.label, conf, r.sub_type))

    # Append margin header/footer regions (already in original-image coords)
    for r in margin_regions:
        x1 = max(0, min(orig_w, r.bbox.x))
        y1 = max(0, min(orig_h, r.bbox.y))
        x2 = max(0, min(orig_w, r.bbox.x + r.bbox.w))
        y2 = max(0, min(orig_h, r.bbox.y + r.bbox.h))
        if x2 > x1 and y2 > y1:
            conf = max(0.0, min(1.0, float(r.confidence)))
            result_regions.append(Region(BBox(x1, y1, x2 - x1, y2 - y1), r.label, conf, r.sub_type))
        
    result = DetectionResult(filename, (orig_w, orig_h), result_regions)
    visualizer.save_json(result, os.path.join(json_dir, f"{stem}.json"))

    print(f"Processed {filename} -> {len(result_regions)} regions")

def main():
    parser = argparse.ArgumentParser(description="Manuscript Layout Region Detection")
    parser.add_argument("--input", required=True, help="Input image file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    input_path = args.input
    out_dir = args.output

    if not os.path.exists(input_path):
        print(f"Input path does not exist: {input_path}")
        return

    if os.path.isdir(input_path):
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
        raw_files = []
        for ext in extensions:
            raw_files.extend(glob.glob(os.path.join(input_path, ext)))
            raw_files.extend(glob.glob(os.path.join(input_path, ext.upper())))
        seen = set()
        files = []
        for f in raw_files:
            norm_f = os.path.normpath(f)
            if norm_f not in seen:
                seen.add(norm_f)
                files.append(f)
    else:
        files = [input_path]

    if not files:
        print(f"No valid images found in {input_path}")
        return

    text_detector = TextDetector()
    visualizer = Visualizer()

    for f in files:
        process_image(f, out_dir, text_detector, visualizer)

if __name__ == "__main__":
    main()
