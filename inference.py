import os
import cv2
import argparse
import glob
from pathlib import Path

from preprocessing import detect_page, preprocess_image
from detector import TextDetector
from filler_detector import FillerDetector
from postprocess import Postprocessor
from classifier import Classifier
from visualization import Visualizer
from models import DetectionResult, Region, BBox

def process_image(image_path: str, out_dir: str, text_detector: TextDetector, visualizer: Visualizer):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read {image_path}")
        return

    orig_h, orig_w = image.shape[:2]

    # 1. Page boundary detection
    page_bbox, mask, cropped = detect_page(image)
    page_w, page_h = cropped.shape[1], cropped.shape[0]

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

    # 7. Postprocessing / refinement
    final_regions = postprocessor.refine_regions(regions)

    # 8. Output generation
    filename = Path(image_path).name
    stem = Path(image_path).stem

    annotated = visualizer.draw_regions(image, final_regions, page_bbox)

    annotated_dir = os.path.join(out_dir, "annotated")
    json_dir = os.path.join(out_dir, "json")
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    cv2.imwrite(os.path.join(annotated_dir, f"{stem}_annotated.jpg"), annotated)

    # Save original coordinates in JSON by adjusting with page_bbox offset
    result_regions = []
    for r in final_regions:
        x = r.bbox.x + page_bbox.x
        y = r.bbox.y + page_bbox.y
        result_regions.append(Region(BBox(x, y, r.bbox.w, r.bbox.h), r.label, r.confidence, r.sub_type))
        
    result = DetectionResult(filename, (orig_w, orig_h), result_regions)
    visualizer.save_json(result, os.path.join(json_dir, f"{stem}.json"))

    print(f"Processed {filename} -> {len(final_regions)} regions")

def main():
    parser = argparse.ArgumentParser(description="Manuscript Layout Region Detection")
    parser.add_argument("--input", required=True, help="Input image file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    input_path = args.input
    out_dir = args.output

    if os.path.isdir(input_path):
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(input_path, ext)))
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
