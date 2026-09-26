# Manuscript Layout Region Detection

This is a robust computer vision pipeline for detecting semantic layout regions in digitized historical manuscript images.

## Supported Region Classes
The system detects and categorizes layout elements into the following semantic classes:
- **header**: Text located at the top margin (e.g., page numbers, titles, digitizer watermarks).
- **footer**: Text located at the bottom margin.
- **main_text**: The primary manuscript text body.
- **side_text**: Marginalia or commentary written on the sides of the main text.
- **filler**: Non-primary text elements, including English digitizer stamps, holes, and decorative ink elements.

## Architecture & Pipeline
The detection pipeline is implemented as a sequence of stages:
1. **Page Detection & Preprocessing**: Adaptive thresholding and morphological operations to crop the main manuscript leaf and deskew it. Header/footer watermarks outside the leaf are detected via projection profiling.
2. **Text Detection**: Uses CRAFT (if available) or a morphological fallback (dilation + contour finding) to locate word/character bounding boxes.
3. **Hierarchical Merging (Grouping)**: A custom union-find based agglomerative clustering algorithm groups word boxes into lines and then into coherent regional blocks.
4. **Zone Classification**: Rule-based classification assigns a label to each region based on its coordinates, aspect ratio, text density, and Tesseract OCR language detection (for English filler).
5. **Filler/Non-Text Detection**: Geometric heuristics (circularity, solidity) identify decorative elements and punch-holes in the manuscript.
6. **Region Refinement**: Final conflict resolution to remove subsumed regions, discard margin noise, and enforce strict image boundary clamping.

## Installation
Install the required Python packages:
```bash
pip install -r requirements.txt
```
*Note: You also need to have Tesseract OCR installed on your system for English text detection (used for classifying filler text).*

## Usage
The pipeline supports both single-image and batch directory processing.

Run inference via the CLI:
```bash
python inference.py --input ./data/test_images --output ./results
```
- `--input`: Path to a single image file or a directory of images.
- `--output`: Directory where the results will be saved.

## Output Format
For each processed image, two outputs are generated in the output directory:
- `json/<image_name>.json`: A JSON file containing the image size and a list of detected regions (label, bounding box, confidence score, and optional subtype).
- `annotated/<image_name>_annotated.jpg`: A visualization of the original image with color-coded bounding boxes and labels drawn over the detected regions.

## Project Structure
- `inference.py`: CLI entry point and main pipeline orchestration.
- `preprocessing.py`: Page detection, deskewing, and image binarization.
- `detector.py`: Text word/character detection (CRAFT / Morphological fallback).
- `postprocess.py`: Bounding box merging and conflict resolution.
- `classifier.py`: Rule-based region classification.
- `filler_detector.py`: Non-text layout element detection.
- `models.py`: Data classes (Region, BBox, etc.).
- `config.py`: Global configuration parameters and thresholds.
- `visualization.py`: Utilities for drawing bounding boxes and exporting JSON.
- `tests/`: QA test suite validating the pipeline.

## Testing
Run the test suite using `unittest`:
```bash
python -m unittest discover -s tests -v
```

## Limitations
- **Resolution Dependent**: Morphological fallbacks and gap thresholds rely on relative dimensions, but extreme resolution differences might still affect merging accuracy.
- **Noise Sensitivity**: Heavily textured manuscript leaves (e.g., palm leaves with strong horizontal grains) may occasionally introduce small noise artifacts, which the refinement stage attempts to filter.
