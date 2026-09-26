# Manuscript Layout Region Detection

This is a robust computer vision pipeline for detecting semantic layout regions in digitized historical manuscript images. It utilizes a hybrid approach combining deep learning text detection (CRAFT) and traditional morphological operations to accurately identify and categorize manuscript structures.

## Supported Region Classes
The system detects and categorizes layout elements into the following semantic classes:
- **header**: Text located at the top margin (e.g., page numbers, titles, digitizer watermarks).
- **footer**: Text located at the bottom margin.
- **main_text**: The primary manuscript text body.
- **side_text**: Marginalia or commentary written on the sides of the main text.
- **filler**: Non-primary text elements, including English digitizer stamps, holes, and decorative ink elements.

## Architecture & Model Rationale

The pipeline executes through several distinct phases. The combination of techniques was chosen to balance robustness against historical document noise (like textured palm leaves or degraded paper) and execution efficiency.

1. **Page Detection & Preprocessing**
   - **Rationale**: Manuscript scans often include large backgrounds, bindings, or color targets. Adaptive thresholding and morphological operations crop the main manuscript leaf and deskew it to ensure geometric heuristics downstream operate on a normalized coordinate space.
   - **Margin Recovery**: Header/footer watermarks outside the leaf are recovered using a projection profile technique on the original uncropped margins.

2. **Text Detection (CRAFT vs. Morphological Fallback)**
   - **Rationale**: The pipeline uses the **CRAFT (Character Region Awareness for Text Detection)** model as its primary text detector. CRAFT excels at detecting text with irregular shapes, varying sizes, and complex layouts common in historical manuscripts, predicting character centers and affinities. 
   - **Fallback**: Recognizing that CRAFT is computationally intensive and requires PyTorch, the system implements a robust **morphological fallback** (dilation + contour finding on a binarized image) when CRAFT is unavailable. This ensures the pipeline remains lightweight and easily deployable in resource-constrained environments.

3. **Hierarchical Merging (Grouping)**
   - **Rationale**: Text detectors (like CRAFT) typically output word or character-level bounding boxes. Manuscript layout analysis requires macro-regions (lines, paragraphs, blocks). A custom union-find based agglomerative clustering algorithm intelligently groups adjacent text boxes based on distance thresholds, forming coherent regional blocks.

4. **Zone Classification & Script Detection**
   - **Rationale**: Once blocks are formed, they must be classified. We use a rule-based system relying on geometric features (normalized coordinates, aspect ratio) and text density. 
   - **Tesseract OCR**: To distinguish English digitizer stamps ("filler") from indigenous manuscript text, the pipeline uses **Tesseract OCR**. By running Tesseract in OSD (Orientation and Script Detection) mode on header/footer regions, we can detect Latin script. If Latin script is detected, the region is classified as `filler`.

5. **Filler/Non-Text Detection**
   - **Rationale**: Manuscripts often contain punch-holes (for binding strings) or decorative non-text elements. We use geometric heuristics (circularity, solidity) on binary connected components to isolate these elements as `filler`.

6. **Region Refinement**
   - **Rationale**: A final pass resolves conflicts by removing subsumed overlapping regions, discarding margin noise, and enforcing strict image boundary clamping to prevent out-of-bounds coordinates.

## Installation & Setup

### 1. Python Dependencies
Clone the repository and install all required Python packages in one go. This includes the core libraries as well as the deep-learning models (CRAFT) for the highest text detection accuracy:
```bash
pip install -r requirements.txt
```
*(This installs `opencv-python-headless`, `numpy`, `Pillow`, `pytesseract`, and `craft-text-detector`. If CRAFT fails to load for any reason, the pipeline will automatically fall back to a lightweight OpenCV morphological detector).*

### 2. Tesseract OCR (System Requirement)
Tesseract must be installed on your system to enable script detection (differentiating English filler text from the main manuscript text).

- **Ubuntu/Debian:**
  ```bash
  sudo apt-get install tesseract-ocr
  ```
- **macOS (Homebrew):**
  ```bash
  brew install tesseract
  ```
- **Windows:**
  Download the installer from the [UB-Mannheim Tesseract repository](https://github.com/UB-Mannheim/tesseract/wiki) and ensure the executable is added to your system's `PATH`.

## Execution Instructions

### Single Image or Batch Inference
Run the CLI pipeline using `inference.py`. It supports both single-file processing and batch directory processing.

```bash
python inference.py --input ./data/test_images --output ./results
```

**Arguments:**
- `--input`: Path to a single image file (e.g., `.jpg`, `.png`, `.tif`) or a directory containing images.
- `--output`: Directory where the results will be saved. (Will be created if it doesn't exist).

### Output Format
For each processed image, two outputs are generated in the `--output` directory:
- `json/<image_name>.json`: A JSON file containing the original image size and a list of detected regions (label, absolute bounding box coordinates `[x, y, w, h]`, confidence score, and optional subtype).
- `annotated/<image_name>_annotated.jpg`: A visualization of the original image with color-coded bounding boxes and labels drawn over the detected regions.

### Running the Test Suite
To ensure the pipeline is functioning correctly, you can run the QA and basic unit test suites:

```bash
python -m unittest discover -s tests -v
```

## Limitations & Future Work
- **Resolution Dependent**: Morphological fallbacks and gap thresholds rely on relative dimensions. Extreme resolution variations might require tuning thresholds in `config.py`.
- **Noise Sensitivity**: Heavily textured manuscript leaves (e.g., palm leaves with strong horizontal grains) may occasionally introduce small noise artifacts, which the refinement stage attempts to filter.
