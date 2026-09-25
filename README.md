# Manuscript Layout Region Detection

This is a pipeline for detecting regions in manuscript images, including:
- Header
- Footer
- Main Text
- Side Text
- Filler (English text, pencil text, decorative elements)

## Dependencies
Install the required packages using:
`pip install -r requirements.txt`

You also need to have Tesseract OCR installed on your system for English text detection (used for filler detection).

## Usage
`python inference.py --input ./data/test_images --output ./results`

This command processes all supported images in the input directory and creates two output folders inside the results directory:
- `annotated/`: Contains images with drawn bounding boxes.
- `json/`: Contains JSON files with detected regions and confidence scores.

## Architecture
The pipeline uses:
1. OpenCV for page detection and preprocessing (adaptive thresholding, morphology).
2. CRAFT (Pretrained model) for text bounding box detection.
3. Hierarchical merging to group word boxes into block regions.
4. Rule-based classification based on heuristics (position on page, density, script).
5. Fallback mechanisms everywhere for robust execution.
