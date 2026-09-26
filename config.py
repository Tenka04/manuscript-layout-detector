# All magic numbers in one place. Every module imports from here.

# Page detection
PAGE_BG_BLUR_KERNEL = 15
PAGE_PADDING_RATIO = 0.02

# Preprocessing
DESKEW_ANGLE_RANGE = (-5, 5)  # degrees
DESKEW_ANGLE_STEP = 0.5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
ADAPTIVE_THRESH_BLOCK = 31
ADAPTIVE_THRESH_C = 10
MORPH_KERNEL_SIZE = 3

# Text detection (CRAFT)
CRAFT_TEXT_THRESHOLD = 0.7
CRAFT_LINK_THRESHOLD = 0.4
CRAFT_LOW_TEXT = 0.4

# Fallback text detection
FALLBACK_DILATE_KERNEL = (25, 1)
FALLBACK_MIN_AREA = 100

# Zone boundaries (fraction of page dimension)
HEADER_ZONE_RATIO = 0.15
FOOTER_ZONE_RATIO = 0.85
SIDE_ZONE_RATIO = 0.15
MIN_REGION_WIDTH_RATIO = 0.3
MAX_HEADER_HEIGHT_RATIO = 0.25

# Merging
LINE_MERGE_Y_TOLERANCE_FACTOR = 0.6
BLOCK_MERGE_GAP_FACTOR = 2.0
BLOCK_MERGE_X_OVERLAP_THRESHOLD = 0.5

# Non-text detection
CIRCULARITY_THRESHOLD = 0.7
SOLIDITY_THRESHOLD = 0.3
MIN_DECORATIVE_AREA_RATIO = 0.05
HOLE_AREA_MIN_RATIO = 0.005
HOLE_AREA_MAX_RATIO = 0.03

# Filler detection
FILLER_TEXT_DENSITY_THRESHOLD = 0.02

# Minimum region width (fraction of page width) to keep in refinement.
# Regions narrower than this are treated as margin noise and discarded.
MIN_REFINED_WIDTH_RATIO = 0.04

# Output
BBOX_COLORS = {
    "header": (255, 107, 107),
    "footer": (78, 205, 196),
    "main_text": (69, 183, 209),
    "side_text": (150, 206, 180),
    "filler": (255, 234, 167),
}
BBOX_THICKNESS = 3
LABEL_FONT_SCALE = 0.7
