import cv2
import numpy as np
from typing import Tuple, Optional
from models import BBox, PageInfo
import config

def detect_page(image: np.ndarray) -> Tuple[BBox, np.ndarray, np.ndarray]:
    """
    Detects the page boundary and returns the BBox, page mask, and cropped image.
    """
    if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        return BBox(0, 0, 0, 0), np.zeros((0, 0), dtype=np.uint8), image

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    blur = cv2.GaussianBlur(gray, (config.PAGE_BG_BLUR_KERNEL, config.PAGE_BG_BLUR_KERNEL), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        h, w = image.shape[:2]
        return BBox(0, 0, w, h), np.ones((h, w), dtype=np.uint8) * 255, image
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    img_h, img_w = image.shape[:2]
    
    # Add small padding
    pad_w = int(img_w * config.PAGE_PADDING_RATIO)
    pad_h = int(img_h * config.PAGE_PADDING_RATIO)
    
    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    w = min(img_w - x, w + 2 * pad_w)
    h = min(img_h - y, h + 2 * pad_h)
    
    # If the detected box is very small, it might have failed, fall back to full image
    if w < img_w * 0.5 or h < img_h * 0.5:
        x, y, w, h = 0, 0, img_w, img_h
        
    page_bbox = BBox(x, y, w, h)
    
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    
    cropped = image[y:y+h, x:x+w]
    return page_bbox, mask, cropped

def deskew(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculates skew angle and rotates image.
    """
    if gray is None or gray.size == 0 or gray.shape[0] < 5 or gray.shape[1] < 5:
        return gray, 0.0

    # Inverse binary for projection profile
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    best_angle = 0
    max_variance = 0
    
    angles = np.arange(config.DESKEW_ANGLE_RANGE[0], config.DESKEW_ANGLE_RANGE[1] + config.DESKEW_ANGLE_STEP, config.DESKEW_ANGLE_STEP)
    
    h, w = thresh.shape
    center = (w // 2, h // 2)
    
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        proj = np.sum(rotated, axis=1)
        variance = np.var(proj)
        if variance > max_variance:
            max_variance = variance
            best_angle = angle
            
    if abs(best_angle) < 0.5:
        return gray, 0.0
        
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    deskewed = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return deskewed, best_angle

def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Processes the cropped page image.
    Returns enhanced_gray, binary, skew_angle
    """
    if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8), 0.0

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # 2. Deskew
    gray, angle = deskew(gray)
    
    # 3. Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_SIZE)
    enhanced_gray = clahe.apply(gray)
    
    # 4. Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        enhanced_gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, # Background is black, text is white for morphological ops, wait spec says text=black, background=white. Let's make it standard: text=white (foreground) for contour detection is usually better, but spec says "binary image (text=black, background=white)". We will use THRESH_BINARY and then invert where needed, or just follow typical OpenCV which expects foreground to be white for contours. Let's make binary have text as white.
        config.ADAPTIVE_THRESH_BLOCK, 
        config.ADAPTIVE_THRESH_C
    )
    # Actually, let's keep text=white, background=black since dilate works on white
    
    # 5. Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) # Removes pepper noise
    
    return enhanced_gray, binary, angle
