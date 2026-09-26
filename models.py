from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

@dataclass 
class TextBox:
    bbox: BBox
    confidence: float  # 0.0-1.0

@dataclass
class Region:
    bbox: BBox
    label: str          # header|footer|main_text|side_text|filler
    confidence: float   # 0.0-1.0
    sub_type: str = ""  # For filler: "english"|"decorative"|"pencil"


@dataclass
class DetectionResult:
    image_path: str
    image_size: Tuple[int, int]  # (width, height)
    regions: List[Region] = field(default_factory=list)
