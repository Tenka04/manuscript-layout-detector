import glob
import json
import os

valid_labels = {'header', 'footer', 'main_text', 'side_text', 'filler'}
json_files = sorted(glob.glob('results/json/*.json'))

total_regions = 0
violations = []

print(f"Found {len(json_files)} JSON files in results/json:")
for f in json_files:
    with open(f, 'r') as fp:
        data = json.load(fp)
    
    img_name = data.get('image')
    img_w = data['image_size']['width']
    img_h = data['image_size']['height']
    regions = data.get('regions', [])
    print(f"  {os.path.basename(f)}: image_size=({img_w}x{img_h}), {len(regions)} regions")
    
    for idx, r in enumerate(regions):
        total_regions += 1
        label = r.get('label')
        conf = r.get('confidence')
        bbox = r.get('bbox', {})
        x1 = bbox.get('x1', bbox.get('x'))
        y1 = bbox.get('y1', bbox.get('y'))
        x2 = bbox.get('x2', bbox.get('x', 0) + bbox.get('w', 0))
        y2 = bbox.get('y2', bbox.get('y', 0) + bbox.get('h', 0))
        
        # Check label
        if label not in valid_labels:
            violations.append(f"[{os.path.basename(f)}] Region {idx}: invalid label '{label}'")
        # Check confidence
        if not (isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0):
            violations.append(f"[{os.path.basename(f)}] Region {idx}: invalid confidence {conf}")
        # Check coordinates
        if x1 < 0:
            violations.append(f"[{os.path.basename(f)}] Region {idx}: x1={x1} < 0")
        if y1 < 0:
            violations.append(f"[{os.path.basename(f)}] Region {idx}: y1={y1} < 0")
        if x2 > img_w:
            violations.append(f"[{os.path.basename(f)}] Region {idx}: x2={x2} > img_w={img_w}")
        if y2 > img_h:
            violations.append(f"[{os.path.basename(f)}] Region {idx}: y2={y2} > img_h={img_h}")
        if x1 >= x2:
            violations.append(f"[{os.path.basename(f)}] Region {idx}: x1={x1} >= x2={x2}")
        if y1 >= y2:
            violations.append(f"[{os.path.basename(f)}] Region {idx}: y1={y1} >= y2={y2}")

print(f"\nTotal regions verified: {total_regions}")
if violations:
    print(f"FAILURE: Found {len(violations)} violations:")
    for v in violations:
        print(" -", v)
    exit(1)
else:
    print("ALL REGIONS FULLY VERIFIED! Zero violations found.")
    exit(0)
