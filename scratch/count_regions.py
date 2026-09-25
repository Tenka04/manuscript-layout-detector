import json, os

json_dir = "results/json"
for f in sorted(os.listdir(json_dir)):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(json_dir, f)) as fp:
        data = json.load(fp)
    regions = data.get("regions", [])
    counts = {}
    for r in regions:
        label = r["label"]
        counts[label] = counts.get(label, 0) + 1
    print(f"{f}: total={len(regions)}, {counts}")
