import json, os

for result_dir, label in [("results/json", "BEFORE"), ("results_after_fix/json", "AFTER")]:
    print(f"\n=== {label} ({result_dir}) ===")
    if not os.path.exists(result_dir):
        print("  Directory not found")
        continue
    for f in sorted(os.listdir(result_dir)):
        if not f.endswith('.json'):
            continue
        with open(os.path.join(result_dir, f)) as fp:
            data = json.load(fp)
        regions = data.get("regions", [])
        counts = {}
        for r in regions:
            label_r = r["label"]
            counts[label_r] = counts.get(label_r, 0) + 1
        print(f"  {f}: total={len(regions)}, {counts}")
