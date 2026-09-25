import json

for fname in ["6.json", "7.json", "3.json"]:
    path = f"results_after_fix/json/{fname}"
    try:
        data = json.load(open(path))
    except:
        print(f"Cannot open {path}")
        continue
    print(f"\n=== {fname} ===")
    for r in data["regions"]:
        b = r["bbox"]
        sub = r.get("sub_type", "")
        print(f"  {r['label']:12s} conf={r['confidence']:5.2f}  x={b['x']:5d} y={b['y']:5d} w={b['w']:5d} h={b['h']:4d}  {sub}")
