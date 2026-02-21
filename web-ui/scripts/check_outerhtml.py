import json
import pathlib

snapshot_dir = pathlib.Path("tmp/dom_snapshots/75215413-b5fe-422f-9052-4cbe65dd10dc")
snapshots = sorted(snapshot_dir.glob("*.json"), key=lambda p: int(p.stem))

# Get snapshot 7 and 9 (input_text pre_action)
for snap_num in [7, 9]:
    with open(snapshots[snap_num], 'r', encoding='utf-8') as f:
        snap = json.load(f)
    
    meta = snap.get('metadata', {})
    print(f"\nSnapshot {snap_num}: {meta.get('reason')} - {meta.get('action_params', {})}")
    
    elements = snap.get('elements', [])
    inputs = [e for e in elements if e.get('identity', {}).get('tagName') in ['input', 'textarea']]
    
    print("Input fields (visible ones):")
    for inp in inputs:
        if inp.get('state', {}).get('visible', False):
            ident = inp.get('identity', {})
            outerhtml = inp.get('integrity', {}).get('outerHTML', '')[:150]
            print(f"  Order {ident.get('order')}: {outerhtml}")
