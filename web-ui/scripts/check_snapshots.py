import json
import pathlib

snapshot_dir = pathlib.Path("tmp/dom_snapshots/75215413-b5fe-422f-9052-4cbe65dd10dc")
snapshots = sorted(snapshot_dir.glob("*.json"), key=lambda p: int(p.stem))

print("Checking snapshots 1 and 2 for input fields:\n")
for snap_path in snapshots[1:3]:
    with open(snap_path, 'r', encoding='utf-8') as f:
        snap = json.load(f)
    
    meta = snap.get('metadata', {})
    print(f"{snap_path.stem}: {meta.get('reason', '')}")
    print(f"  Action params: {meta.get('action_params', {})}")
    
    elements = snap.get('elements', [])
    inputs = [e for e in elements if e.get('identity', {}).get('tagName') in ['input', 'textarea']]
    
    print("  Input fields:")
    for inp in inputs:
        ident = inp.get('identity', {})
        attrs = inp.get('attributes', {})
        state = inp.get('state', {})
        print(f"    Order {ident.get('order')}: type={attrs.get('type', 'N/A')} id={attrs.get('id', 'N/A')} visible={state.get('visible')}")
    print()
