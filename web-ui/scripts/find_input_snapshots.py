import json
import pathlib

snapshot_dir = pathlib.Path("tmp/dom_snapshots/75215413-b5fe-422f-9052-4cbe65dd10dc")
snapshots = sorted(snapshot_dir.glob("*.json"), key=lambda p: int(p.stem))

print(f"Total snapshots: {len(snapshots)}\n")
print("Looking for pre_action:input_text snapshots:\n")

for snap_path in snapshots:
    with open(snap_path, 'r', encoding='utf-8') as f:
        snap = json.load(f)
    
    meta = snap.get('metadata', {})
    reason = meta.get('reason', '')
    
    if 'input_text' in reason:
        print(f"\n{snap_path.stem}: {reason}")
        print(f"  Action params: {meta.get('action_params', {})}")
        
        elements = snap.get('elements', [])
        inputs = [e for e in elements if e.get('identity', {}).get('tagName') in ['input', 'textarea']]
        
        print(f"  Input fields found: {len(inputs)}")
        for inp in inputs:
            ident = inp.get('identity', {})
            attrs = inp.get('attributes', {})
            state = inp.get('state', {})
            print(f"    Order {ident.get('order')}: type={attrs.get('type', 'N/A')} id={attrs.get('id', 'N/A')} visible={state.get('visible')}")
