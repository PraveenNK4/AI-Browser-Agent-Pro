import json
import pathlib

snapshot_dir = pathlib.Path("tmp/dom_snapshots/75215413-b5fe-422f-9052-4cbe65dd10dc")
snapshots = sorted(snapshot_dir.glob("*.json"), key=lambda p: int(p.stem))

# Get snapshot 7 (first input_text pre_action)
with open(snapshots[7], 'r', encoding='utf-8') as f:
    snap = json.load(f)

elements = snap.get('elements', [])
inputs = [e for e in elements if e.get('identity', {}).get('tagName') in ['input', 'textarea']]

print("Checking input fields in snapshot 7:\n")
for inp in inputs:
    ident = inp.get('identity', {})
    attrs = inp.get('attributes', {})
    state = inp.get('state', {})
    selectors = inp.get('selector_provenance', {})
    
    print(f"Order {ident.get('order')}:")
    print(f"  Attributes: {attrs}")
    print(f"  State: {state}")
    print(f"  Selectors (preferred): {selectors.get('preferred_selector', 'N/A')}")
    print()
