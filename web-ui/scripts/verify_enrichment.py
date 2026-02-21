import json

with open('tmp/agent_history/75215413-b5fe-422f-9052-4cbe65dd10dc/75215413-b5fe-422f-9052-4cbe65dd10dc.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

actions = [a for step in data.get('history', []) for a in step.get('model_output', {}).get('action', []) if a]

print("Checking first 5 actions:\n")
for i, action in enumerate(actions[:5]):
    action_name = list(action.keys())[0]
    action_params = action.get(action_name, {})
    dom_ctx = action.get('dom_context', {})
    text = action_params.get('text', '')[:40] if isinstance(action_params, dict) else ''
    attrs = dom_ctx.get('attributes', {})
    selectors = dom_ctx.get('selectors', {})
    print(f"{i}: {action_name}")
    print(f"   Text: '{text}'")
    print(f"   Type: {attrs.get('type', 'N/A')}, ID: {attrs.get('id', 'N/A')}, Name: {attrs.get('name', 'N/A')}")
    print(f"   Selector (preferred): {selectors.get('preferred_selector', 'N/A')}")
    print(f"   Selector (css): {selectors.get('css', 'N/A')}")
    print()
