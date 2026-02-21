import json

snap = json.load(open('tmp/dom_snapshots/aba1497f-eb8c-4fb0-991f-fe83d00d833b/24.json'))

# Find ALL elements that contain AdminServer-01 in their HTML
candidates = []
for el in snap['elements']:
    html = el.get('integrity', {}).get('outerHTML', '').lower()
    if 'adminserver-01' in html:
        candidates.append(el)

print(f"Found {len(candidates)} elements containing 'AdminServer-01'\n")

for el in candidates[:15]:
    tag = el['identity']['tagName']
    order = el['identity']['order']
    elem_id = el['attribute_fingerprint'].get('id', 'N/A')
    css = el['selector_provenance'].get('css', '')[:100]
    text = el['selector_provenance'].get('text', '')[:80]
    
    print(f"Index {order}: <{tag}> id={elem_id}")
    print(f"  CSS: {css}")
    print(f"  Text: {text}")
    print()
