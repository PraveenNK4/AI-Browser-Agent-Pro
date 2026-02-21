import json

snap = json.load(open('tmp/dom_snapshots/aba1497f-eb8c-4fb0-991f-fe83d00d833b/24.json'))
el = [e for e in snap['elements'] if e['identity']['order'] == 222][0]

print(f"Element 222: {el['identity']['tagName']}")
print(f"ID: {el['attribute_fingerprint'].get('id')}")
print(f"CSS: {el['selector_provenance'].get('css', '')[:150]}")
print(f"Text preview: {el['selector_provenance'].get('text', '')[:300]}")
print(f"\nHTML length: {len(el['integrity'].get('outerHTML', ''))}")

# Check if it contains admin server data
html = el['integrity'].get('outerHTML', '').lower()
print(f"Contains 'adminserver': {'adminserver' in html}")
print(f"Contains 'status': {'status' in html}")
print(f"Contains 'errors': {'errors' in html}")
