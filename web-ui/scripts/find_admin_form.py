import json

snap = json.load(open('tmp/dom_snapshots/aba1497f-eb8c-4fb0-991f-fe83d00d833b/24.json'))

# Find the anchor element with AdminServer-01
admin_link = [el for el in snap['elements'] if 'AdminServer-01' in str(el.get('selector_provenance', {}))][0]
print(f"Found AdminServer-01 link at index {admin_link['identity']['order']}")
print(f"CSS: {admin_link['selector_provenance']['css'][:150]}...\n")

# Find form with id admin-servers
forms = [el for el in snap['elements'] if el['identity']['tagName'] == 'form']
print(f"Total forms found: {len(forms)}\n")

for form in forms:
    form_id = form['attribute_fingerprint'].get('id', '')
    if 'admin' in form_id.lower():
        print(f"Index {form['identity']['order']}: form#{form_id}")
        print(f"  CSS: {form.get('selector_provenance', {}).get('css', '')[:100]}")
        outerhtml = form.get('integrity', {}).get('outerHTML', '')
        print(f"  HTML length: {len(outerhtml)}")
        print(f"  Contains AdminServer-01: {'AdminServer-01' in outerhtml}")
        print()
