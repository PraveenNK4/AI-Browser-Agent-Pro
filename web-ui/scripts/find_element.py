import json

snap = json.load(open('tmp/dom_snapshots/aba1497f-eb8c-4fb0-991f-fe83d00d833b/24.json'))

# Find the anchor element with AdminServer-01
admin_link = [el for el in snap['elements'] if 'AdminServer-01' in str(el.get('selector_provenance', {}))][0]
print(f"Found AdminServer-01 link at index {admin_link['identity']['order']}")
print(f"CSS: {admin_link['selector_provenance']['css']}\n")

# Now find parent tables
tables = [el for el in snap['elements'] if el['identity']['tagName'] == 'table']
print(f"Total tables found: {len(tables)}\n")

# Find tables with admin or server in their selectors
data_tables = []
for el in tables:
    selector = el.get('selector_provenance', {})
    css = selector.get('css', '').lower()
    text = selector.get('text', '').lower()
    outerhtml = el.get('integrity', {}).get('outerHTML', '').lower()
    
    # Look for tables containing server/admin info
    if 'admin' in css or 'server' in text or 'adminserver-01' in outerhtml:
        data_tables.append(el)
        print(f"Index {el['identity']['order']}: table")
        print(f"  ID: {el['attribute_fingerprint'].get('id', 'N/A')}")
        print(f"  CSS: {css[:100]}")
        print(f"  Has AdminServer-01: {'adminserver-01' in outerhtml}")
        print()

if data_tables:
    print(f"\nBest candidate is likely index {data_tables[0]['identity']['order']}")

