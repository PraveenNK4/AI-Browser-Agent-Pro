
import json
import sys

def find_target_elements(snapshot_path, search_terms, output_path):
    try:
        with open(snapshot_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        elements = data.get('elements', [])
        results = []
        for el in elements:
            state = el.get('state', {})
            prov = el.get('selector_provenance', {})
            integrity = el.get('integrity', {})
            
            search_blob = f"{prov.get('text', '')} {prov.get('aria', '')} {integrity.get('outerHTML', '')}".lower()
            
            if any(term.lower() in search_blob for term in search_terms):
                results.append({
                    'order': el.get('identity', {}).get('order'),
                    'tagName': el.get('identity', {}).get('tagName'),
                    'text': prov.get('text', ''),
                    'aria': prov.get('aria', ''),
                    'visible': state.get('visible'),
                    'outerHTML': integrity.get('outerHTML', '')[:200]
                })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"{'Order':<6} | {'Tag':<10} | {'Visible':<8} | {'ARIA':<30} | {'HTML Snippet'}\n")
            f.write("-" * 200 + "\n")
            for el in results:
                f.write(f"{el['order']:<6} | {el['tagName']:<10} | {str(el['visible']):<8} | {el['aria'][:30]:<30} | {el['outerHTML']}\n")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    snapshot = r'c:\Users\pnandank\Downloads\Dynamic_Scrapper_4008 1\Dynamic_Scrapper_4008\Dynamic_Scrapper\web-ui\tmp\dom_snapshots\b89818a4-e285-4612-bca7-4b4432dff278\20.json'
    find_target_elements(snapshot, ["Add", "Upload", "Create"], "elements_output.txt")
