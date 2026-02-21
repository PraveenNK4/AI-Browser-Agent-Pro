import json
import os

file_path = r"c:\Users\pnandank\Downloads\Dynamic_Scrapper_4008 1\Dynamic_Scrapper_4008\Dynamic_Scrapper\web-ui\tmp\agent_history\20260202_103114_navigate-to-httpotcsvmotxlabnet8080otcscsexeappnod_4865526c\20260202_103114_navigate-to-httpotcsvmotxlabnet8080otcscsexeappnod_4865526c.json"

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

history = data['history']

# Fix Step 7: "Add Item" click
# Original was pointing to "Create Business Workspace"
step_7 = next((item for item in history if item['metadata']['step_number'] == 7), None)
if step_7:
    print("Found Step 7. Patching 'Add Item' selector...")
    dom = step_7['model_output']['action'][0]['dom_context']
    
    # Update selectors
    dom['selectors']['css'] = "li.csui-more-dropdown-wrapper > a[title='Add item']"
    dom['selectors']['xpath'] = "//li[contains(@class, 'csui-more-dropdown-wrapper')]/a[@title='Add item']"
    dom['selectors']['preferred'] = "li.csui-more-dropdown-wrapper > a[title='Add item']"
    
    # Update comprehensive selectors
    dom['comprehensive']['recommended_selector'] = {
        "type": "css",
        "selector": "li.csui-more-dropdown-wrapper > a[title='Add item']",
        "stability": 100,
        "description": "Corrected Selector via Patch"
    }
    
    # Update attributes to match reality
    dom['attributes']['title'] = "Add item"
    dom['attributes']['aria-label'] = "Add item"
    dom['tagName'] = "a"
    dom['outerHTML'] = '<a href="#" role="button" class="binf-dropdown-toggle csui-acc-focusable" title="Add item" aria-label="Add item">...</a>'

# Fix Step 8: "Document" click (Upload)
# Original was pointing to "Favorites"
step_8 = next((item for item in history if item['metadata']['step_number'] == 8), None)
if step_8:
    print("Found Step 8. Patching 'Document' selector...")
    dom = step_8['model_output']['action'][0]['dom_context']
    
    # Update selectors
    dom['selectors']['css'] = "li[data-csui-command='add'][data-csui-addtype='144'] > a"
    dom['selectors']['xpath'] = "//li[@data-csui-command='add' and @data-csui-addtype='144']/a"
    dom['selectors']['text'] = "Document"
    dom['selectors']['preferred'] = "li[data-csui-command='add'][data-csui-addtype='144'] > a"
    
    # Update comprehensive selectors
    dom['comprehensive']['recommended_selector'] = {
        "type": "css",
        "selector": "li[data-csui-command='add'][data-csui-addtype='144'] > a",
        "stability": 100,
        "description": "Corrected Selector via Patch"
    }
    
    # Update attributes
    dom['attributes']['class_list'] = ["csui-toolitem", "csui-toolitem-textonly"]
    dom['attributes']['title'] = "Document" # Often has title matched to text
    dom['tagName'] = "a"
    dom['outerHTML'] = '<a href="#" class="csui-toolitem csui-toolitem-textonly">Document</a>'

# Fix Step 5: "Sign In" click (Login)
# Original was pointing to hidden "#signup" input
step_5 = next((item for item in history if item['metadata']['step_number'] == 5), None)
if step_5:
    print("Found Step 5. Patching 'Sign In' selector...")
    dom = step_5['model_output']['action'][0]['dom_context']
    
    # Update selectors
    dom['selectors']['css'] = "#loginbutton"
    dom['selectors']['xpath'] = "//*[@id='loginbutton']"
    dom['selectors']['preferred'] = "#loginbutton"
    
    # Update comprehensive selectors
    dom['comprehensive']['recommended_selector'] = {
        "type": "id",
        "selector": "#loginbutton",
        "stability": 100,
        "description": "Corrected Selector via Patch"
    }
    
    # Update attributes
    dom['attributes']['id'] = "loginbutton"
    dom['attributes']['type'] = "submit"
    dom['attributes']['class_list'] = ["submit-btn", "highlighted-btn"]
    del dom['attributes']['name'] # loginbutton doesn't have name="signup"
    
    # Update state
    dom['state']['visible'] = True
    dom['state']['enabled'] = True
    
    dom['tagName'] = "button"
    dom['outerHTML'] = '<button id="loginbutton" type="submit" class="submit-btn highlighted-btn">Sign in</button>'

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("Successfully patched history.json")
