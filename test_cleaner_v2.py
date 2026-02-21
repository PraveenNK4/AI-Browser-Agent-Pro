import json
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "web-ui"))

from src.utils.llm_script_generator import clean_history_json

history_path = r"c:\Users\pnandank\Downloads\Dynamic_Scrapper_4008 1\Dynamic_Scrapper_4008\Dynamic_Scrapper\web-ui\tmp\agent_history\20260203_100128_navigate-to-httpotcsvmotxlabnet8080otcscsexeappnod_25a48e68\20260203_100128_navigate-to-httpotcsvmotxlabnet8080otcscsexeappnod_25a48e68.json"
output_path = "test_cleaned_v2.json"

if os.path.exists(history_path):
    with open(history_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned = clean_history_json(data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    
    print(f"✅ Cleaned JSON saved to {output_path}")
else:
    print(f"❌ History file not found: {history_path}")
