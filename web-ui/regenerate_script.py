
import sys
import os
import json
from pathlib import Path

# Add src to sys.path
sys.path.append(os.path.abspath(r"C:\Users\pnandank\Downloads\Dynamic_Scrapper_4008 1\Dynamic_Scrapper_4008\Dynamic_Scrapper\web-ui"))

from src.utils.generate_playwright_script import generate_playwright_script

history_path = Path(r"C:\Users\pnandank\Downloads\Dynamic_Scrapper_4008 1\Dynamic_Scrapper_4008\Dynamic_Scrapper\web-ui\tmp\agent_history\85d5682e-3109-474a-8e1f-3f4caf0978a2\85d5682e-3109-474a-8e1f-3f4caf0978a2_p1.json")
snapshot_dir = Path(r"C:\Users\pnandank\Downloads\Dynamic_Scrapper_4008 1\Dynamic_Scrapper_4008\Dynamic_Scrapper\web-ui\tmp\dom_snapshots")
output_path = Path(r"C:\Users\pnandank\Downloads\Dynamic_Scrapper_4008 1\Dynamic_Scrapper_4008\Dynamic_Scrapper\web-ui\tmp\agent_history\85d5682e-3109-474a-8e1f-3f4caf0978a2\process_1_playwright_fixed.py")

sensitive_data = {
    "{{OTCS_USERNAME}}": "otadmin@otds.admin",
    "{{OTCS_PASSWORD}}": "Otds@123"
}

try:
    script_content = generate_playwright_script(history_path, snapshot_dir, sensitive_data=sensitive_data)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    print(f"Successfully regenerated script at {output_path}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
