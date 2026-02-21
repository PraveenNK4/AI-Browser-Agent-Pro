
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from src.utils.generate_playwright_script import generate_playwright_script

history_path = Path(r"c:\Users\pnandank\Downloads\Dynamic_Scrapper_4008 1\Dynamic_Scrapper_4008\Dynamic_Scrapper\web-ui\tmp\agent_history\20260202_103114_navigate-to-httpotcsvmotxlabnet8080otcscsexeappnod_4865526c\20260202_103114_navigate-to-httpotcsvmotxlabnet8080otcscsexeappnod_4865526c.json")
output_path = Path(r"c:\Users\pnandank\Downloads\Dynamic_Scrapper_4008 1\Dynamic_Scrapper_4008\Dynamic_Scrapper\web-ui\tmp\agent_history\20260202_103114_navigate-to-httpotcsvmotxlabnet8080otcscsexeappnod_4865526c\20260202_103114_navigate-to-httpotcsvmotxlabnet8080otcscsexeappnod_4865526c_FIXED.py")

script = generate_playwright_script(history_path)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(script)

print(f"Fixed script generated at: {output_path}")
