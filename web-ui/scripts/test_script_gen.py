import pathlib
import sys
sys.path.insert(0, 'web-ui/src')

from utils.generate_playwright_script import generate_playwright_script

run_id = "277132fe-277e-4be7-83e6-3bdb33bcd7c0"
history_path = pathlib.Path(f"tmp/agent_history/{run_id}/{run_id}.json")
script_path = pathlib.Path(f"tmp/agent_history/{run_id}/{run_id}.py")

if history_path.exists():
    print(f"Generating script for {run_id}...")
    script_content = generate_playwright_script(history_path)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✓ Script saved to: {script_path}")
    print(f"✓ Script size: {len(script_content)} chars")
else:
    print(f"✗ History file not found: {history_path}")
