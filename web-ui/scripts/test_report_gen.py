import pathlib
import sys
sys.path.insert(0, 'web-ui/src')

from utils.generate_playwright_report import generate_playwright_report

run_id = "277132fe-277e-4be7-83e6-3bdb33bcd7c0"
history_path = pathlib.Path(f"tmp/agent_history/{run_id}/{run_id}.json")
screenshots_dir = pathlib.Path(f"tmp/agent_history/{run_id}/{run_id}_screenshots")

print(f"Testing report generation for {run_id}...")
print(f"History file exists: {history_path.exists()}")
print(f"Screenshots dir exists: {screenshots_dir.exists()}")

report_path = generate_playwright_report(
    history_path,
    screenshots_dir=screenshots_dir if screenshots_dir.exists() else None
)

if report_path:
    print(f"\n✓ Report generated: {report_path}")
    print(f"✓ Report size: {report_path.stat().st_size} bytes")
else:
    print("\n✗ Report generation failed")
