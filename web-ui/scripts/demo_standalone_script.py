"""
Demonstration: Running the generated Playwright script standalone 
to show automatic report generation with screenshots.
"""

import pathlib
import subprocess
import sys

run_id = "277132fe-277e-4be7-83e6-3bdb33bcd7c0"
script_path = pathlib.Path(f"tmp/agent_history/{run_id}/{run_id}.py")

print("=" * 80)
print("DEMONSTRATION: Running Playwright Script Standalone")
print("=" * 80)
print(f"\nScript: {script_path}")
print(f"This script will:")
print("  1. Execute the automation tasks")
print("  2. Capture screenshots at each step")
print("  3. Automatically generate a report with all screenshots")
print("\n" + "=" * 80)
print("Script Preview (first 30 lines):")
print("=" * 80 + "\n")

with open(script_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines[:30], 1):
        print(f"{i:3}: {line.rstrip()}")

print("\n" + "=" * 80)
print("Expected Output When Run:")
print("=" * 80)
print("""
Step 1: go_to_url
Step 2: input_text
Step 3: input_text
Step 4: click_element_by_index
Step 5: go_to_url
Step 6: extract_content
Step 7: done

Screenshots saved to: ./277132fe-277e-4be7-83e6-3bdb33bcd7c0_screenshots
Execution log saved to: ./277132fe-277e-4be7-83e6-3bdb33bcd7c0_execution_log.json

Generating report from captured screenshots...
✓ Report generated: ./277132fe-277e-4be7-83e6-3bdb33bcd7c0_playwright_report.docx
Report ready at: ./277132fe-277e-4be7-83e6-3bdb33bcd7c0_playwright_report.docx
""")

print("=" * 80)
print("To run the script:")
print("=" * 80)
print(f"  python {script_path}")
print("\nOr with environment variables:")
print("  set OTCS_USERNAME=myusername")
print("  set OTCS_PASSWORD=mypassword")
print(f"  python {script_path}")
print("\n" + "=" * 80)
