"""End-to-end test: run generate_script on Admin Server history and save result."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.llm_script_generator import generate_script

history_path = r"tmp\agent_history\orchestration_20260222_205153\Process 1 Admin Server Validation\orchestration_20260222_205153_Process 1 Admin Server Validation.json"
output_path = r"tmp\_test_generated_script_v7.py"
objective = "Navigate to Admin Server page, extract the Status and Errors columns from the Admin Server table, and report results"

generate_script(
    history_path=history_path,
    output_path=output_path,
    objective=objective,
)
print(f"\nDone! Script saved to: {output_path}")
