"""
Example usage of the process orchestration system.

This demonstrates how to use the orchestration feature with the following capabilities:
- Mandatory processes get fresh browser contexts
- Non-mandatory processes share browser context and continue from same state
- Execution continues even if non-mandatory processes fail
- All output files are centralized in one directory
"""
import asyncio
import logging
from src.orchestrator import (
    parse_processes,
    run_all_processes,
    generate_process_summary,
    generate_combined_playwright_script,
)
from src.orchestrator.report_generator import save_summary_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_orchestration():
    """Example: Execute multiple processes with intelligent browser management."""
    
    # Define processes - Mandatory gets fresh context, others share context
    prompt = """
Mandatory Process: Login and Setup
- Navigate to https://example.com
- Enter username
- Enter password
- Click login

Process 1: First Task
- Fill out form A
- Submit form

Process 2: Second Task
- Navigate to settings
- Update preferences
- Save changes

Process 3: Cleanup
- Clear browser data
- Logout
"""
    
    # Parse processes from prompt
    processes = parse_processes(prompt)
    logger.info(f"Parsed {len(processes)} processes")
    
    # Setup (you would get these from your actual setup)
    # browser = ... (your browser instance)
    # llm = ... (your LLM instance)
    # kwargs = {...}
    
    # Run orchestration with centralized output
    # results, output_dir = await run_all_processes(
    #     processes=processes,
    #     browser=browser,
    #     llm=llm,
    #     orchestration_output_dir="./outputs/orchestration_run_1",
    #     **kwargs
    # )
    
    # Generate reports
    # summary = generate_process_summary(results)
    # save_summary_report(summary, output_dir)
    # generate_combined_playwright_script(results, f"{output_dir}/combined_script.py")
    
    logger.info("Example orchestration complete")


async def example_with_error_handling():
    """Example: Handle process failures gracefully."""
    
    processes_text = """
Mandatory Process: Critical Setup
- Initialize system
- Verify connections

Process 1: Optional Task A
- Do task A
- Might fail but continue

Process 2: Optional Task B
- Do task B
- Also continues on failure

Process 3: Final Task
- Cleanup
- Always runs if not aborted by mandatory failure
"""
    
    processes = parse_processes(processes_text)
    
    # In this example:
    # - If "Critical Setup" fails -> ABORT (it's mandatory)
    # - If "Optional Task A" fails -> CONTINUE
    # - If "Optional Task B" fails -> CONTINUE
    # - "Final Task" always runs unless mandatory process failed
    
    logger.info(f"Setup for error handling example with {len(processes)} processes")


if __name__ == "__main__":
    # Note: You need to setup your browser, llm, and other components before running
    # asyncio.run(example_orchestration())
    # asyncio.run(example_with_error_handling())
    
    logger.info("Orchestration examples defined. See docstrings for usage.")
