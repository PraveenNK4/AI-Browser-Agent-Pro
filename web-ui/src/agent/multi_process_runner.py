
import asyncio
import re
import logging
import os
import pathlib
import uuid
import time
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass, field

from browser_use.agent.service import Agent as BrowserUseAgent
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.agent.views import AgentHistoryList
from src.utils.llm_script_generator import generate_script as generate_llm_script
from src.utils.strip_and_enrich_history import strip_and_enrich
from src.utils.generate_playwright_report import generate_playwright_report
from src.utils.utils import slugify

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    """Result of a single process execution."""
    name: str
    status: str  # "PASS", "FAIL", "TIMEOUT", "STOPPED"
    duration: float
    errors: List[str] = field(default_factory=list)
    history_file: Optional[str] = None
    script_file: Optional[str] = None
    gif_file: Optional[str] = None
    docx_file: Optional[str] = None


class MultiProcessRunner:
    def __init__(
        self,
        browser: Browser,
        browser_context: BrowserContext,
        llm,
        controller = None,
        page_extraction_llm=None,
        use_vision: bool = True,
        max_steps: int = 25,
        max_actions: int = 3,
        max_input_tokens: int = 128000,
        available_file_paths: Optional[List[str]] = None,
        save_agent_history_path: str = "./tmp/agent_history",
        controller_class = None,  # Class ref to instantiate fresh
        sensitive_data: Optional[Dict[str, str]] = None
    ):
        self.browser = browser
        self.browser_context = browser_context
        self.controller = controller
        self.controller_class = controller_class
        self.sensitive_data = sensitive_data or {}
        self.llm = llm
        self.page_extraction_llm = page_extraction_llm
        self.use_vision = use_vision
        self.max_steps = max_steps
        self.max_actions = max_actions
        self.max_input_tokens = max_input_tokens
        self.available_file_paths = available_file_paths or []
        self.save_agent_history_path = save_agent_history_path
        
        self.task_id = str(uuid.uuid4())
        self.process_results: List[ProcessResult] = []
        pathlib.Path(self.save_agent_history_path).mkdir(parents=True, exist_ok=True)

    def _parse_prompt(self, full_prompt: str) -> List[Dict[str, str]]:
        """Splits the prompt into sequential processes."""
        text = full_prompt.strip()
        
        # Check for horizontal rule splitting
        if "\n---" in text:
            raw_segments = re.split(r'\n-{3,}\n', text)
            tasks = []
            for i, seg in enumerate(raw_segments):
                if seg.strip():
                    # Try to extract a name from the segment
                    name = self._extract_process_name(seg, i + 1)
                    tasks.append({"name": name, "prompt": seg.strip()})
            return tasks

        # Check for "Process N:" pattern
        process_pattern = re.compile(r'(?:^|\n)Process\s+\d+:', re.IGNORECASE)
        if process_pattern.search(text):
            lines = text.split('\n')
            tasks = []
            current_task_lines = []
            current_name = "Process 1"
            
            for line in lines:
                match = re.match(r'^Process\s+(\d+):\s*(.*)', line, re.IGNORECASE)
                if match:
                    if current_task_lines:
                        tasks.append({"name": current_name, "prompt": "\n".join(current_task_lines).strip()})
                    # Extract name after "Process N:"
                    remainder = match.group(2).strip()
                    if remainder:
                        current_name = f"Process {match.group(1)}: {remainder.split('.')[0]}"
                    else:
                        current_name = f"Process {match.group(1)}"
                    current_task_lines = [remainder] if remainder else []
                else:
                    current_task_lines.append(line)
            
            if current_task_lines:
                tasks.append({"name": current_name, "prompt": "\n".join(current_task_lines).strip()})
            
            if tasks:
                return tasks

        return [{"name": "Process 1", "prompt": text}]

    def _extract_process_name(self, segment: str, index: int) -> str:
        """Extract a meaningful name from process segment."""
        lines = segment.strip().split('\n')
        first_line = lines[0].strip() if lines else ""
        if first_line and len(first_line) < 60:
            return f"Process {index}: {first_line[:50]}"
        return f"Process {index}"

    def _enrich_prompt(self, prompt: str) -> str:
        """Add robustness features to the prompt."""
        tool_guide = """
Tool Parameter Guide:
- validation: Use 'equals', 'contains', 'running' (for status)
- input_text: Use index for element selection.
- clicks: Prefer index unless text is unique.
- timeouts: Operations time out after 600s.
"""
        negative_constraints = """
CRITICAL INSTRUCTION: You must strictly adhere to all negative constraints. 
1. Do not hallucinate completion. Verify every step with DOM view.
2. DO NOT navigate to individual item pages if the data is available in the list/table.
3. Extract data from the current view. Do NOT click on rows unless explicitly asked to "click" or "drill down".
4. Once data is extracted, STOP. Do not explore further.
"""
        return f"{negative_constraints}\n{tool_guide}\n\nTASK:\n{prompt}"

    async def run(
        self, 
        full_prompt: str, 
        step_callback=None, 
        done_callback=None, 
        on_agent_created=None,
        on_process_start=None,
        on_process_complete=None
    ):
        """
        Run multi-process execution.
        
        Callbacks:
            step_callback: Called after each agent step
            done_callback: Called when all processes complete
            on_agent_created: Called when a new agent is created
            on_process_start: Called when a process starts (name, index, total)
            on_process_complete: Called when a process completes (ProcessResult)
        """
        tasks = self._parse_prompt(full_prompt)
        total_processes = len(tasks)
        logger.info(f"MultiProcessRunner: Identified {total_processes} processes.")
        
        overall_history = []
        self.process_results = []
        total_start_time = time.time()
        
        for i, task_info in enumerate(tasks):
            process_name = task_info['name']
            raw_prompt = task_info['prompt']
            enriched_prompt = self._enrich_prompt(raw_prompt)
            
            # Notify UI of process start
            if on_process_start:
                await on_process_start(process_name, i + 1, total_processes)
            
            logger.info(f"Starting {process_name}...")
            process_start_time = time.time()
            process_errors = []
            status = "PASS"
            
            # Create Agent with GIF generation for this process
            history_dir = pathlib.Path(self.save_agent_history_path) / self.task_id
            history_dir.mkdir(parents=True, exist_ok=True)
            gif_path = str(history_dir / f"process_{i+1}.gif")
            
            # Create FRESH controller if class provided (Fixes Windows Multiprocessing/Pickling issues)
            current_controller = self.controller
            if self.controller_class:
                logger.info(f"Instantiating FRESH controller for {process_name}")
                current_controller = self.controller_class()
                # Ensure context file paths and sensitive data are passed to new controller
                current_controller.available_file_paths = self.available_file_paths
                current_controller.sensitive_data = self.sensitive_data

            agent = BrowserUseAgent(
                task=enriched_prompt,
                llm=self.llm,
                browser=self.browser,
                browser_context=self.browser_context,
                controller=current_controller,
                use_vision=self.use_vision,
                max_input_tokens=self.max_input_tokens,
                max_actions_per_step=self.max_actions,
                register_new_step_callback=step_callback,
                available_file_paths=self.available_file_paths,
                generate_gif=gif_path,  # Per-process GIF
            )
            
            if on_agent_created:
                on_agent_created(agent)
            
            process_id = f"{self.task_id}_p{i+1}"
            agent.state.agent_id = process_id
            
            # Run with Timeout
            try:
                await asyncio.wait_for(agent.run(max_steps=self.max_steps), timeout=600.0)
                # Check if agent completed successfully
                if agent.state.stopped:
                    status = "STOPPED"
            except asyncio.TimeoutError:
                logger.error(f"{process_name} Timed out after 600s.")
                status = "TIMEOUT"
                process_errors.append("Process timed out after 600s")
            except Exception as e:
                logger.error(f"{process_name} Failed: {e}")
                status = "FAIL"
                process_errors.append(str(e))
            
            # Collect errors from agent history
            if agent.state.history:
                for item in agent.state.history.history:
                    if hasattr(item, 'result') and item.result:
                        for action_result in item.result:
                            if hasattr(action_result, 'error') and action_result.error:
                                process_errors.append(str(action_result.error))
            
            # Separate hard errors from defensive blocks (prefixed with 🚫)
            hard_errors = [e for e in process_errors if "🚫" not in e]
            defensive_blocks = [e for e in process_errors if "🚫" in e]

            # Improved status detection
            if status == "PASS":
                # Check if agent actually completed (has is_done flag)
                agent_completed = False
                if agent.state.history and agent.state.history.history:
                    last_item = agent.state.history.history[-1]
                    if hasattr(last_item, 'result') and last_item.result:
                        for action_result in last_item.result:
                            if hasattr(action_result, 'is_done') and action_result.is_done:
                                agent_completed = True
                                break
                
                # Determine final status
                if hard_errors:
                    # If hard errors occurred, it's at best PARTIAL
                    status = "PARTIAL" if agent_completed else "FAIL"
                elif not agent_completed:
                    # No hard errors but didn't finish normally
                    status = "PARTIAL"
                else:
                    # COMPLETED and NO HARD ERRORS (Defensive blocks allowed)
                    status = "PASS"
            
            process_duration = time.time() - process_start_time
            
            # Save artifacts for this process
            result = await self._save_process_artifacts(
                agent, process_id, process_name, i + 1, 
                status, process_duration, process_errors, gif_path
            )
            self.process_results.append(result)
            
            # Notify UI of process completion
            if on_process_complete:
                await on_process_complete(result)
            
            if agent.state.history:
                overall_history.extend(agent.state.history.history)
                
            if agent.state.stopped:
                logger.info("Process stopped by user. Aborting subsequent processes.")
                break

        total_duration = time.time() - total_start_time
        
        if done_callback:
            combined_history_list = AgentHistoryList(history=overall_history)
            await done_callback(combined_history_list)
        
        return self.process_results, total_duration

    async def _save_process_artifacts(
        self, agent, process_id, process_name, process_num,
        status, duration, errors, gif_path
    ) -> ProcessResult:
        """Generates artifacts for a single process."""
        history_dir = pathlib.Path(self.save_agent_history_path) / self.task_id
        history_dir.mkdir(parents=True, exist_ok=True)
        
        slug = slugify(process_name)
        history_file = history_dir / f"{process_id}.json"
        script_file = history_dir / f"{slug}_playwright.py"
        docx_file = history_dir / f"{slug}_report.docx"
        
        result = ProcessResult(
            name=process_name,
            status=status,
            duration=duration,
            errors=errors[:5],  # Limit to first 5 errors
            history_file=str(history_file) if history_file.exists() else None,
            gif_file=gif_path if os.path.exists(gif_path) else None,
        )
        

        try:
            # Save history JSON
            agent.save_history(str(history_file))
            result.history_file = str(history_file)
            
            # Generate Playwright script via LLM
            if history_file.exists():
                try:
                    # Determine model and provider from self.llm
                    model_name = "qwen2.5:14b" # Default fallback
                    provider = "ollama"
                    
                    if hasattr(self.llm, 'model_name'):
                        model_name = self.llm.model_name
                    elif hasattr(self.llm, 'model'):
                        model_name = self.llm.model
                        
                    logger.info(f"Generating Playwright script via LLM ({model_name})...")
                    script_path, script_content = generate_llm_script(
                        str(history_file),
                        model_name=model_name,
                        provider=provider
                    )
                    # result.script_file is set to the generated path
                    result.script_file = script_path
                except Exception as script_e:
                    logger.warning(f"Failed to generate LLM script for {process_name}: {script_e}")
                    # Write simple fallback script if LLM fails
                    with open(script_file, 'w', encoding='utf-8') as f:
                        f.write(f"# Script generation failed: {script_e}\n# Check history JSON for details.")
                    result.script_file = str(script_file)
            
            # Generate DOCX report for this process
            if history_file.exists():
                try:
                    report_path = generate_playwright_report(
                        history_json_path=history_file
                    )
                    if report_path and report_path.exists():
                        # Rename to our convention
                        import shutil
                        shutil.move(str(report_path), str(docx_file))
                        result.docx_file = str(docx_file)
                except Exception as report_e:
                    logger.warning(f"Failed to generate DOCX for {process_name}: {report_e}")
                    # Create a text-based error report as fallback
                    error_txt = docx_file.with_suffix('.txt')
                    with open(error_txt, 'w', encoding='utf-8') as f:
                        f.write(f"PROCESS FAILED: {process_name}\n")
                        f.write(f"STATUS: {status}\n")
                        f.write(f"ERRORS: {errors}\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"Report generation failed: {report_e}\n")
                    result.docx_file = str(error_txt)  # Point to text file instead
                    
        except Exception as e:
            logger.error(f"Failed to save artifacts for {process_name}: {e}")
            # Last resort fallback
            try:
                with open(history_dir / f"process_{process_num}_FATAL.txt", 'w') as f:
                    f.write(f"Process {process_name} crashed and failed to save history.\nError: {e}\nProcess Errors: {errors}")
            except:
                pass
        
        return result

    def get_summary_markdown(self, total_duration: float) -> str:
        """Generate a markdown summary table of all process results."""
        passed = sum(1 for r in self.process_results if r.status == "PASS")
        partial = sum(1 for r in self.process_results if r.status == "PARTIAL")
        failed = sum(1 for r in self.process_results if r.status in ["FAIL", "TIMEOUT", "STOPPED"])
        total = len(self.process_results)
        
        lines = [
            "## Final Report",
            "",
            f"**Summary:** ✅ {passed} passed | ⚠️ {partial} partial | ❌ {failed} failed (of {total})",
            "",
            "| Process | Status | Duration | Details |",
            "|---------|--------|----------|---------|",
        ]
        
        for result in self.process_results:
            # Status icons: PASS=✅, FAIL=❌, PARTIAL=⚠️, TIMEOUT=⏱️, STOPPED=🛑
            if result.status == "PASS":
                status_icon = "✅"
                detail = "Completed successfully"
            elif result.status == "FAIL":
                status_icon = "❌"
                detail = f"Errors: {result.errors[0][:50]}..." if result.errors else "Failed"
            elif result.status == "PARTIAL":
                status_icon = "⚠️"
                detail = f"Partial: {result.errors[0][:50]}..." if result.errors else "Incomplete"  
            elif result.status == "TIMEOUT":
                status_icon = "⏱️"
                detail = "Timed out"
            else:
                status_icon = "🛑"
                detail = "Stopped"
            lines.append(f"| {result.name} | {status_icon} {result.status} | {result.duration:.1f}s | {detail} |")
        
        lines.append("")
        lines.append(f"**Total Duration:** {total_duration:.1f}s")
        
        return "\n".join(lines)
