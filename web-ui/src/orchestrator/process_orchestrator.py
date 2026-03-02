"""Process orchestration for multi-step automation tasks.

Key Features:
- Fresh browser context for each MANDATORY process
- Shared browser context for non-mandatory processes (continues from previous state)
- Continues execution even if non-mandatory processes fail
- All output files stored in a centralized orchestration directory
- Isolated controllers per process for clean separation
"""
import asyncio
import os
import logging
from typing import List, Optional, Dict
from datetime import datetime

from .process_models import Process, ProcessResult
from .exceptions import ProcessFailed, ProcessHallucinationDetected
from src.utils.dom_snapshot import set_run_id

logger = logging.getLogger(__name__)


def reset_process_ui_state(webui_manager):
    """Reset UI state for a new process to enforce isolation."""
    if webui_manager:
        # Fully reset failure tracking and stop signals
        webui_manager.bu_last_action = None
        webui_manager.bu_repeated_action_count = 0
        webui_manager.bu_previous_tokens = 0
        webui_manager.bu_step_start_time = None
        webui_manager.bu_hallucination_triggered = False
        webui_manager.bu_should_stop_agent = False
        webui_manager.bu_consecutive_failures = 0
        webui_manager.bu_step_failures = {}


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for file system compatibility."""
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = __import__('re').sub(invalid_chars, '_', filename)
    return sanitized.strip('_')


async def run_process(
    process: Process,
    browser,
    browser_context,
    llm,
    orchestration_output_dir: str,
    agent_history_path: str = "./tmp/agent_history",
    planner_llm=None,
    planner_system_message=None,
    page_extraction_llm=None,
    available_file_paths: List[str] = None,
    use_vision: bool = False,
    max_input_tokens: int = 128000,
    max_actions_per_step: int = 3,
    max_steps: int = 25,
    ask_callback=None,
    step_callback=None,
    done_callback=None,
    webui_manager=None,
    sensitive_data: dict = None,
    extend_system_prompt: str = None,
) -> ProcessResult:
    """
    Execute a single process with isolated agent and controller.
    
    Args:
        process: Process definition
        browser: Browser instance
        browser_context: Browser context (fresh for mandatory, shared for others)
        llm: Language model for agent
        orchestration_output_dir: Centralized output directory for all files
        agent_history_path: Base directory for history/report generation
        planner_llm: Optional planner LLM
        planner_system_message: Optional planner constraints
        page_extraction_llm: Optional page extraction LLM
        available_file_paths: Available file paths for agent
        use_vision: Whether to use vision
        max_input_tokens: Max input tokens
        max_actions_per_step: Max actions per step
        max_steps: Max steps to execute
        ask_callback: Callback for user input
        step_callback: Callback for step completion
        done_callback: Callback for process completion
        webui_manager: Optional web UI manager
        sensitive_data: Optional sensitive data vault
        
    Returns:
        ProcessResult with execution details
    """
    from src.agent.browser_use.browser_use_agent import BrowserUseAgent
    from src.controller.custom_controller import CustomController
    from browser_use.agent.views import AgentHistoryList
    
    # Create process-specific directory within orchestration output
    process_name_sanitized = sanitize_filename(process.name)
    process_dir = os.path.join(orchestration_output_dir, process_name_sanitized)
    os.makedirs(process_dir, exist_ok=True)

    # Determine task identifier aligned with existing single-process flow
    process_dir_absolute = os.path.abspath(process_dir)
    agent_history_root = os.path.abspath(agent_history_path)
    try:
        task_identifier = os.path.relpath(process_dir_absolute, agent_history_root)
    except ValueError:
        task_identifier = process_name_sanitized

    task_identifier = task_identifier.replace("\\", "/")

    if webui_manager:
        webui_manager.bu_agent_task_id = task_identifier
        webui_manager.bu_docx_path = None
    
    task_str = "\n".join(process.steps) if process.steps else process.name
    
    logger.info(f"{'='*60}")
    logger.info(f"EXECUTING: {process.name}")
    logger.info(f"Mandatory: {process.is_mandatory}")
    logger.info(f"Steps: {len(process.steps)}")
    logger.info(f"Output Directory: {process_dir}")
    logger.info(f"{'='*60}")
    
    # Reset UI state for this process
    reset_process_ui_state(webui_manager)
    
    # Create isolated controller for this process
    controller = CustomController(ask_assistant_callback=ask_callback)
    try:
        await controller.setup_mcp_client(None)
    except Exception as e:
        logger.warning(f"Failed to setup MCP client: {e}")
    
    controller.available_file_paths = available_file_paths

    # Prepare sensitive data vault
    resolved_sensitive_data = dict(sensitive_data or {})
    try:
        import re
        from src.utils.vault import vault

        vault_matches = re.findall(r'@vault\.([a-zA-Z0-9_]+)', task_str)
        for vault_key in vault_matches:
            creds = vault.get_credentials(vault_key)
            if not creds:
                logger.warning(f"Vault entry not found: {vault_key}")
                continue

            replacement_parts = []
            for field_name, value in creds.items():
                # CRITICAL: browser-use library expects <secret>KEY</secret> format
                # NOT {{KEY}} format - the library's _replace_sensitive_data looks for XML tags
                key_name = f"{vault_key.upper()}_{field_name.upper()}"
                placeholder_for_task = f"<secret>{key_name}</secret>"
                replacement_parts.append(f"{field_name.upper()}: {placeholder_for_task}")
                resolved_sensitive_data[key_name] = value

            replacement_text = ", ".join(replacement_parts)
            task_str = task_str.replace(f'@vault.{vault_key}', replacement_text)
            process.steps = [step.replace(f'@vault.{vault_key}', replacement_text) for step in process.steps]
    except ImportError:
        logger.debug("Vault module not available; skipping credential injection")
    except Exception as vault_err:
        logger.warning(f"Failed to resolve vault credentials: {vault_err}")

    # Set sensitive data on controller
    if resolved_sensitive_data:
        controller.sensitive_data = resolved_sensitive_data
    
    # File paths within process directory
    history_dir = os.path.join(agent_history_path, task_identifier)
    os.makedirs(history_dir, exist_ok=True)

    file_base = sanitize_filename(task_identifier.replace('/', '_')) or process_name_sanitized
    set_run_id(file_base)
    history_file = os.path.join(history_dir, f"{file_base}.json")
    gif_path = os.path.join(history_dir, f"{file_base}.gif")
    playwright_script_path = os.path.join(history_dir, f"{file_base}_playwright.py")
    docx_path = None  # Will be set when report is generated
    
    # Track if process completed via done()
    process_completed = False
    process_completion_lock = asyncio.Lock()
    
    async def done_callback_wrapper(history: AgentHistoryList):
        """Wrapper to track completion and stop agent."""
        nonlocal process_completed
        async with process_completion_lock:
            if not process_completed:
                process_completed = True
                logger.info(f"Process '{process.name}' signaled completion via done()")
                
                # Stop the agent immediately
                if webui_manager and webui_manager.bu_agent:
                    webui_manager.bu_agent.stop()
                
                # Call original done callback if provided
                if done_callback:
                    try:
                        await done_callback(history)
                    except Exception as e:
                        logger.warning(f"Error in done callback: {e}")
    
    try:
        # Create isolated agent for this process
        agent_kwargs = {
            'task': task_str,
            'llm': llm,
            'available_file_paths': available_file_paths,
            'include_attributes': [
                'title', 'type', 'name', 'role', 'aria-label', 'placeholder',
                'value', 'alt', 'aria-expanded', 'data-date-format'
            ],
            'browser': browser,
            'browser_context': browser_context,
            'controller': controller,
            'register_new_step_callback': step_callback,
            'register_done_callback': done_callback_wrapper,
            'use_vision': use_vision,
            'max_input_tokens': max_input_tokens,
            'max_actions_per_step': max_actions_per_step,
            'enable_memory': False,
        }

        if page_extraction_llm:
            agent_kwargs['page_extraction_llm'] = page_extraction_llm

        if planner_llm:
            agent_kwargs['planner_llm'] = planner_llm
            agent_kwargs['planner_interval'] = 3
            if planner_system_message:
                agent_kwargs['planner_system_message'] = planner_system_message
        else:
            agent_kwargs['planner_interval'] = 0

        if extend_system_prompt:
            agent_kwargs['extend_system_message'] = extend_system_prompt

        agent = BrowserUseAgent(**agent_kwargs)
        
        agent.state.agent_id = process_name_sanitized
        agent.settings.generate_gif = gif_path
        # agent.playwright_script_path = playwright_script_path  # DISABLED: We use LLM generation in orchestrator instead
        
        # Set webui_manager.bu_agent for callbacks
        if webui_manager:
            webui_manager.bu_agent = agent
            if hasattr(webui_manager, 'bu_chat_history'):
                process_header = f"\n{'='*60}\nProcess: {process.name}"
                if process.is_mandatory:
                    process_header += " (Mandatory)"
                process_header += f"\n{'='*60}\n{task_str}"
                webui_manager.bu_chat_history.append({"role": "user", "content": process_header})
        
        logger.info(f"Running process: {process.name}")
        
        # Execute process
        try:
            await agent.run(max_steps=max_steps)
        except ProcessHallucinationDetected as hallucination_error:
            logger.error(f"Hallucination detected in process '{process.name}': {hallucination_error}")
            raise
        except Exception as e:
            logger.error(f"Error during agent execution for '{process.name}': {e}", exc_info=True)
        
        # Get final history
        history: AgentHistoryList = agent.state.history

        # Determine success status aligned with existing callbacks
        success = False
        error_msg = None

        async with process_completion_lock:
            if process_completed:
                logger.info(f"Process '{process.name}' completed successfully")
                success = True
            else:
                steps_taken = len(history.history) if history and hasattr(history, 'history') else 0
                if steps_taken >= max_steps:
                    logger.warning(f"Process '{process.name}' hit max_steps limit")
                    error_msg = "Process reached maximum step limit"
                else:
                    logger.warning(f"Process '{process.name}' did not complete explicitly")
                    error_msg = "Process did not signal completion"

        if not process_completed and history and done_callback:
            try:
                await done_callback(history)
            except Exception as callback_err:
                logger.warning(f"Done callback failed for '{process.name}': {callback_err}")

        # Save history just like single-agent flow
        try:
            agent.save_history(history_file)
            logger.info(f"Saved history for process '{process.name}': {history_file}")
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")

        # Capture generated DOCX path (handled by _handle_done)
        if webui_manager and hasattr(webui_manager, 'bu_docx_path'):
            docx_path = webui_manager.bu_docx_path

        # Enrich history and generate Playwright script using existing utilities
        # For non-mandatory processes, include mandatory process steps for standalone execution
        mandatory_history_path = None
        try:
            from pathlib import Path
            from src.utils.strip_and_enrich_history import strip_and_enrich

            history_path = Path(history_file)
            if history_path.exists():
                logger.info("Enriching history with DOM context...")
                strip_and_enrich(history_path)

                # If this is a non-mandatory process and we have mandatory process history, merge them
                if not process.is_mandatory and orchestration_output_dir:
                    # Find mandatory process history in orchestration directory
                    orch_dir = Path(orchestration_output_dir)
                    for entry in orch_dir.iterdir():
                        if entry.is_dir() and 'mandatory' in entry.name.lower():
                            potential_history = list(entry.glob("*.json"))
                            if potential_history:
                                mandatory_history_path = potential_history[0]
                                logger.info(f"Found mandatory process history: {mandatory_history_path}")
                                break

                logger.info("Generating Playwright script via LLM...")
                try:
                    from src.utils.llm_script_generator import generate_script
                    from src.utils.config import SCRIPT_GEN_MODEL, SCRIPT_GEN_PROVIDER

                    # Use the dedicated script-generation model (fast coder model),
                    # NOT the agent's task model. Set SCRIPT_GEN_MODEL / SCRIPT_GEN_PROVIDER
                    # in your .env to override.
                    logger.info(f"Script gen model: {SCRIPT_GEN_PROVIDER}/{SCRIPT_GEN_MODEL}")

                    # For non-mandatory processes, pass the mandatory history so scripts include login
                    mand_hist_path = str(mandatory_history_path) if mandatory_history_path and os.path.exists(mandatory_history_path) else None
                    if mand_hist_path:
                        logger.info(f"Including mandatory history in script generation: {mand_hist_path}")
                    
                    # Generate the script using LLM (with optional mandatory history merge)
                    output_path, code = generate_script(
                        str(history_path), 
                        output_path=playwright_script_path,
                        model_name=SCRIPT_GEN_MODEL,
                        provider=SCRIPT_GEN_PROVIDER,
                        objective=task_str,
                        mandatory_history_path=mand_hist_path
                    )
                    logger.info(f"✅ LLM Playwright script saved: {output_path}")
                except Exception as llm_err:
                    logger.warning(f"LLM Script generation failed: {llm_err}. Falling back to heuristic...")
                    from src.utils.generate_playwright_script import generate_playwright_script
                    # Use history_path.parent as snapshot_dir fallback
                    snapshot_dir = history_path.parent
                    script_content = generate_playwright_script(history_path, snapshot_dir=snapshot_dir)
                    with open(playwright_script_path, 'w', encoding='utf-8') as script_file:
                        script_file.write(script_content)
                logger.info(f"Playwright script saved: {playwright_script_path}")
        except Exception as enrich_err:
            logger.warning(f"Could not enrich history or generate script: {enrich_err}", exc_info=True)
        
        # Mandatory process validation
        if process.is_mandatory and not success:
            logger.error(f"Mandatory process '{process.name}' failed")
            if not error_msg:
                error_msg = "Mandatory process did not complete successfully"
            raise ProcessFailed(error_msg)
        
        logger.info(f"Process '{process.name}' execution complete - Success: {success}")
        
        return ProcessResult(
            process=process,
            history=history,
            success=success,
            error=error_msg,
            docx_path=docx_path if docx_path and os.path.exists(docx_path) else None,
            gif_path=gif_path if os.path.exists(gif_path) else None,
            playwright_script_path=playwright_script_path if os.path.exists(playwright_script_path) else None,
            history_path=history_file if os.path.exists(history_file) else None,
        )
        
    except ProcessHallucinationDetected as e:
        logger.error(f"Hallucination detected in process '{process.name}': {e}")
        error_msg = f"Hallucination detected: {str(e)}"
        
        if 'agent' in locals():
            try:
                agent.save_history(history_file)
            except Exception as save_error:
                logger.warning(f"Failed to save history on hallucination: {save_error}")
        
        if process.is_mandatory:
            raise ProcessFailed(error_msg)
        
        return ProcessResult(
            process=process,
            history=agent.state.history if 'agent' in locals() else None,
            success=False,
            error=error_msg,
            gif_path=gif_path if os.path.exists(gif_path) else None,
            playwright_script_path=playwright_script_path if os.path.exists(playwright_script_path) else None,
            history_path=history_file if os.path.exists(history_file) else None,
        )
        
    except ProcessFailed as e:
        logger.error(f"Mandatory process failed: {e}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in process '{process.name}': {e}", exc_info=True)
        error_msg = str(e)
        
        if process.is_mandatory:
            raise ProcessFailed(error_msg)
        
        return ProcessResult(
            process=process,
            history=agent.state.history if 'agent' in locals() else None,
            success=False,
            error=error_msg,
            gif_path=gif_path if os.path.exists(gif_path) else None,
            playwright_script_path=playwright_script_path if os.path.exists(playwright_script_path) else None,
            history_path=history_file if os.path.exists(history_file) else None,
        )
    finally:
        # Cleanup controller
        if controller:
            try:
                await controller.close()
            except Exception as e:
                logger.debug(f"Error closing controller: {e}")


async def run_all_processes(
    processes: List[Process],
    browser,
    llm,
    orchestration_output_dir: Optional[str] = None,
    browser_context_config=None,
    **kwargs
) -> tuple[List[ProcessResult], str]:
    """
    Execute all processes with intelligent browser context management.
    
    KEY FEATURES:
    - Fresh browser context for MANDATORY processes only
    - Shared browser context for non-mandatory processes (continues from same state)
    - Continues execution even if non-mandatory processes fail
    - All output files stored in centralized orchestration directory
    
    Args:
        processes: List of processes to execute
        browser: Browser instance
        llm: Language model for agents
        orchestration_output_dir: Central output directory for all files
        browser_context_config: Browser context configuration
        **kwargs: Additional arguments passed to run_process
        
    Returns:
        Tuple of (results list, orchestration output directory)
    """
    agent_history_path = kwargs.pop('agent_history_path', './tmp/agent_history')
    
    # Create orchestration output directory
    if orchestration_output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        orchestration_output_dir = os.path.join(
            agent_history_path,
            f'orchestration_{timestamp}'
        )
    
    os.makedirs(orchestration_output_dir, exist_ok=True)
    
    import time
    start_time = time.time()
    
    logger.info(f"{'='*80}")
    logger.info(f"ORCHESTRATION START")
    logger.info(f"Total Processes: {len(processes)}")
    logger.info(f"Output Directory: {orchestration_output_dir}")
    logger.info(f"{'='*80}\n")
    
    results: List[ProcessResult] = []
    shared_browser_context = None
    shared_context_is_active = False
    
    for idx, process in enumerate(processes):
        logger.info(f"\n{'='*80}")
        logger.info(f"Process {idx + 1}/{len(processes)}: {process.name}")
        logger.info(f"Mandatory: {process.is_mandatory}")
        logger.info(f"{'='*80}\n")
        
        process_browser_context = None
        promote_context_to_shared = False
        
        try:
            if process.is_mandatory:
                # MANDATORY: Create fresh browser context
                if shared_browser_context and shared_context_is_active:
                    try:
                        await shared_browser_context.close()
                        shared_browser_context = None
                        shared_context_is_active = False
                        logger.info("Closed shared context for mandatory process")
                    except Exception as e:
                        logger.warning(f"Error closing shared context: {e}")
                
                # Create fresh context for mandatory process
                if browser:
                    try:
                        process_browser_context = await browser.new_context(
                            config=browser_context_config
                        )
                        logger.info("Created fresh browser context for mandatory process")
                    except Exception as e:
                        logger.warning(f"Failed to create fresh context: {e}")
                        process_browser_context = None
            else:
                # NON-MANDATORY: Reuse shared context
                if not shared_browser_context:
                    # Create shared context on first non-mandatory process
                    if browser:
                        try:
                            shared_browser_context = await browser.new_context(
                                config=browser_context_config
                            )
                            shared_context_is_active = True
                            logger.info("Created shared browser context for non-mandatory processes")
                        except Exception as e:
                            logger.warning(f"Failed to create shared context: {e}")
                            shared_browser_context = None
                
                process_browser_context = shared_browser_context
                if shared_browser_context:
                    logger.info("Using shared browser context (continues from previous state)")
            
            # Execute the process
            result = await run_process(
                process=process,
                browser=browser,
                browser_context=process_browser_context,
                llm=llm,
                orchestration_output_dir=orchestration_output_dir,
                agent_history_path=agent_history_path,
                **kwargs
            )
            results.append(result)
            
            if result.success:
                logger.info(f"✅ Process {idx + 1} PASSED: {process.name}")
                if process.is_mandatory and process_browser_context:
                    # Keep the successful mandatory context for subsequent processes
                    shared_browser_context = process_browser_context
                    shared_context_is_active = True
                    promote_context_to_shared = True
                    logger.info("Promoted mandatory process context to shared context for remaining processes")
            else:
                logger.error(f"❌ Process {idx + 1} FAILED: {process.name}")
                if result.error:
                    logger.error(f"   Error: {result.error}")
            
        except ProcessFailed as e:
            # Mandatory process failed - abort remaining processes
            logger.error(f"❌ MANDATORY PROCESS FAILED: {process.name}")
            logger.error(f"   Aborting remaining processes")
            results.append(
                ProcessResult(
                    process=process,
                    history=None,
                    success=False,
                    error=str(e)
                )
            )
            break
            
        except Exception as e:
            logger.error(f"❌ Error executing process '{process.name}': {e}", exc_info=True)
            results.append(
                ProcessResult(
                    process=process,
                    history=None,
                    success=False,
                    error=f"Execution error: {str(e)}"
                )
            )
            
            if process.is_mandatory:
                # Mandatory process failed
                logger.error("Aborting due to mandatory process failure")
                break
            else:
                # Non-mandatory process failed - continue with next
                logger.info("Continuing with next process despite failure")
        
        finally:
            # For mandatory processes, close the fresh context after completion unless promoted
            if (
                process.is_mandatory
                and process_browser_context
                and process_browser_context != shared_browser_context
                and not promote_context_to_shared
            ):
                try:
                    await process_browser_context.close()
                    logger.info("Closed fresh browser context for mandatory process")
                except Exception as e:
                    logger.warning(f"Error closing fresh context: {e}")
    
    # Cleanup shared context
    if shared_browser_context:
        try:
            await shared_browser_context.close()
            logger.info("Closed shared browser context at end of orchestration")
        except Exception as e:
            logger.warning(f"Error closing shared context: {e}")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"ORCHESTRATION COMPLETE")
    logger.info(f"{'='*80}")
    success_count = sum(1 for r in results if r.success)
    failed_count = len(results) - success_count
    logger.info(f"Executed: {len(results)}/{len(processes)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failed_count}")
    
    total_duration = time.time() - start_time
    logger.info(f"Total Duration: {total_duration:.2f}s")
    logger.info(f"Output Directory: {orchestration_output_dir}")
    logger.info(f"Output Directory: {orchestration_output_dir}")
    logger.info(f"{'='*80}\n")
    
    return results, orchestration_output_dir