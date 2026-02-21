"""Report generation from process results."""
import os
import logging
from typing import List
from .process_models import ProcessResult

logger = logging.getLogger(__name__)


def generate_combined_playwright_script(
    results: List[ProcessResult],
    output_path: str
) -> str:
    """
    Generate a single Playwright script containing all processes.
    Each process is clearly separated with headers indicating pass/fail status.
    
    Args:
        results: List of process results
        output_path: Path to save combined script
        
    Returns:
        Path to generated script
    """
    try:
        scripts = []
        
        # Header
        scripts.append("# Combined Playwright Script - Multi-Process Execution")
        scripts.append("# Generated from process orchestration")
        scripts.append("# This script contains all processes executed in sequence")
        scripts.append("")
        
        for idx, result in enumerate(results, 1):
            status = "PASSED" if result.success else "FAILED"
            error_info = f" - Error: {result.error}" if result.error else ""
            mandatory_label = " (Mandatory)" if result.process.is_mandatory else ""
            
            scripts.append("")
            scripts.append(f"# {'='*60}")
            scripts.append(f"# Process {idx}: {result.process.name}{mandatory_label}")
            scripts.append(f"# Status: {status}{error_info}")
            scripts.append(f"# {'='*60}")
            scripts.append("")
            
            try:
                if result.history and result.playwright_script_path and os.path.exists(result.playwright_script_path):
                    # Read the generated script
                    try:
                        with open(result.playwright_script_path, 'r', encoding='utf-8') as f:
                            process_script = f.read()
                        scripts.append(process_script)
                        logger.info(f"  Added script for process {idx}: {result.process.name}")
                    except Exception as read_error:
                        logger.warning(f"  Failed to read script for process {idx}: {read_error}")
                        scripts.append(f"# Failed to read script: {str(read_error)}")
                elif result.history:
                    # Generate inline if not yet generated
                    try:
                        from src.agent.browser_use.browser_use_agent import generate_playwright_script_from_history
                        temp_path = output_path.replace(".py", f"_process_{idx}.py")
                        if generate_playwright_script_from_history(result.history, temp_path):
                            with open(temp_path, 'r', encoding='utf-8') as f:
                                scripts.append(f.read())
                            logger.info(f"  Generated and added script for process {idx}: {result.process.name}")
                        else:
                            scripts.append(f"# Failed to generate script for process {idx}")
                            logger.warning(f"  Script generation returned False for process {idx}")
                    except Exception as gen_error:
                        logger.warning(f"  Failed to generate script for process {idx}: {gen_error}")
                        scripts.append(f"# Failed to generate script: {str(gen_error)}")
                else:
                    scripts.append(f"# No history available for process {idx}")
                    logger.info(f"  No history for process {idx}: {result.process.name}")
            except Exception as process_error:
                logger.error(f"  Error processing script for process {idx}: {process_error}")
                scripts.append(f"# Error processing this process: {str(process_error)}")
        
        # Write combined script
        combined_content = "\n".join(scripts)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_content)
        
        logger.info(f"✅ Combined Playwright script saved: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Failed to generate combined Playwright script: {e}", exc_info=True)
        # Try to write error file
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# Error generating combined script\n# {str(e)}\n")
            logger.info(f"⚠️ Wrote error file to: {output_path}")
            return output_path
        except Exception as write_error:
            logger.error(f"Could not write error file: {write_error}")
            return None


def generate_process_summary(results: List[ProcessResult]) -> str:
    """
    Generate a markdown summary of all process executions.
    
    Args:
        results: List of process results
        
    Returns:
        Markdown formatted summary
    """
    try:
        lines = []
        lines.append("# Multi-Process Execution Summary")
        lines.append("")
        
        total = len(results)
        passed = sum(1 for r in results if r.success)
        failed = total - passed
        mandatory_passed = sum(1 for r in results if r.success and r.process.is_mandatory)
        mandatory_failed = sum(1 for r in results if not r.success and r.process.is_mandatory)
        
        lines.append(f"**Total Processes:** {total}")
        lines.append(f"**Passed:** {passed}")
        lines.append(f"**Failed:** {failed}")
        lines.append(f"**Mandatory Passed:** {mandatory_passed}")
        lines.append(f"**Mandatory Failed:** {mandatory_failed}")
        lines.append("")
        
        for idx, result in enumerate(results, 1):
            try:
                status_emoji = "✅" if result.success else "❌"
                mandatory_label = " (Mandatory)" if result.process.is_mandatory else ""
                lines.append(f"## {status_emoji} Process {idx}: {result.process.name}{mandatory_label}")
                lines.append("")
                lines.append(f"**Status:** {'Passed' if result.success else 'Failed'}")
                
                if result.error:
                    lines.append(f"**Error:** {result.error}")
                    lines.append("")
                
                if result.history:
                    try:
                        steps_count = len(result.history.history) if hasattr(result.history, 'history') else 0
                        lines.append(f"**Steps Executed:** {steps_count}")
                        
                        if hasattr(result.history, 'total_duration_seconds'):
                            try:
                                duration = result.history.total_duration_seconds()
                                lines.append(f"**Duration:** {duration:.2f}s")
                            except Exception:
                                pass
                        
                        if hasattr(result.history, 'total_input_tokens'):
                            try:
                                tokens = result.history.total_input_tokens()
                                lines.append(f"**Tokens Used:** {tokens}")
                            except Exception:
                                pass
                    except Exception as e:
                        logger.warning(f"Could not extract metrics for process {idx}: {e}")
                        lines.append(f"**Note:** History available but metrics could not be extracted")
                else:
                    lines.append(f"**Note:** No execution history available")
                
                # Add file paths
                lines.append("")
                lines.append("**Generated Files:**")
                if result.playwright_script_path and os.path.exists(result.playwright_script_path):
                    lines.append(f"- Playwright Script: `{os.path.basename(result.playwright_script_path)}`")
                if result.gif_path and os.path.exists(result.gif_path):
                    lines.append(f"- GIF Recording: `{os.path.basename(result.gif_path)}`")
                if result.docx_path and os.path.exists(result.docx_path):
                    lines.append(f"- DOCX Report: `{os.path.basename(result.docx_path)}`")
                
                lines.append("")
            except Exception as process_error:
                logger.error(f"Error generating summary for process {idx}: {process_error}")
                lines.append(f"## ⚠️ Process {idx}: Error generating summary")
                lines.append(f"**Error:** {str(process_error)}")
                lines.append("")
        
        summary = "\n".join(lines)
        logger.info(f"✅ Generated process summary ({len(lines)} lines)")
        return summary
        
    except Exception as e:
        logger.error(f"❌ Failed to generate process summary: {e}", exc_info=True)
        return f"# Multi-Process Execution Summary\n\n**Error:** Failed to generate summary - {str(e)}\n\n**Total Processes:** {len(results) if results else 0}"


def save_summary_report(summary: str, output_dir: str, filename: str = "orchestration_summary.md") -> str:
    """
    Save summary report to file.
    
    Args:
        summary: Summary content
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    try:
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"✅ Saved summary report: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ Failed to save summary report: {e}")
        return None


def save_orchestration_metadata(
    results: List[ProcessResult],
    output_dir: str,
    filename: str = "orchestration_metadata.json"
) -> str:
    """
    Save orchestration metadata (process info, results summary).
    
    Args:
        results: List of process results
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    import json
    try:
        metadata = {
            "total_processes": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "mandatory_count": sum(1 for r in results if r.process.is_mandatory),
            "processes": [
                {
                    "name": r.process.name,
                    "is_mandatory": r.process.is_mandatory,
                    "success": r.success,
                    "error": r.error,
                    "steps_count": len(r.process.steps),
                    "files": {
                        "history": r.history is not None,
                        "gif": os.path.exists(r.gif_path) if r.gif_path else False,
                        "script": os.path.exists(r.playwright_script_path) if r.playwright_script_path else False,
                        "docx": os.path.exists(r.docx_path) if r.docx_path else False,
                    }
                }
                for r in results
            ]
        }
        
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Saved orchestration metadata: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ Failed to save metadata: {e}")
        return None