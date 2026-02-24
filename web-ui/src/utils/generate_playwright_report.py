"""
Generate DOCX report from Playwright script execution with screenshots and step summaries.
"""

import json
from pathlib import Path
from typing import Optional


def generate_playwright_report(
    history_json_path: Path,
    screenshots_dir: Optional[Path] = None,
    execution_log_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Generate DOCX report from Playwright execution history using the script template.
    
    Args:
        history_json_path: Path to agent history JSON
        screenshots_dir: Directory containing numbered screenshots (0.png, 1.png, etc.)
        execution_log_path: Optional path to execution log JSON with step details
    
    Returns:
        Path to generated DOCX report or None if generation failed
    """
    try:
        from src.utils.report_templates import generate_script_report

        with history_json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_steps = data.get('history', []) if isinstance(data, dict) else data
        
        # Build the steps list for the report
        steps = []
        for step_idx, step in enumerate(raw_steps, 1):
            if not isinstance(step, dict):
                continue
            
            model_output = step.get('model_output', {})
            actions = model_output.get('action', [])
            
            if not actions:
                continue
            
            for action in actions:
                if not isinstance(action, dict) or not action:
                    continue
                action_name = list(action.keys())[0]
                
                output_text = ""
                result = step.get('result', {})
                if isinstance(result, dict):
                    output_text = result.get('extracted_content', '') or result.get('url', '') or ''
                elif isinstance(result, list) and result:
                    first_r = result[0] if isinstance(result[0], dict) else {}
                    output_text = first_r.get('extracted_content', '') or first_r.get('url', '') or str(result[0])[:200]
                
                steps.append({
                    "action": action_name,
                    "output": output_text[:200] if output_text else "N/A",
                })
        
        report_path = history_json_path.parent / f"{history_json_path.stem}_playwright_report.docx"
        
        result = generate_script_report(
            script_name=f"{history_json_path.stem}.py",
            steps=steps,
            screenshots_dir=str(screenshots_dir) if screenshots_dir else None,
            output_path=str(report_path),
            status="Execution Complete",
        )
        
        return Path(result) if result else None
    
    except Exception as e:
        print(f"Error generating Playwright report: {e}")
        return None
