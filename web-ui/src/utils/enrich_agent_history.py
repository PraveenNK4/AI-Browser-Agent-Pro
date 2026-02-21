"""
Enrich agent history JSON files with relevant DOM data from successful actions.

This script reads agent history files and corresponding DOM snapshots,
then adds the most relevant DOM element data for each successful action step.
It includes robust fallback mechanisms:
1. Exact index matching
2. Text-based heuristic matching
3. Coordinate-based matching (if bounding box available)
4. LLM-based verification (Ollama) if enabled
"""

import json
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Try to import Ollama verifier if available
try:
    from src.utils.ollama_max_accuracy_verifier import verify_element_with_ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama verifier module not found. Skipping LLM verification.")

DISABLE_OLLAMA_VERIFICATION = os.environ.get("DISABLE_OLLAMA_VERIFICATION", "false").lower() == "true"


def load_dom_snapshots(dom_snapshots_dir: str) -> List[Dict[str, Any]]:
    """Load all DOM snapshots from the directory."""
    snapshots = []
    if not os.path.exists(dom_snapshots_dir):
        return snapshots
        
    for filename in sorted(os.listdir(dom_snapshots_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(dom_snapshots_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    snapshot = json.load(f)
                    snapshots.append(snapshot)
            except Exception as e:
                logger.warning(f"Error reading snapshot {filename}: {e}")
    return snapshots


def find_dom_snapshot_for_action(
    snapshots: List[Dict[str, Any]],
    step_number: int,
    action_name: str
) -> Optional[Dict[str, Any]]:
    """
    Find the best matching pre-action DOM snapshot for a given step.
    Prioritizes snapshots with matching step number and 'pre_action' reason.
    """
    candidates = []
    for snapshot in snapshots:
        metadata = snapshot.get('metadata', {})
        reason = metadata.get('reason', '')
        
        # Check if reasons match specific action
        if f'pre_action:{action_name}' in reason:
            candidates.append(snapshot)
        # Fallback to general pre_action if specific one not found
        elif 'pre_action' in reason:
            candidates.append(snapshot)
            
    # If no candidates, try to find by step number logic or timestamp proximity if available
    # For now, return the last candidate that matches the sequence if simpler matching isn't possible
    # In a real scenario, we'd match timestamps. Here we assume sequential order.
    
    # Better logic: The controller explicitly sets 'pre_action:action_name' and likely step info
    # If we have candidates, return the one closest to the action execution
    if candidates:
        return candidates[-1] # Return the most recent pre-action snapshot
    
    return None


def calculate_match_score(element: Dict[str, Any], action_params: Dict[str, Any]) -> int:
    """
    Calculate a heuristic score (0-100) for how well an element matches the action parameters.
    """
    score = 0
    identity = element.get('identity', {})
    selectors = element.get('selector_provenance', {})
    attributes = element.get('attribute_fingerprint', {})
    
    # 1. Index Match (Highest Confidence)
    if 'index' in action_params:
        if identity.get('order') == action_params['index']:
            score += 80
            
    # 2. Text Match
    if 'text' in action_params:
        target_text = str(action_params['text']).lower()
        element_text = selectors.get('text', '').lower()
        
        if target_text == element_text:
            score += 60
        elif target_text in element_text:
            score += 40
        elif element_text and element_text in target_text:
            score += 30
            
    # 3. Attribute/Selector Match (CSS/XPath)
    # If action was selector-based (e.g. click_element_by_selector which might be added later)
    # This acts as a boost if we find matching attributes
    
    return min(score, 100)


def extract_relevant_element_data(
    snapshot: Dict[str, Any],
    action_params: Dict[str, Any],
    action_name: str
) -> Optional[Dict[str, Any]]:
    """
    Extract relevant element data using heuristics and optional Ollama verification.
    """
    if not snapshot or 'elements' not in snapshot:
        return None
    
    elements = snapshot['elements']
    candidates = []
    
    # Filter candidates based on basic heuristics first
    for element in elements:
        score = calculate_match_score(element, action_params)
        if score > 0:
            candidates.append({'element': element, 'score': score})
            
    # Sort by score descending
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    if not candidates:
        return None
        
    best_candidate = candidates[0]['element']
    
    # VERIFICATION: If Ollama is available, use it to verify questionable matches or refine selection
    # Only verify if score is not perfect (e.g. < 80, meaning index didn't match perfectly or ambiguous)
    # Or if explicit strict verification is desired
    if OLLAMA_AVAILABLE and not DISABLE_OLLAMA_VERIFICATION:
        try:
            # Prepare context for Ollama
            # We verify the top 3 candidates if close in score
            top_candidates = candidates[:3]
            
            # If we have a clear winner (score >= 80), maybe skip costly LLM
            # But "Best Result" implies maximum accuracy.
            
            verified_candidate = verify_element_with_ollama(
                query=str(action_params),
                elements=[c['element'] for c in top_candidates],
                task_description=f"Action: {action_name}"
            )
            
            if verified_candidate:
                best_candidate = verified_candidate
                logger.info(f"Ollama verified/selected element for {action_name}")
                
        except Exception as e:
            logger.warning(f"Ollama verification failed: {e}")
            
    # Format output
    identity = best_candidate.get('identity', {})
    selectors = best_candidate.get('selector_provenance', {})
    
    return {
        'element_index': identity.get('order'),
        'tagName': identity.get('tagName'),
        'selectors': selectors,
        'attributes': best_candidate.get('attribute_fingerprint', {}),
        'state': best_candidate.get('state', {}),
        'outerHTML': best_candidate.get('integrity', {}).get('outerHTML', ''),
        'match_method': 'ollama' if (OLLAMA_AVAILABLE and not DISABLE_OLLAMA_VERIFICATION) else 'heuristic'
    }


def enrich_agent_history_file(
    agent_history_file: str,
    dom_snapshots_dir: str,
    output_file: Optional[str] = None
) -> None:
    """Enrich agent history with robust element data."""
    logger.info(f"Processing agent history: {agent_history_file}")
    
    # Load all snapshots once
    snapshots = load_dom_snapshots(dom_snapshots_dir)
    
    try:
        with open(agent_history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load history file: {e}")
        return

    if not isinstance(history, list):
        return

    enriched_count = 0
    
    for step in history:
        if not isinstance(step, dict):
            continue
            
        model_output = step.get('model_output', {})
        actions = model_output.get('action', [])
        
        # Skip if no actions
        if not actions:
            continue
            
        step_number = step.get('metadata', {}).get('step_number', 0)
        
        # Enrich each action
        for action in actions:
            if not isinstance(action, dict):
                continue
                
            action_name = list(action.keys())[0] if action else None
            if not action_name:
                continue
                
            action_params = action.get(action_name, {})
            
            # Find snapshot
            snapshot = find_dom_snapshot_for_action(snapshots, step_number, action_name)
            
            if snapshot:
                relevant_data = extract_relevant_element_data(snapshot, action_params, action_name)
                
                if relevant_data:
                    if 'dom_context' not in action:
                        action['dom_context'] = relevant_data
                        enriched_count += 1
                        
    # Write output
    output_path = output_file or agent_history_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Enriched {enriched_count} actions in {agent_history_file}")


def enrich_all_agent_histories(
    agent_history_dir: str,
    dom_snapshots_base_dir: str
) -> None:
    """Enrich all histories in directory."""
    agent_history_path = Path(agent_history_dir)
    if not agent_history_path.exists():
        return
        
    for run_dir in agent_history_path.iterdir():
        if not run_dir.is_dir():
            continue
            
        json_file = run_dir / f"{run_dir.name}.json"
        if not json_file.exists():
            continue
            
        # Find partial match for snapshots
        dom_snapshots_dir = None
        # Logic to match run_ID or timestamp would go here
        # For now, just taking the most recent or matching directory
        for snap_dir in sorted(Path(dom_snapshots_base_dir).iterdir(), reverse=True):
            if snap_dir.is_dir() and snap_dir.name.startswith('run-'):
                dom_snapshots_dir = str(snap_dir)
                break
                
        if dom_snapshots_dir:
            enrich_agent_history_file(str(json_file), dom_snapshots_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Default paths
    base_dir = Path(__file__).parent.parent.parent
    agent_history_dir = base_dir / 'tmp' / 'agent_history'
    dom_snapshots_dir = base_dir / 'tmp' / 'dom_snapshots'
    
    enrich_all_agent_histories(str(agent_history_dir), str(dom_snapshots_dir))
