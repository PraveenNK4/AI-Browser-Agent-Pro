"""
Validation Script: DOM Capture Implementation

This script validates that:
1. Pre-action DOM capture is working
2. DOM snapshots include action parameters
3. Agent history can be enriched
4. Enriched data contains expected fields
"""

import json
import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error"""
    pass


def validate_dom_snapshot_structure(snapshot_path: str) -> dict:
    """Validate a single DOM snapshot has correct structure"""
    logger.info(f"Validating: {snapshot_path}")
    
    with open(snapshot_path, 'r') as f:
        snapshot = json.load(f)
    
    # Check metadata
    if 'metadata' not in snapshot:
        raise ValidationError("Missing 'metadata' field")
    
    metadata = snapshot['metadata']
    required_fields = ['run_id', 'step_index', 'timestamp', 'reason', 'url', 'page_dom_hash']
    
    for field in required_fields:
        if field not in metadata:
            raise ValidationError(f"Missing metadata field: {field}")
    
    # Check if pre-action snapshot has action_params
    reason = metadata.get('reason', '')
    if 'pre_action:' in reason:
        if 'action_params' not in metadata:
            logger.warning(f"Pre-action snapshot missing action_params: {snapshot_path}")
        else:
            logger.info(f"✓ Has action_params: {metadata['action_params']}")
    
    # Check elements
    if 'elements' not in snapshot:
        raise ValidationError("Missing 'elements' field")
    
    if not isinstance(snapshot['elements'], list):
        raise ValidationError("'elements' must be a list")
    
    # Validate element structure
    if snapshot['elements']:
        element = snapshot['elements'][0]
        required_element_fields = [
            'identity', 'selector_provenance', 'attribute_fingerprint', 
            'state', 'integrity'
        ]
        
        for field in required_element_fields:
            if field not in element:
                raise ValidationError(f"Element missing field: {field}")
    
    logger.info(f"✓ Valid structure ({len(snapshot['elements'])} elements)")
    return snapshot


def validate_pre_action_capture(dom_snapshots_dir: str) -> bool:
    """Validate that pre-action snapshots are being created"""
    logger.info("\n=== Validating Pre-Action DOM Capture ===")
    
    if not Path(dom_snapshots_dir).exists():
        logger.error(f"DOM snapshots directory not found: {dom_snapshots_dir}")
        return False
    
    # Find all pre-action snapshots
    pre_action_count = 0
    post_action_count = 0
    
    for filename in os.listdir(dom_snapshots_dir):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(dom_snapshots_dir, filename)
        
        try:
            snapshot = validate_dom_snapshot_structure(filepath)
            reason = snapshot['metadata']['reason']
            
            if 'pre_action:' in reason:
                pre_action_count += 1
            elif 'post_action:' in reason:
                post_action_count += 1
        except ValidationError as e:
            logger.error(f"✗ Invalid snapshot {filename}: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Error reading {filename}: {e}")
            return False
    
    logger.info(f"\n✓ Found {pre_action_count} pre-action snapshots")
    logger.info(f"✓ Found {post_action_count} post-action snapshots")
    
    if pre_action_count == 0:
        logger.warning("⚠️  No pre-action snapshots found!")
        logger.info("   This might mean no actions have been executed yet.")
        return False
    
    return True


def validate_enrichment(enriched_file: str) -> bool:
    """Validate enriched agent history structure"""
    logger.info("\n=== Validating Enriched History ===")
    
    if not Path(enriched_file).exists():
        logger.error(f"Enriched file not found: {enriched_file}")
        return False
    
    with open(enriched_file, 'r') as f:
        history = json.load(f)
    
    if not isinstance(history, list):
        logger.error("History must be a list")
        return False
    
    enriched_actions = 0
    
    for step_num, step in enumerate(history):
        model_output = step.get('model_output', {})
        actions = model_output.get('action', [])
        
        for action in actions:
            if not isinstance(action, dict):
                continue
            
            # Check each action for dom_context
            for action_name, action_data in action.items():
                if isinstance(action_data, dict) and 'dom_context' in action_data:
                    enriched_actions += 1
                    
                    # Validate dom_context structure
                    dom_ctx = action_data['dom_context']
                    
                    if 'selectors' not in dom_ctx:
                        logger.warning(f"Step {step_num}: Missing 'selectors' in dom_context")
                    
                    if 'state' not in dom_ctx:
                        logger.warning(f"Step {step_num}: Missing 'state' in dom_context")
                    
                    logger.info(f"✓ Step {step_num}: {action_name} has dom_context")
    
    logger.info(f"\n✓ Found {enriched_actions} enriched actions")
    
    if enriched_actions == 0:
        logger.warning("⚠️  No enriched actions found!")
        logger.info("   Make sure to run the enrichment script first.")
        return False
    
    return True


def validate_element_data_quality(enriched_file: str) -> bool:
    """Validate quality of enriched element data"""
    logger.info("\n=== Validating Element Data Quality ===")
    
    if not Path(enriched_file).exists():
        logger.error(f"Enriched file not found: {enriched_file}")
        return False
    
    with open(enriched_file, 'r') as f:
        history = json.load(f)
    
    quality_checks = {
        'has_preferred_selector': 0,
        'has_multiple_selectors': 0,
        'has_state_info': 0,
        'has_bounding_box': 0,
        'has_outerHTML': 0,
    }
    
    total_enriched = 0
    
    for step in history:
        model_output = step.get('model_output', {})
        actions = model_output.get('action', [])
        
        for action in actions:
            for action_name, action_data in action.items():
                if not isinstance(action_data, dict) or 'dom_context' not in action_data:
                    continue
                
                total_enriched += 1
                dom_ctx = action_data['dom_context']
                
                # Check selector quality
                selectors = dom_ctx.get('selectors', {})
                if selectors.get('preferred'):
                    quality_checks['has_preferred_selector'] += 1
                
                if len([s for s in selectors.values() if s]) >= 2:
                    quality_checks['has_multiple_selectors'] += 1
                
                # Check state info
                state = dom_ctx.get('state', {})
                if 'visible' in state and 'enabled' in state:
                    quality_checks['has_state_info'] += 1
                
                if state.get('bounding_box'):
                    quality_checks['has_bounding_box'] += 1
                
                # Check outerHTML
                if dom_ctx.get('outerHTML'):
                    quality_checks['has_outerHTML'] += 1
    
    if total_enriched == 0:
        logger.warning("No enriched actions to check")
        return False
    
    # Report quality metrics
    logger.info(f"\nQuality Metrics (out of {total_enriched} enriched actions):")
    for check, count in quality_checks.items():
        percentage = (count / total_enriched) * 100
        status = "✓" if percentage >= 80 else "⚠️"
        logger.info(f"  {status} {check}: {count} ({percentage:.1f}%)")
    
    # Overall quality score
    avg_quality = sum(quality_checks.values()) / (len(quality_checks) * total_enriched) * 100
    logger.info(f"\nOverall Quality Score: {avg_quality:.1f}%")
    
    return avg_quality >= 70  # 70% threshold


def main():
    """Run all validations"""
    logger.info("=" * 60)
    logger.info("DOM Capture Implementation Validation")
    logger.info("=" * 60)
    
    # Configuration
    base_dir = Path(__file__).parent.parent.parent
    dom_snapshots_dir = base_dir / 'tmp' / 'dom_snapshots'
    agent_history_dir = base_dir / 'tmp' / 'agent_history'
    
    # Find most recent DOM snapshot directory
    latest_snapshot_dir = None
    if dom_snapshots_dir.exists():
        snapshot_dirs = [d for d in dom_snapshots_dir.iterdir() if d.is_dir()]
        if snapshot_dirs:
            latest_snapshot_dir = max(snapshot_dirs, key=lambda d: d.stat().st_mtime)
    
    # Find most recent agent history
    latest_history_file = None
    if agent_history_dir.exists():
        for run_dir in agent_history_dir.iterdir():
            if run_dir.is_dir():
                json_file = run_dir / f"{run_dir.name}.json"
                if json_file.exists():
                    latest_history_file = json_file
                    break
    
    # Run validations
    results = {}
    
    # 1. Validate pre-action capture
    if latest_snapshot_dir:
        results['pre_action_capture'] = validate_pre_action_capture(str(latest_snapshot_dir))
    else:
        logger.warning("No DOM snapshot directories found")
        results['pre_action_capture'] = False
    
    # 2. Validate enrichment (if enriched file exists)
    if latest_history_file:
        enriched_file = latest_history_file.parent / 'enriched.json'
        if enriched_file.exists():
            results['enrichment'] = validate_enrichment(str(enriched_file))
            results['data_quality'] = validate_element_data_quality(str(enriched_file))
        else:
            logger.info("\n=== Enriched file not found ===")
            logger.info("Run: python web-ui/src/utils/enrich_agent_history.py")
            results['enrichment'] = None
            results['data_quality'] = None
    else:
        logger.warning("No agent history files found")
        results['enrichment'] = None
        results['data_quality'] = None
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)
    
    for check, result in results.items():
        if result is None:
            status = "⊘ SKIPPED"
        elif result:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        
        logger.info(f"{status}: {check.replace('_', ' ').title()}")
    
    # Overall result
    passed_checks = sum(1 for r in results.values() if r is True)
    total_checks = sum(1 for r in results.values() if r is not None)
    
    if total_checks == 0:
        logger.warning("\n⚠️  No checks could be performed")
        logger.info("   Make sure to run the agent first to generate data")
    elif passed_checks == total_checks:
        logger.info("\n✓ All validations passed!")
    else:
        logger.warning(f"\n⚠️  {passed_checks}/{total_checks} validations passed")
    
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
