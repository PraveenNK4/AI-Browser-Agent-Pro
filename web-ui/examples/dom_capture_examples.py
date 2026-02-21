"""
Example: Using DOM Capture and History Enrichment

This example demonstrates:
1. How DOM is automatically captured before actions
2. How to enrich agent history with DOM data
3. How to use enriched data for replay
"""

import json
import asyncio
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_view_dom_snapshot():
    """Example 1: View a DOM snapshot captured before an action"""
    logger.info("=== Example 1: View DOM Snapshot ===")
    
    # Path to a DOM snapshot
    snapshot_path = "tmp/dom_snapshots/run-1769412566-c218e4d85b9846e79261899ccccacf70/12.json"
    
    if not Path(snapshot_path).exists():
        logger.warning(f"Snapshot not found: {snapshot_path}")
        logger.info("DOM snapshots are created automatically when actions are executed")
        return
    
    with open(snapshot_path, 'r') as f:
        snapshot = json.load(f)
    
    metadata = snapshot['metadata']
    logger.info(f"Snapshot captured: {metadata['reason']}")
    logger.info(f"Action params: {metadata.get('action_params', {})}")
    logger.info(f"Total elements: {len(snapshot['elements'])}")
    
    # Find a specific element (e.g., index 5)
    for element in snapshot['elements'][:5]:  # Show first 5 elements
        identity = element['identity']
        selectors = element['selector_provenance']
        state = element['state']
        
        logger.info(f"\nElement {identity['order']}:")
        logger.info(f"  Tag: {identity['tagName']}")
        logger.info(f"  Preferred selector: {selectors['preferred']}")
        logger.info(f"  Visible: {state['visible']}, Enabled: {state['enabled']}")


def example_2_enrich_history():
    """Example 2: Enrich agent history with DOM data"""
    logger.info("\n=== Example 2: Enrich Agent History ===")
    
    from src.utils.enrich_agent_history import enrich_agent_history_file
    
    # Paths
    agent_history = "tmp/agent_history/19709e01-539c-43c6-80c2-337df37299d1/19709e01-539c-43c6-80c2-337df37299d1.json"
    dom_snapshots = "tmp/dom_snapshots/run-1769412566-c218e4d85b9846e79261899ccccacf70"
    output = "tmp/enriched_example.json"
    
    if not Path(agent_history).exists():
        logger.warning(f"Agent history not found: {agent_history}")
        logger.info("Run an agent first to generate history")
        return
    
    if not Path(dom_snapshots).exists():
        logger.warning(f"DOM snapshots not found: {dom_snapshots}")
        logger.info("DOM snapshots are created automatically during agent runs")
        return
    
    # Enrich the history
    logger.info("Enriching agent history...")
    try:
        enrich_agent_history_file(agent_history, dom_snapshots, output)
        logger.info(f"✓ Enriched history saved to: {output}")
        
        # Show example enriched data
        with open(output, 'r') as f:
            enriched = json.load(f)
        
        for step in enriched[:3]:  # Show first 3 steps
            if 'dom_context' in str(step):
                model_output = step.get('model_output', {})
                actions = model_output.get('action', [])
                
                for action in actions:
                    if 'dom_context' in action:
                        action_name = list(action.keys())[0]
                        dom_ctx = action.get(action_name, {}).get('dom_context', {})
                        
                        logger.info(f"\n✓ Enriched action: {action_name}")
                        logger.info(f"  Element: {dom_ctx.get('tagName')}")
                        logger.info(f"  Selector: {dom_ctx.get('selectors', {}).get('preferred')}")
                        break
    except Exception as e:
        logger.error(f"Error enriching history: {e}")


def example_3_replay_action():
    """Example 3: Replay an action using enriched DOM data"""
    logger.info("\n=== Example 3: Replay Action from DOM Data ===")
    
    # Load enriched history
    enriched_file = "tmp/enriched_example.json"
    
    if not Path(enriched_file).exists():
        logger.warning(f"Enriched file not found: {enriched_file}")
        logger.info("Run Example 2 first to create enriched history")
        return
    
    with open(enriched_file, 'r') as f:
        history = json.load(f)
    
    # Find a click action with DOM context
    for step in history:
        model_output = step.get('model_output', {})
        actions = model_output.get('action', [])
        
        for action in actions:
            if 'click_element_by_index' in action:
                click_data = action['click_element_by_index']
                
                if 'dom_context' in click_data:
                    dom_ctx = click_data['dom_context']
                    
                    logger.info("Found click action with DOM context:")
                    logger.info(f"  Element index: {dom_ctx.get('element_index')}")
                    logger.info(f"  Tag: {dom_ctx.get('tagName')}")
                    logger.info(f"  Text: {dom_ctx.get('selectors', {}).get('text')}")
                    
                    # Show how to replay
                    selector = dom_ctx.get('selectors', {}).get('preferred')
                    logger.info(f"\nTo replay this action:")
                    logger.info(f"  await page.click('{selector}')")
                    
                    return
    
    logger.info("No click actions with DOM context found")


def example_4_element_verification():
    """Example 4: Verify element state before action"""
    logger.info("\n=== Example 4: Element State Verification ===")
    
    enriched_file = "tmp/enriched_example.json"
    
    if not Path(enriched_file).exists():
        logger.warning(f"Enriched file not found: {enriched_file}")
        return
    
    with open(enriched_file, 'r') as f:
        history = json.load(f)
    
    # Check element states
    for step in history[:5]:  # First 5 steps
        model_output = step.get('model_output', {})
        actions = model_output.get('action', [])
        
        for action in actions:
            for action_name, action_data in action.items():
                if isinstance(action_data, dict) and 'dom_context' in action_data:
                    dom_ctx = action_data['dom_context']
                    state = dom_ctx.get('state', {})
                    
                    logger.info(f"\nAction: {action_name}")
                    logger.info(f"  Visible: {state.get('visible')}")
                    logger.info(f"  Enabled: {state.get('enabled')}")
                    logger.info(f"  Editable: {state.get('editable')}")
                    logger.info(f"  Bounding box: {state.get('bounding_box')}")
                    
                    # Verification logic example
                    if not state.get('visible'):
                        logger.warning("  ⚠️  Element was not visible!")
                    if not state.get('enabled'):
                        logger.warning("  ⚠️  Element was disabled!")


def main():
    """Run all examples"""
    logger.info("=" * 60)
    logger.info("DOM Capture and History Enrichment Examples")
    logger.info("=" * 60)
    
    try:
        # Example 1: View DOM snapshot
        example_1_view_dom_snapshot()
        
        # Example 2: Enrich history
        example_2_enrich_history()
        
        # Example 3: Replay action
        example_3_replay_action()
        
        # Example 4: Verify element state
        example_4_element_verification()
        
    except Exception as e:
        logger.error(f"Example error: {e}", exc_info=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("Examples complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
