"""
Integration Example: Using Comprehensive Element Capture with Agent

This shows how to integrate the comprehensive element capture
into your existing browser automation workflow.
"""

import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.comprehensive_element_capture import (
    ComprehensiveElementCapture,
    capture_and_save_element
)


async def example_1_capture_single_element():
    """Example 1: Capture data from a single element."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Capture Single Element")
    print("="*80)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Navigate to a test page
        await page.goto("https://www.google.com")
        await page.wait_for_load_state("networkidle")
        
        # Find the search input
        element = await page.query_selector('textarea[name="q"]')
        
        if element:
            # Capture comprehensive element data
            element_data = await capture_and_save_element(
                page, 
                element, 
                index=0,
                action_context="input_search_query"
            )
            
            print(f"\n✓ Captured {len(element_data.get('selectors', []))} selector candidates")
            print(f"✓ Recommended selector: {element_data['recommendedSelector']['selector']}")
        
        await browser.close()


async def example_2_capture_during_interaction():
    """Example 2: Capture elements during agent interaction."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Capture During Interaction")
    print("="*80)
    
    capturer = ComprehensiveElementCapture(output_dir=Path("tmp/element_data"))
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Navigate to a test page
        await page.goto("https://www.google.com")
        await page.wait_for_load_state("networkidle")
        
        # Simulate agent finding and clicking multiple elements
        elements_to_capture = [
            {'selector': 'textarea[name="q"]', 'action': 'input', 'index': 0},
            {'selector': 'input[name="btnK"]', 'action': 'click', 'index': 1},
        ]
        
        all_element_data = []
        
        for elem_info in elements_to_capture:
            element = await page.query_selector(elem_info['selector'])
            
            if element:
                # Capture BEFORE interaction
                element_data = await capturer.capture_element_data(
                    page,
                    element,
                    index=elem_info['index'],
                    action_context=f"pre_{elem_info['action']}"
                )
                
                # Save to JSON
                filename = f"element_{elem_info['action']}_idx{elem_info['index']}.json"
                await capturer.save_element_data(element_data, filename)
                
                all_element_data.append(element_data)
                
                print(f"\n✓ Captured: {elem_info['action']} on {element_data['tagName']}")
                print(f"  Best selector: {element_data['recommendedSelector']['selector']}")
        
        # Save combined data
        combined_file = capturer.output_dir / "all_elements_summary.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_element_data, f, indent=2)
        
        print(f"\n✓ Saved {len(all_element_data)} elements to: {combined_file}")
        
        await browser.close()


async def example_3_integrate_with_dom_snapshot():
    """Example 3: Integrate with existing DOM snapshot system."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Integrate with DOM Snapshots")
    print("="*80)
    
    # This shows how to enhance existing DOM snapshots with comprehensive element data
    
    capturer = ComprehensiveElementCapture()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        await page.goto("https://www.google.com")
        await page.wait_for_load_state("networkidle")
        
        # Get all interactive elements
        interactive_elements = await page.query_selector_all(
            'button, input, textarea, a, select, [role="button"]'
        )
        
        print(f"\nFound {len(interactive_elements)} interactive elements")
        print("Capturing comprehensive data for first 5...")
        
        enhanced_snapshot = {
            'metadata': {
                'url': page.url,
                'title': await page.title(),
                'captured_at': 'now',
                'element_count': len(interactive_elements)
            },
            'elements': []
        }
        
        for idx, element in enumerate(interactive_elements[:5]):
            element_data = await capturer.capture_element_data(
                page,
                element,
                index=idx,
                action_context="snapshot"
            )
            enhanced_snapshot['elements'].append(element_data)
            
            print(f"  {idx + 1}. {element_data['tagName']:10s} - {element_data['recommendedSelector']['selector'][:60]}")
        
        # Save enhanced snapshot
        snapshot_file = capturer.output_dir / "enhanced_snapshot.json"
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_snapshot, f, indent=2)
        
        print(f"\n✓ Saved enhanced snapshot to: {snapshot_file}")
        
        await browser.close()


async def example_4_selector_comparison():
    """Example 4: Compare selectors across page refreshes."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Selector Stability Analysis")
    print("="*80)
    
    capturer = ComprehensiveElementCapture()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Capture element data on first visit
        await page.goto("https://www.google.com")
        await page.wait_for_load_state("networkidle")
        
        element_v1 = await page.query_selector('textarea[name="q"]')
        if element_v1:
            data_v1 = await capturer.capture_element_data(page, element_v1, index=0)
        else:
            raise Exception("Element not found")
        
        print("\n📸 First capture:")
        print(f"  Element hash: {data_v1['elementHash']}")
        print(f"  Stable selectors (>70): {len([s for s in data_v1['selectors'] if s['stability'] > 70])}")
        
        # Refresh page and capture again
        await page.reload()
        await page.wait_for_load_state("networkidle")
        
        element_v2 = await page.query_selector('textarea[name="q"]')
        if element_v2:
            data_v2 = await capturer.capture_element_data(page, element_v2, index=0)
        else:
            raise Exception("Element not found after refresh")
        
        print("\n📸 After refresh:")
        print(f"  Element hash: {data_v2['elementHash']}")
        print(f"  Stable selectors (>70): {len([s for s in data_v2['selectors'] if s['stability'] > 70])}")
        
        # Compare
        if data_v1['elementHash'] == data_v2['elementHash']:
            print("\n✅ Element hash STABLE - same element across refreshes")
        else:
            print("\n⚠️  Element hash CHANGED - element may have changed")
        
        # Compare selectors
        selectors_v1 = {s['selector'] for s in data_v1['selectors']}
        selectors_v2 = {s['selector'] for s in data_v2['selectors']}
        
        common_selectors = selectors_v1 & selectors_v2
        print(f"\n🔄 Common selectors: {len(common_selectors)}/{len(selectors_v1)}")
        
        await browser.close()


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("COMPREHENSIVE ELEMENT CAPTURE - EXAMPLES")
    print("="*80)
    
    examples = [
        ("Single Element", example_1_capture_single_element),
        ("During Interaction", example_2_capture_during_interaction),
        ("With DOM Snapshots", example_3_integrate_with_dom_snapshot),
        ("Stability Analysis", example_4_selector_comparison),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples) + 1}. Run all")
    print(f"  0. Exit")
    
    try:
        choice = input("\nSelect example (0-5): ").strip()
        choice_num = int(choice)
        
        if choice_num == 0:
            print("Exiting...")
            return
        elif choice_num == len(examples) + 1:
            print("\nRunning all examples...\n")
            for name, example_func in examples:
                asyncio.run(example_func())
        elif 1 <= choice_num <= len(examples):
            name, example_func = examples[choice_num - 1]
            print(f"\nRunning: {name}\n")
            asyncio.run(example_func())
        else:
            print("Invalid choice")
    
    except ValueError:
        print("Invalid input")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")


if __name__ == "__main__":
    main()
