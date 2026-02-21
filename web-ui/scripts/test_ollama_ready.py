#!/usr/bin/env python3
"""Test if Ollama is ready for verification"""

import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'utils'))

try:
    from ollama_max_accuracy_verifier import verify_step_element
    print("✓ Ollama verifier imported successfully")
    
    # Test verification
    is_valid, result = verify_step_element(
        step_name='Test Step',
        action_name='extract_content',
        element_text='Admin Server Status: Active',
        html_context='<div><table><tr><td>Admin Server Status: Active</td></tr></table></div>',
        selector='table > tr > td',
        extraction_requirement='Extract admin server information',
        keywords=['admin', 'status', 'server']
    )
    
    print(f"\n=== VERIFICATION RESULT ===")
    print(f"Valid: {is_valid}")
    print(f"Confidence: {result.get('confidence', 'N/A')}%")
    print(f"Method: {result.get('method', 'N/A')}")
    print(f"Timestamp: {result.get('timestamp', 'N/A')}")
    
    if is_valid:
        print("\n✓ Ollama verification PASSED - System ready!")
    else:
        print(f"\n⚠ Ollama verification FAILED")
        print(f"Reason: {result.get('reason', 'Unknown')}")
        
    # Show stages
    stages = result.get('verification_stages', {})
    if stages:
        print("\nVerification Stages:")
        for stage_name, stage_data in stages.items():
            passed = "✓" if stage_data.get('passed') else "✗"
            confidence = stage_data.get('confidence', 0)
            print(f"  {passed} {stage_name}: {confidence}%")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
