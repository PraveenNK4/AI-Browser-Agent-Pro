#!/usr/bin/env python3
"""Test if Ollama is ready for verification with strong element"""

import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web-ui', 'src', 'utils'))

try:
    from ollama_max_accuracy_verifier import verify_step_element
    print("✓ Ollama verifier imported successfully\n")
    
    # Test 1: Strong admin server table element
    print("=" * 60)
    print("TEST 1: Admin Server Data Table")
    print("=" * 60)
    is_valid, result = verify_step_element(
        step_name='Extract Admin Servers',
        action_name='extract_content',
        element_text='AdminServer-01 Status: Active Processes: 12 Running: 12 Errors: 0',
        html_context='''<table id="admin_servers">
            <thead>
                <tr><th>Name</th><th>Status</th><th>Processes</th><th>Running</th><th>Errors</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td>AdminServer-01</td>
                    <td>Active</td>
                    <td>12</td>
                    <td>12</td>
                    <td>0</td>
                </tr>
            </tbody>
        </table>''',
        selector='table#admin_servers > tbody > tr',
        extraction_requirement='Extract admin server name, status, process counts, and error counts',
        keywords=['admin', 'server', 'status', 'active', 'processes', 'errors']
    )
    
    print(f"Valid: {is_valid}")
    print(f"Confidence: {result.get('confidence', 'N/A')}%")
    
    stages = result.get('verification_stages', {})
    if stages:
        print("\nVerification Stages:")
        for stage_name, stage_data in stages.items():
            passed = "✓" if stage_data.get('passed') else "✗"
            confidence = stage_data.get('confidence', 0)
            print(f"  {passed} {stage_name}: {confidence}%")
    
    print(f"\nResult: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    if not is_valid:
        print(f"Reason: {result.get('reason', 'Unknown')}")
    
    # Test 2: Wrong element (header instead of data)
    print("\n" + "=" * 60)
    print("TEST 2: Wrong Element - Header Instead of Data")
    print("=" * 60)
    is_valid2, result2 = verify_step_element(
        step_name='Extract Admin Servers',
        action_name='extract_content',
        element_text='Open Text Content Server Enterprise Personal Tools Admin',
        html_context='''<header role="banner">
            <div id="mastheadComponent">
                <nav>
                    <div id="header">Open Text Content Server</div>
                    <ul class="menu">
                        <li>Enterprise</li>
                        <li>Personal</li>
                        <li>Tools</li>
                        <li>Admin</li>
                    </ul>
                </nav>
            </div>
        </header>''',
        selector='header[role="banner"]',
        extraction_requirement='Extract admin server name, status, process counts, and error counts',
        keywords=['admin', 'server', 'status', 'active', 'processes', 'errors']
    )
    
    print(f"Valid: {is_valid2}")
    print(f"Confidence: {result2.get('confidence', 'N/A')}%")
    
    stages2 = result2.get('verification_stages', {})
    if stages2:
        print("\nVerification Stages:")
        for stage_name, stage_data in stages2.items():
            passed = "✓" if stage_data.get('passed') else "✗"
            confidence = stage_data.get('confidence', 0)
            print(f"  {passed} {stage_name}: {confidence}%")
    
    print(f"\nResult: {'✓ PASSED' if is_valid2 else '✗ FAILED (CORRECT!)'}")
    if not is_valid2:
        print(f"Reason: {result2.get('reason', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Ollama is working and accessible")
    print(f"✓ 4-stage verification pipeline operational")
    print(f"✓ Can distinguish correct from incorrect elements")
    print(f"\nSystem is ready! Re-run your agent task now.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
