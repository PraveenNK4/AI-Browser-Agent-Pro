"""
Test script for Ollama Max Accuracy Verification system.
Demonstrates 4-stage verification for 100% accurate DOM element selection.
"""

import sys
import pathlib

# Add src to path
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from utils.ollama_max_accuracy_verifier import OllamaMaxAccuracyVerifier, verify_step_element
import json


def test_admin_servers_extraction():
    """Test extraction of admin servers table with Ollama verification."""
    
    print("\n" + "="*80)
    print("TEST 1: Admin Servers Table Extraction (Valid Data Element)")
    print("="*80 + "\n")
    
    # Example: Admin server data table
    element_text = """
    AdminServer01 | Active | 8 processes | 0 errors | 128GB
    AdminServer02 | Active | 7 processes | 0 errors | 256GB
    AdminServer03 | Inactive | 0 processes | 2 errors | 64GB
    AdminServer04 | Active | 9 processes | 1 error | 512GB
    """
    
    element_html = """
    <table class="admin-table">
        <thead>
            <tr>
                <th>Server Name</th>
                <th>Status</th>
                <th>Processes</th>
                <th>Errors</th>
                <th>Memory</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>AdminServer01</td>
                <td>Active</td>
                <td>8 processes</td>
                <td>0 errors</td>
                <td>128GB</td>
            </tr>
        </tbody>
    </table>
    """
    
    element_selector = "table.admin-table tbody"
    
    extraction_requirement = "Extract all admin servers with their status, process count, error count, and memory"
    expected_keywords = ["server", "admin", "status", "active", "processes", "errors", "memory", "name"]
    
    is_valid, result = verify_step_element(
        step_name="Extract Admin Servers",
        step_action="extract_table",
        element_text=element_text,
        element_html=element_html,
        element_selector=element_selector,
        extraction_requirement=extraction_requirement,
        expected_keywords=expected_keywords
    )
    
    print(f"\nRESULT: {'✓ VALID' if is_valid else '✗ INVALID'}")
    print(f"Final Confidence: {result.get('confidence', 0):.0f}%")
    print(f"Element Hash: {result.get('element_hash', 'unknown')}")
    print(f"\nReason:\n{result.get('reason', 'No reason provided')}")
    
    if not is_valid:
        print(f"\nRecommendations:\n{result.get('recommendations', 'No recommendations')}")
    
    return is_valid


def test_navigation_menu_rejection():
    """Test that navigation menu is properly rejected."""
    
    print("\n" + "="*80)
    print("TEST 2: Navigation Menu (Should be REJECTED)")
    print("="*80 + "\n")
    
    # Example: Navigation menu (should be rejected)
    element_text = "Home | Dashboard | Settings | Admin | Logout | No items to display"
    
    element_html = """
    <nav role="navigation">
        <ul>
            <li><a href="/">Home</a></li>
            <li><a href="/dashboard">Dashboard</a></li>
            <li><a href="/settings">Settings</a></li>
            <li><a href="/admin">Admin</a></li>
            <li><a href="/logout">Logout</a></li>
        </ul>
        <p>No items to display</p>
    </nav>
    """
    
    element_selector = "nav[role='navigation']"
    
    extraction_requirement = "Extract admin servers data"
    expected_keywords = ["server", "admin", "status", "active", "processes"]
    
    is_valid, result = verify_step_element(
        step_name="Extract Admin Servers",
        step_action="extract_table",
        element_text=element_text,
        element_html=element_html,
        element_selector=element_selector,
        extraction_requirement=extraction_requirement,
        expected_keywords=expected_keywords
    )
    
    print(f"\nRESULT: {'✓ VALID' if is_valid else '✗ INVALID (Expected!)'}")
    print(f"Final Confidence: {result.get('confidence', 0):.0f}%")
    
    print(f"\nReason:\n{result.get('reason', 'No reason provided')}")
    
    if not is_valid:
        print(f"\nRecommendations:\n{result.get('recommendations', 'No recommendations')}")
    
    return not is_valid  # Test passes if properly rejected


def test_ambiguous_multiple_tables():
    """Test selection with multiple similar elements to find best match."""
    
    print("\n" + "="*80)
    print("TEST 3: Multiple Tables - Choose Best Match")
    print("="*80 + "\n")
    
    # Target element (best match)
    element_text = """
    AdminServer01 | Active | 10 | 0 errors
    AdminServer02 | Active | 8 | 0 errors
    AdminServer03 | Inactive | 0 | 2 errors
    """
    
    element_html = """
    <table class="admin-servers">
        <tr>
            <th>Server</th><th>Status</th><th>Processes</th><th>Errors</th>
        </tr>
        <tr>
            <td>AdminServer01</td><td>Active</td><td>10</td><td>0 errors</td>
        </tr>
    </table>
    """
    
    # Nearby elements for cross-validation
    nearby = [
        {"text": "Login | Password | Remember me | No items"},  # Navigation
        {"text": "2024-01-26 | System Health | Good | 99.9% uptime"},  # Status bar
    ]
    
    element_selector = "table.admin-servers"
    
    extraction_requirement = "Get all admin servers with status and error count"
    expected_keywords = ["admin", "server", "status", "processes", "errors", "active"]
    
    is_valid, result = verify_step_element(
        step_name="Extract Admin Servers",
        step_action="extract_table",
        element_text=element_text,
        element_html=element_html,
        element_selector=element_selector,
        extraction_requirement=extraction_requirement,
        expected_keywords=expected_keywords,
        nearby_elements=nearby
    )
    
    print(f"\nRESULT: {'✓ VALID' if is_valid else '✗ INVALID'}")
    print(f"Final Confidence: {result.get('confidence', 0):.0f}%")
    print(f"Element Hash: {result.get('element_hash', 'unknown')}")
    
    # Show verification stages
    stages = result.get('verification_stages', {})
    print(f"\nVerification Stages:")
    print(f"  1. Structure: {stages.get('structure', {}).get('confidence', 0):.0f}%")
    print(f"  2. Content: {stages.get('content', {}).get('confidence', 0):.0f}%")
    print(f"  3. Step-Specific: {stages.get('step_specific', {}).get('confidence', 0):.0f}%")
    print(f"  4. Cross-Validation: {stages.get('cross_validation', {}).get('confidence', 0):.0f}%")
    
    print(f"\nReason:\n{result.get('reason', 'No reason provided')}")
    
    return is_valid


def main():
    """Run all tests."""
    
    print("\n" + "="*80)
    print("OLLAMA MAX ACCURACY VERIFICATION TEST SUITE")
    print("="*80)
    print("\nModel: llama2.5-32b | Temperature: 0.1 (maximum accuracy)")
    print("\nThis test demonstrates 4-stage verification for 100% accurate DOM element selection:")
    print("  1. Structure Validation - Checks if element has proper data structure")
    print("  2. Content Matching - Verifies content matches extraction requirement")
    print("  3. Step-Specific Verification - Ensures element is appropriate for the step")
    print("  4. Cross-Validation - Compares with neighboring elements to confirm correctness")
    print("\nMinimum confidence threshold: 90% (all stages must pass)")
    
    try:
        # Run tests
        test1_pass = test_admin_servers_extraction()
        test2_pass = test_navigation_menu_rejection()
        test3_pass = test_ambiguous_multiple_tables()
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"\nTest 1 (Valid Data Element): {'✓ PASSED' if test1_pass else '✗ FAILED'}")
        print(f"Test 2 (Reject Navigation): {'✓ PASSED' if test2_pass else '✗ FAILED'}")
        print(f"Test 3 (Choose Best Match): {'✓ PASSED' if test3_pass else '✗ FAILED'}")
        
        total_pass = sum([test1_pass, test2_pass, test3_pass])
        print(f"\nOverall: {total_pass}/3 tests passed")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("\n1. Run on actual agent history:")
        print("   python src/utils/strip_and_enrich_history.py <path_to_history.json>")
        print("\n2. Check output JSON for verification data:")
        print("   - Each extract_content step will have 'dom_context.verification' field")
        print("   - Shows element selection confidence and reasoning")
        print("   - Tracks element hash for consistency across runs")
        print("\n3. Adjust min_confidence if needed (default: 90% for maximum strictness)")
        print("="*80 + "\n")
    
    except ConnectionError as e:
        print(f"\n✗ ERROR: {e}")
        print("\nPlease ensure Ollama is running:")
        print("  1. Pull the model: ollama pull llama2.5-32b")
        print("  2. Start Ollama: ollama serve")
        print("  3. Run this test again")


if __name__ == "__main__":
    main()
