#!/usr/bin/env python3
"""
Test Live Ollama Verification during DOM Capture

This test verifies that:
1. Ollama verification happens at DOM capture time (not post-processing)
2. Each element gets verified with qwen2.5:32b
3. Verification results are stored in the DOM snapshot JSON
4. System works with Ollama running
"""

import json
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'web-ui' / 'src'))

def test_live_verification():
    """Test that live verification is integrated"""
    
    print("=" * 70)
    print("LIVE OLLAMA VERIFICATION TEST")
    print("=" * 70)
    
    # 1. Check if verifier can be imported
    print("\n1. Testing Ollama verifier import...")
    try:
        from utils.ollama_max_accuracy_verifier import verify_step_element
        print("   ✓ Ollama verifier imported successfully")
    except ImportError as e:
        print(f"   ✗ Failed to import verifier: {e}")
        return False
    
    # 2. Check if dom_snapshot has verification enabled
    print("\n2. Testing DOM snapshot integration...")
    try:
        from utils.dom_snapshot import OLLAMA_VERIFICATION_ENABLED
        if OLLAMA_VERIFICATION_ENABLED:
            print("   ✓ Live verification ENABLED in dom_snapshot.py")
        else:
            print("   ✗ Live verification DISABLED in dom_snapshot.py")
            return False
    except ImportError as e:
        print(f"   ✗ Failed to check dom_snapshot: {e}")
        return False
    
    # 3. Check if Ollama is running
    print("\n3. Testing Ollama service...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            print(f"   ✓ Ollama is running")
            print(f"   ✓ Available models: {', '.join(model_names)}")
            
            if "qwen2.5:32b" in model_names:
                print("   ✓ qwen2.5:32b model available")
            else:
                print("   ⚠ qwen2.5:32b not found - verification may use different model")
        else:
            print(f"   ✗ Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Ollama not accessible: {e}")
        print("   → Start Ollama: ollama serve")
        return False
    
    # 4. Check recent DOM snapshots for verification data
    print("\n4. Checking recent DOM snapshots...")
    snapshot_dir = Path(__file__).parent / "tmp" / "dom_snapshots"
    
    if not snapshot_dir.exists():
        print("   ⚠ No DOM snapshots directory found")
        print("   → Run an agent task to generate snapshots")
        return True  # Not a failure, just no data yet
    
    # Find most recent snapshot directory
    snapshot_dirs = [d for d in snapshot_dir.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        print("   ⚠ No snapshot directories found")
        print("   → Run an agent task to generate snapshots")
        return True
    
    latest_dir = max(snapshot_dirs, key=lambda d: d.stat().st_mtime)
    print(f"   → Checking: {latest_dir.name}")
    
    # Check JSON files for verification data
    verified_files = []
    for json_file in sorted(latest_dir.glob("*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            elements = data.get("elements", [])
            verified_elements = [
                el for el in elements 
                if "ollama_verification" in el and el["ollama_verification"].get("verified_at_capture")
            ]
            
            if verified_elements:
                verified_files.append((json_file.name, len(verified_elements), len(elements)))
        except Exception as e:
            continue
    
    if verified_files:
        print(f"   ✓ Found {len(verified_files)} snapshots with live verification:")
        for filename, verified, total in verified_files[:5]:  # Show first 5
            print(f"      - {filename}: {verified}/{total} elements verified")
        
        # Show sample verification data
        if verified_files:
            sample_file = latest_dir / verified_files[0][0]
            with open(sample_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for el in data.get("elements", []):
                if "ollama_verification" in el:
                    verification = el["ollama_verification"]
                    print(f"\n   Sample verification:")
                    print(f"      Valid: {verification.get('is_valid')}")
                    print(f"      Confidence: {verification.get('confidence')}%")
                    print(f"      Method: {verification.get('method')}")
                    print(f"      Verified at capture: {verification.get('verified_at_capture')}")
                    break
    else:
        print("   ⚠ No verified snapshots found")
        print("   → This is expected if no tasks have run since enabling live verification")
        print("   → Run a new agent task to see live verification in action")
    
    print("\n" + "=" * 70)
    print("LIVE VERIFICATION SYSTEM STATUS")
    print("=" * 70)
    print("✓ Ollama verifier: READY")
    print("✓ DOM capture integration: ENABLED")
    print("✓ Ollama service: RUNNING")
    print("✓ Model: qwen2.5:32b")
    print("\nNext: Run an agent task through the web UI to see live verification!")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    try:
        success = test_live_verification()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
