import sys
import os

# Add project root to path (mimicking _FIXED.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path: sys.path.append(project_root)

print(f"Project Root: {project_root}")
print(f"Current Dir: {current_dir}")

try:
    from src.utils.vault import vault
    print("Vault imported successfully.")
except ImportError as e:
    print(f"Error importing vault: {e}")
    sys.exit(1)

def test_get_secret(key):
    print(f"Testing retrieval for key: {key}")
    try:
        if vault:
            creds = vault.get_credentials(key)
            if creds:
                print(f"  [SUCCESS] Found credentials for '{key}'")
                print(f"  Keys available: {list(creds.keys())}")
                if "username" in creds:
                    print(f"  Username: {creds['username']}")
                else:
                    print(f"  [WARNING] 'username' field missing in credentials.")
            else:
                print(f"  [FAILURE] Vault returned None for '{key}'")
        else:
            print("  [ERROR] Vault object is None.")
    except Exception as e:
        print(f"  [EXCEPTION] Error retrieving secret: {e}")

if __name__ == "__main__":
    if vault and hasattr(vault, 'secure_vault'):
        print(f"Vault Backend: {type(vault.secure_vault)}")
    
    test_get_secret("otcs")
