
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from src.utils.vault import vault

def test_vault():
    print(f"Current Working Directory: {os.getcwd()}")
    
    # Test 1: List raw names from secure vault
    if vault.secure_vault:
        print("[*] Secure vault connected successfully.")
        data = vault.secure_vault._load_data()
        print(f"[*] Raw keys in vault: {list(data.keys())}")
    else:
        print("[-] Secure vault NOT connected.")
        return

    # Test 2: Use wrapper API with different casings
    keys_to_test = ["otcs", "OTCS"]
    for key in keys_to_test:
        print(f"[*] Testing get_credentials for: '{key}'")
        creds = vault.get_credentials(key)
        if creds:
            print(f"  [+] Found! Fields: {list(creds.keys())}")
            # Check fields
            print(f"  [+] 'username' exists: {'username' in creds}")
            print(f"  [+] 'password' exists: {'password' in creds}")
        else:
            print(f"  [-] NOT found.")

if __name__ == "__main__":
    test_vault()
