
import os
import json
import sys
from cryptography.fernet import Fernet
import logging

# Ensure we can import from src if needed, though this is standalone
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VAULT_FILE = os.path.join(os.path.dirname(__file__), 'credentials.vault')
KEY_FILE = os.path.join(os.path.dirname(__file__), '.vault_key')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("vault")

class CredentialVault:
    def __init__(self):
        self._load_key()
        
    def _load_key(self):
        """Load or create the encryption key."""
        if os.path.exists(KEY_FILE):
            with open(KEY_FILE, 'rb') as f:
                self.key = f.read()
        else:
            logger.info("🔑 Generating new vault key...")
            self.key = Fernet.generate_key()
            with open(KEY_FILE, 'wb') as f:
                f.write(self.key)
            logger.info(f"✅ Key saved to {KEY_FILE} (Keep this secure!)")
            
        self.cipher = Fernet(self.key)

    def _load_data(self):
        """Load and decrypt vault data."""
        # print(f"[DEBUG-VAULT] Loading from: {VAULT_FILE}")
        if not os.path.exists(VAULT_FILE):
            print(f"[DEBUG-VAULT] File not found: {VAULT_FILE}")
            return {}
        
        try:
            with open(VAULT_FILE, 'rb') as f:
                encrypted_data = f.read()
            if not encrypted_data:
                return {}
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"❌ Failed to decrypt vault: {e}")
            return {}

    def _save_data(self, data):
        """Encrypt and save vault data."""
        json_data = json.dumps(data)
        encrypted_data = self.cipher.encrypt(json_data.encode())
        with open(VAULT_FILE, 'wb') as f:
            f.write(encrypted_data)

    def add_credential(self):
        """Interactive CLI to add credentials."""
        name = input("Credential name (key): ").strip()
        if not name:
            print("❌ Name cannot be empty.")
            return

        creds = {}
        print("Enter fields (Press Enter with empty field name to finish):")
        while True:
            field = input("Field name: ").strip()
            if not field:
                break
            value = input(f"{field}: ").strip()
            creds[field] = value
        
        if not creds:
            print("❌ No fields added.")
            return

        data = self._load_data()
        data[name] = creds
        self._save_data(data)
        print(f"✅ Stored credentials for: {name}")

    def get_credential(self, name):
        """Retrieve a credential dictionary."""
        data = self._load_data()
        return data.get(name)

    def list_credentials(self):
        """List all stored credential keys."""
        data = self._load_data()
        if not data:
            print("📭 Vault is empty.")
            return
        print("🔐 Stored Credentials:")
        for key in data.keys():
            print(f" - {key}")

    def delete_credential(self, name):
        """Delete a credential."""
        data = self._load_data()
        if name in data:
            del data[name]
            self._save_data(data)
            print(f"🗑️ Deleted credential: {name}")
        else:
            print(f"❌ Credential not found: {name}")

if __name__ == "__main__":
    vault = CredentialVault()
    
    if len(sys.argv) < 2:
        print("Usage: python credential_vault.py [add|list|get <name>|delete <name>]")
        sys.exit(1)
        
    cmd = sys.argv[1].lower()
    
    if cmd == "add":
        vault.add_credential()
    elif cmd == "list":
        vault.list_credentials()
    elif cmd == "get":
        if len(sys.argv) < 3:
            print("Usage: python credential_vault.py get <name>")
        else:
            name = sys.argv[2]
            creds = vault.get_credential(name)
            if creds:
                print(f"🔐 Credentials for {name}:")
                for k, v in creds.items():
                    # Mask password for display
                    val = "****" if "pass" in k.lower() or "token" in k.lower() else v
                    print(f"  {k}: {val}")
            else:
                print(f"❌ Not found: {name}")
    elif cmd == "delete":
        if len(sys.argv) < 3:
            print("Usage: python credential_vault.py delete <name>")
        else:
            vault.delete_credential(sys.argv[2])
    else:
        print(f"❌ Unknown command: {cmd}")
