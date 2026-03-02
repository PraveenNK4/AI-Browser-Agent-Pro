import logging
from typing import Dict, Optional
import sys
import os

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from backend.credential_vault import CredentialVault as SecureVault
except ImportError:
    # Fallback to local import if path hacking fails (e.g. running from different dir)
    from web_ui.backend.credential_vault import CredentialVault as SecureVault

logger = logging.getLogger(__name__)

class LocalVault:
    """
    Wrapper around the secure encrypted backend.credential_vault module.
    Maintains the API expected by the rest of the application.
    """
    def __init__(self):
        try:
            self.secure_vault = SecureVault()
        except Exception as e:
            logger.error(f"Failed to initialize secure vault: {e}")
            self.secure_vault = None

    def get_credentials(self, key: str) -> Optional[Dict[str, str]]:
        """Retrieve credentials for a specific key (e.g., 'opentext')."""
        if not self.secure_vault:
            return None
        # Try exact match, then case-insensitive fallback for resilience
        creds = self.secure_vault.get_credential(key)
        if not creds:
            # Check all keys for case-insensitive match
            all_keys = self.list_keys()
            for k in all_keys:
                if k.lower() == key.lower():
                    return self.secure_vault.get_credential(k)
        return creds

    def list_keys(self) -> list[str]:
        """List all keys stored in the vault."""
        if not self.secure_vault:
            return []
        data = self.secure_vault._load_data()
        return list(data.keys())

# Global instance
vault = LocalVault()