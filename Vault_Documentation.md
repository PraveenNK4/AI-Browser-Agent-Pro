# Credential Vault: Documentation & Usage Guide

The **Credential Vault** is a secure, AES-encrypted storage system built into the Dynamic Scrapper. It is designed to ensure that passwords, API keys, and sensitive URLs are **never** hardcoded into python scripts or saved in plaintext `.env` files. 

When the AI agent needs to login to a website, it calls the Vault, decrypts the credentials in memory, and passes them to the browser—all while redacting them from the console logs.

---

## 1. How the Vault Works

The vault system is composed of four main components:
1. **`web-ui/backend/credential_vault.py`**: The core encryption engine and Command Line Interface (CLI) tool used by humans to manage credentials.
2. **`web-ui/src/utils/vault.py`**: A programmatic wrapper (`LocalVault`) used explicitly by the AI Agent scripts to seamlessly fetch credentials during an automation run.
3. **`.vault_key` (File)**: An auto-generated AES encryption key created by the `cryptography.fernet` library. **(WARNING: Never commit this file to Git or share it!)**
4. **`credentials.vault` (File)**: The actual encrypted JSON file that stores your vault data. Without the `.vault_key`, this file is completely unreadable.

---

## 2. Managing Credentials (CLI Usage)

You manage the vault using the backend Python script. Open your terminal, navigate to the `web-ui/backend` folder, and use the following commands:

### A. Adding a New Credential
You can add a credential profile (e.g., for "opentext" or "jira") interactively.
```bash
python credential_vault.py add
```
**Example Flow:**
1. **Credential name (key):** `otcs`
2. **Field name:** `USERNAME`
3. **USERNAME:** `admin_user`
4. **Field name:** `PASSWORD`
5. **PASSWORD:** `MySecretPassword123`
6. **Field name:** *(Leave blank and press Enter to finish)*

### B. Listing Available Credentials
To see all the profiles currently saved in the vault:
```bash
python credential_vault.py list
```
*Output:*
```text
🔐 Stored Credentials:
 - otcs
 - jira_test_account
```

### C. Viewing a Credential
To verify what fields are saved under a profile (passwords/tokens will be masked as `****` for safety):
```bash
python credential_vault.py get otcs
```

### D. Deleting a Credential
To remove a profile entirely from the encrypted file:
```bash
python credential_vault.py delete otcs
```

---

## 3. How the AI Agent Uses the Vault (Code Integration)

If you are writing a custom script or modifying the system, the AI interacts with the vault using the `LocalVault` wrapper.

### Python Example:
```python
from src.utils.vault import vault

# Retrieve the credential dictionary
creds = vault.get_credentials("otcs")

if creds:
    username = creds.get("USERNAME")
    password = creds.get("PASSWORD")
    print(f"Loaded credentials for user: {username}")
```

### AI Smart Detection:
You rarely need to write custom Python code for the vault. The `custom_controller.py` in the web UI is programmed to use the vault automatically. 
If the LLM generates a command to input text like: `{{VAULT_USERNAME}}`, the custom controller intercepts this, queries the Vault in the background, and seamlessly types the real password into the browser without the LLM ever knowing the actual plaintext password.

---

## 4. Security Warnings & Best Practices

> [!WARNING]
> **Backup your `.vault_key`!**
> If you delete the `.vault_key` file located in `web-ui/backend/`, your `credentials.vault` file becomes permanently locked and cannot be recovered. You will have to delete the vault file and re-enter all of your passwords.

> [!CAUTION]
> **Source Control Checklist**
> Ensure that both `.vault_key` and `credentials.vault` are added to your `.gitignore`. You do not want to accidentally push your encryption key or encrypted payload to a public repository.
