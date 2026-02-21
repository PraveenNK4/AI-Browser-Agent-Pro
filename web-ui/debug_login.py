import asyncio
import os
import sys
from playwright.async_api import async_playwright

# Setup path to find 'src'
current_dir = os.getcwd()
if "web-ui" not in current_dir and os.path.exists(os.path.join(current_dir, "web-ui")):
    sys.path.append(os.path.join(current_dir, "web-ui"))
else:
    sys.path.append(current_dir)

from src.utils.vault import vault

async def get_secret(key):
    parts = key.split("_")
    v_key = parts[0] if len(parts) > 1 else key
    creds = vault.get_credentials(v_key) or vault.get_credentials(v_key.lower())
    if not creds: return None
    if any(k in key.upper() for k in ["USERNAME", "USER"]): return creds.get("username")
    if any(k in key.upper() for k in ["PASSWORD", "PWD"]): return creds.get("password")
    return None

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        url = "http://otcsvm.otxlab.net:8080/OTCS/cs.exe?func=ll&objtype=148&objaction=browse"
        print(f"[*] Navigating to {url}")
        await page.goto(url)
        
        await page.wait_for_selector("#otds_username")
        user = await get_secret("OTCS_USER")
        pwd = await get_secret("OTCS_PWD")
        
        print(f"[*] Typing credentials for {user}...")
        await page.fill("#otds_username", user)
        await page.fill('input[type="password"]', pwd)
        
        print("[*] Clicking sign in...")
        await page.click('button:has-text("Sign in"), #loginbutton')
        
        print("[*] Waiting for navigation...")
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
            print(f"[*] Final URL: {page.url}")
            await page.screenshot(path="debug_login.png")
            
            rows = await page.locator("tr").count()
            print(f"[*] Row count: {rows}")
        except Exception as e:
            print(f"[-] Wait failed: {e}")
            await page.screenshot(path="debug_login_fail.png")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
