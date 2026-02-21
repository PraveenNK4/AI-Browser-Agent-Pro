import asyncio
import sys
import os
from playwright.async_api import async_playwright

# Add path to find backend module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web-ui'))
from backend.credential_vault import CredentialVault

async def main():
    # Load credentials from vault
    vault = CredentialVault()
    creds = vault.get_credential('otcs')  # Infer key based on {{OTCS_USERNAME}}
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        # Step 1: Navigate to the URL
        await page.goto('http://otcsvm.otxlab.net:8080/OTCS/cs.exe')
        
        print(f"Extracted content from step 1: {page.url}")

        # Step 2: Input username and password, then click login button
        await page.locator('#otds_username').fill(creds['username'])
        await page.locator('#otds_password').fill(creds['password'])
        await page.locator('#loginbutton').click()
        
        print(f"Extracted content from step 2: {page.url}")

        # Step 3: Click on the 'Functions' menu
        await page.locator('a.functionMenuHotspotWS[id="x2000"]').click()

        print(f"Extracted content from step 3: {page.url}")

        # Step 4: Click on the 'Presentation' link
        await page.locator('#menuItem_Properties_2000').click()
        
        print(f"Extracted content from step 4: {page.url}")

        # Step 5: Extract the description from properties section
        description = await page.get_by_text("creating a test case", exact=False).inner_text()

        print(f"Description extracted: {description}")
    
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())