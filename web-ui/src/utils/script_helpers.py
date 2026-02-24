"""
script_helpers.py
=================
Shared runtime helpers imported by every generated Playwright script.

Generated scripts should NOT redefine these functions — they import them:

    from src.utils.script_helpers import (
        get_secret, find_in_frames,
        resilient_fill, resilient_click, click_and_upload_file,
        maybe_login, resilient_extract_table, generate_report,
    )

All tuneable values (timeouts, selectors, vault prefix) come from
src.utils.config so they can be changed via environment variables.
"""

import io
import os
import re
import sys
import asyncio
from pathlib import Path

# Ensure src is importable regardless of working directory
_here = Path(__file__).resolve().parent
for _candidate in (_here.parent.parent, _here.parent.parent.parent):
    if (_candidate / "src").exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from src.utils.config import (
    VAULT_CREDENTIAL_PREFIX,
    LOGIN_USER_SELECTORS,
    LOGIN_PASS_SELECTORS,
    LOGIN_SUBMIT_SELECTORS,
    TIMEOUT_ELEMENT_VISIBLE_MS,
    TIMEOUT_FILL_MS,
    TIMEOUT_CLICK_MS,
    TIMEOUT_FILE_CHOOSER_MS,
    TIMEOUT_UPLOAD_CLICK_MS,
    TIMEOUT_UPLOAD_FALLBACK_MS,
    TIMEOUT_TABLE_WAIT_MS,
)
from src.utils.vault import vault


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

async def get_secret(key: str):
    """Look up a credential from the vault by composite key.

    Example: get_secret("OTCS_USERNAME") → vault.get_credentials("OTCS")["username"]
    """
    parts = key.split("_")
    v_key = parts[0] if len(parts) > 1 else key
    creds = vault.get_credentials(v_key) or vault.get_credentials(v_key.lower())
    if not creds:
        return None
    if any(k in key.upper() for k in ("USERNAME", "USER")):
        return creds.get("username")
    if any(k in key.upper() for k in ("PASSWORD", "PWD")):
        return creds.get("password")
    return None


# ---------------------------------------------------------------------------
# DOM interaction helpers
# ---------------------------------------------------------------------------

async def find_in_frames(page, selector: str, timeout: int = None):
    """Search for *selector* in the main page and all its iframes.

    Returns the first visible element found, or None.
    """
    visible_timeout = timeout or TIMEOUT_ELEMENT_VISIBLE_MS
    try:
        el = page.locator(selector).first
        if await el.count() > 0 and await el.is_visible(timeout=visible_timeout):
            return el
    except Exception:
        pass
    for frame in page.frames:
        try:
            el = frame.locator(selector).first
            if await el.count() > 0 and await el.is_visible(timeout=visible_timeout):
                return el
        except Exception:
            continue
    return None


async def resilient_fill(page, selector, value: str) -> bool:
    """Fill *value* into the first matching element from *selector* (str or list)."""
    if value is None:
        return False
    selectors = selector if isinstance(selector, list) else [selector]
    for sel in selectors:
        try:
            print(f"[*] Filling {sel}...")
            el = await find_in_frames(page, sel)
            if not el:
                await page.wait_for_selector(sel, timeout=TIMEOUT_FILL_MS)
                el = page.locator(sel).first
            await el.fill(str(value))
            await el.dispatch_event("input")
            await el.dispatch_event("change")
            return True
        except Exception as e:
            print(f"[-] Fill failed on {sel}: {e}")
            continue
    print(f"[-] Fill failed for all selectors: {selectors}")
    return False


async def resilient_click(page, selectors, desc: str = "element") -> bool:
    """Click the first visible match from *selectors* (str or list)."""
    if isinstance(selectors, str):
        selectors = [selectors]
    for sel in selectors:
        try:
            print(f"[*] Clicking {desc} via {sel}...")
            el = await find_in_frames(page, sel)
            if el:
                await el.click(timeout=TIMEOUT_CLICK_MS)
                return True
        except Exception as e:
            print(f"[-] Click failed on {sel}: {e}")
            continue
    print(f"[-] Click failed for all selectors ({desc}): {selectors}")
    return False


async def click_and_upload_file(page, target_name: str, file_path: str) -> bool:
    """Robustly trigger a file-chooser and upload *file_path*.

    Tries three strategies in order:
      1. Click a button/link whose text matches *target_name* exactly.
      2. Set files directly on a hidden <input type="file">.
      3. Click any element with that text and intercept the chooser.
    """
    print(f"[*] Starting robust upload for: {target_name}")
    if not os.path.exists(file_path):
        print(f"[-] File not found: {file_path}")
        return False
    # Strategy 1: native chooser via strict label match
    try:
        target = page.locator("a, button, [role='button']").filter(
            has_text=re.compile(f"^{re.escape(target_name)}$", re.I)
        ).first
        async with page.expect_file_chooser(timeout=TIMEOUT_FILE_CHOOSER_MS) as fc_info:
            await target.click(timeout=TIMEOUT_UPLOAD_CLICK_MS)
        (await fc_info.value).set_files(file_path)
        return True
    except Exception:
        pass
    # Strategy 2: hidden file input
    try:
        await page.locator('input[type="file"]').first.set_input_files(file_path)
        return True
    except Exception:
        pass
    # Strategy 3: global text search
    try:
        target = page.get_by_text(target_name, exact=True).first
        async with page.expect_file_chooser(timeout=TIMEOUT_UPLOAD_FALLBACK_MS) as fc_info:
            await target.click()
        (await fc_info.value).set_files(file_path)
        return True
    except Exception:
        pass
    print(f"[-] All upload strategies failed for: {target_name}")
    return False


# ---------------------------------------------------------------------------
# Login helper
# ---------------------------------------------------------------------------

async def maybe_login(page) -> None:
    """Detect and fill a login form if one is present.

    Uses LOGIN_*_SELECTORS and VAULT_CREDENTIAL_PREFIX from config —
    no hardcoded credentials or selectors.
    Does nothing if the page has no password field.
    """
    username = await get_secret(f"{VAULT_CREDENTIAL_PREFIX}_USERNAME")
    password = await get_secret(f"{VAULT_CREDENTIAL_PREFIX}_PASSWORD")
    if not username or not password:
        return

    # Only proceed if a password field is actually visible
    pass_el = None
    for sel in LOGIN_PASS_SELECTORS:
        pass_el = await find_in_frames(page, sel)
        if pass_el:
            break
    if not pass_el:
        return

    await resilient_fill(page, LOGIN_USER_SELECTORS, username)
    await resilient_fill(page, LOGIN_PASS_SELECTORS, password)
    await resilient_click(page, LOGIN_SUBMIT_SELECTORS, desc="login submit")

    # Wait for navigation after submit
    try:
        await page.wait_for_load_state("networkidle", timeout=TIMEOUT_TABLE_WAIT_MS)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Table extraction helper
# ---------------------------------------------------------------------------

async def resilient_extract_table(
    page,
    table_selector: str,
    header_keywords: list,
    skip_boilerplate: list = None,
) -> list:
    """Extract structured rows from a scoped table element.

    Args:
        page:             Playwright page object.
        table_selector:   CSS/XPath selector pointing to the <table> (or container).
        header_keywords:  Column names to extract (matched case-insensitively).
        skip_boilerplate: Row text patterns to skip (nav rows, headers, etc.).

    Returns:
        List of dicts, one per data row, keyed by header_keywords.

    Raises:
        RuntimeError: If table_selector is not found on page or in any iframe.
    """
    table = await find_in_frames(page, table_selector)
    if not table:
        raise RuntimeError(f"Table selector not found: {table_selector!r}")

    rows = await table.locator("tr").all()

    # If too few rows in main page, search iframes
    if len(rows) < 3:
        for frame in page.frames:
            try:
                f_table = frame.locator(table_selector).first
                if await f_table.count() > 0:
                    f_rows = await f_table.locator("tr").all()
                    if len(f_rows) > len(rows):
                        rows = f_rows
            except Exception:
                continue

    print(f"[*] Analysing {len(rows)} rows in {table_selector!r}...")

    # Resolve column indices from header row(s)
    col_map = {kw: -1 for kw in header_keywords}
    header_row_idx = -1  # track which row was the header
    for i in range(min(20, len(rows))):
        cells = await rows[i].locator("th, td").all()
        if len(cells) < len(header_keywords):
            continue
        h_texts = [(await c.inner_text() or "").strip() for c in cells]
        for kw in header_keywords:
            if col_map[kw] != -1:
                continue
            idx = next(
                (j for j, txt in enumerate(h_texts) if kw.lower() in txt.lower()), -1
            )
            if idx != -1:
                col_map[kw] = idx
        if all(v != -1 for v in col_map.values()):
            header_row_idx = i
            print(f"[*] Resolved columns: {col_map} (header row={i})")
            break

    _boilerplate = skip_boilerplate or [
        "Default", "Name", "Description", "Enterprise",
        "Personal", "Tools", "Status", "Header",
    ]
    results = []
    max_col = max(col_map.values()) if col_map else 0

    for row_idx, row in enumerate(rows):
        # Skip the header row (uses <td> instead of <th> in some apps)
        if row_idx == header_row_idx:
            continue
        cells = await row.locator("td").all()
        if len(cells) <= max_col:
            continue
        all_text = " ".join([(await c.inner_text() or "").strip() for c in cells])
        if any(b in all_text for b in _boilerplate):
            continue
        # Extra guard: skip rows where ALL cell values exactly match header keywords
        cell_texts = [(await cells[col_map[kw]].inner_text() or "").strip() for kw in header_keywords if col_map.get(kw, -1) != -1 and col_map[kw] < len(cells)]
        if cell_texts and all(t.lower() in [kw.lower() for kw in header_keywords] for t in cell_texts):
            continue

        row_data = {}
        valid = True
        for kw, idx in col_map.items():
            cell = cells[idx]
            # Prefer image title (status icons) over text
            img = cell.locator("img[title]").first
            text = (
                await img.get_attribute("title")
                if await img.count() > 0
                else (await cell.inner_text() or "").strip()
            )
            if not text or text == "null":
                valid = False
                break
            row_data[kw] = text

        if valid:
            results.append(row_data)

    print(f"[*] Extracted {len(results)} data rows.")
    return results


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------

async def generate_report(
    scenario: str,
    status: bool,
    screenshot=None,
    output: str = None,
    steps: list = None,
    screenshots_dir: str = None,
) -> None:
    """Print extracted data and save a .docx report using the script template."""
    from datetime import datetime

    print("\n" + "=" * 60)
    print(f"📋 EXTRACTED DATA: {scenario}")
    print("=" * 60)
    if output:
        print(output)
    print("=" * 60 + "\n")

    try:
        from src.utils.report_templates import generate_script_report

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        report_path = os.path.join(script_dir, f"report_{timestamp}.docx")

        # Build steps list if not provided
        if not steps:
            steps = [{"action": scenario, "output": output or "N/A"}]

        result = generate_script_report(
            script_name=f"{scenario}_{timestamp}",
            steps=steps,
            screenshots_dir=screenshots_dir,
            output_path=report_path,
            status="SUCCESS" if status else "FAILED",
            captured_outputs=output,
        )
        if result:
            print(f"[+] Report saved: {result}")
        else:
            # Fallback: basic docx if template fails
            from docx import Document
            doc = Document()
            doc.add_heading(f"Report: {scenario}", 0)
            doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            doc.add_paragraph(f"Status: {'SUCCESS' if status else 'FAILED'}")
            if output:
                doc.add_paragraph(f"Output:\n{output}")
            if screenshot:
                doc.add_picture(io.BytesIO(screenshot))
            doc.save(report_path)
            print(f"[+] Report saved (fallback): {report_path}")
    except Exception as e:
        print(f"[-] Report generation failed: {e}")

