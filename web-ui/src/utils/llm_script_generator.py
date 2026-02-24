"""
LLM Script Generator for Browser Agent
Uses an LLM (via llm_provider) to generate robust Playwright scripts from agent history.
Key Features:
- Redundancy Elimination (Smart Login)
- Context-Aware Code Generation
- Professional Report Integration
"""

import sys
import os
import json
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, Any

# Ensure we can import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.llm_provider import get_llm_model
from src.utils.config import (
    SCRIPT_GEN_TEMPERATURE, SCRIPT_GEN_NUM_CTX, SCRIPT_GEN_NUM_PREDICT,
    OOM_FALLBACK_MODELS, OOM_FALLBACK_CTX, OOM_RETRY_NUM_PREDICT,
    VAULT_CREDENTIAL_PREFIX,
    LOGIN_USER_SELECTORS, LOGIN_PASS_SELECTORS, LOGIN_SUBMIT_SELECTORS,
    TIMEOUT_FIND_IN_FRAMES_MS, TIMEOUT_ELEMENT_VISIBLE_MS,
    TIMEOUT_FILL_MS, TIMEOUT_CLICK_MS,
    TIMEOUT_FILE_CHOOSER_MS, TIMEOUT_UPLOAD_CLICK_MS, TIMEOUT_UPLOAD_FALLBACK_MS,
    TIMEOUT_TABLE_WAIT_MS, TIMEOUT_NETWORKIDLE_MS,
)
from langchain_core.messages import SystemMessage, HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _build_system_prompt() -> str:
    """Build the LLM system prompt. The LLM generates fully self-contained scripts —
    no external helper imports. All functions the script needs are written inline."""

    # Dynamic vault prefix discovery (avoid hardcoding "OTCS")
    from src.utils.vault import vault
    available_keys = vault.list_keys()
    
    # If a prefix is in environment, use it. 
    # Otherwise, if only one key exists, use it.
    # Otherwise, default to "vault_key" as a generic placeholder.
    if VAULT_CREDENTIAL_PREFIX and VAULT_CREDENTIAL_PREFIX != "OTCS":
        vault_prefix = VAULT_CREDENTIAL_PREFIX
    elif len(available_keys) == 1:
        vault_prefix = available_keys[0]
    else:
        # Fallback to config but allow the LLM to override based on context
        vault_prefix = VAULT_CREDENTIAL_PREFIX 

    # Serialize selector lists so the LLM can embed them

    # Serialize selector lists so the LLM can embed them
    user_sels_str   = json.dumps(LOGIN_USER_SELECTORS)
    pass_sels_str   = json.dumps(LOGIN_PASS_SELECTORS)
    submit_sels_str = json.dumps(LOGIN_SUBMIT_SELECTORS)

    # Minimal script header — includes path bootstrap for vault access
    script_header = f'''
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright

# Fix Windows console encoding for special characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Path bootstrap — keeps the script runnable from any working directory
# Ensures we can import from src.utils.vault
current_dir = os.getcwd()
if "web-ui" not in current_dir and os.path.exists(os.path.join(current_dir, "web-ui")):
    sys.path.append(os.path.join(current_dir, "web-ui"))
else:
    sys.path.append(current_dir)

from src.utils.vault import vault
from src.utils.report_templates import generate_script_report

# ── Configuration (adjust via environment or inline) ──
TIMEOUT_FILL_MS        = int(os.environ.get("TIMEOUT_FILL_MS", "{TIMEOUT_FILL_MS}"))
TIMEOUT_CLICK_MS       = int(os.environ.get("TIMEOUT_CLICK_MS", "{TIMEOUT_CLICK_MS}"))
TIMEOUT_TABLE_WAIT_MS  = int(os.environ.get("TIMEOUT_TABLE_WAIT_MS", "{TIMEOUT_TABLE_WAIT_MS}"))
TIMEOUT_NETWORKIDLE_MS = int(os.environ.get("TIMEOUT_NETWORKIDLE_MS", "{TIMEOUT_NETWORKIDLE_MS}"))
TIMEOUT_LOGIN_MS       = 2000

# Vault Prefix (Dynamic Discovery)
VAULT_PREFIX = os.environ.get("VAULT_CREDENTIAL_PREFIX", "{vault_prefix}")
if not vault.get_credentials(VAULT_PREFIX):
    keys = vault.list_keys()
    if keys:
        VAULT_PREFIX = keys[0]

LOGIN_USER_SELECTORS   = {user_sels_str}
LOGIN_PASS_SELECTORS   = {pass_sels_str}
LOGIN_SUBMIT_SELECTORS = {submit_sels_str}

def clean_text(text):
    if not text: return ""
    import re
    # Remove non-breaking spaces and redundant whitespace
    text = text.replace("\\u00a0", " ").replace("\\xa0", " ")
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# Screenshot Directory Setup
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

_screenshot_count = 0

async def take_full_screenshot(page, description):
    global _screenshot_count
    import re
    _screenshot_count += 1
    slug = re.sub(r'[^a-z0-9]+', '_', description.lower()).strip('_')
    if not slug: slug = "screenshot"
    filename = f"{{_screenshot_count:02d}}_{{slug}}.png"
    path = SCREENSHOT_DIR / filename
    try:
        # Use OS-level screen capture to include URL bar and full browser chrome
        import pyautogui
        await page.wait_for_timeout(500)  # Brief pause for rendering
        screenshot = pyautogui.screenshot()
        screenshot.save(str(path))
        print(f"📸 Full-screen screenshot saved: {{path}}")
    except ImportError:
        # Fallback to Playwright page-only screenshot if pyautogui not available
        try:
            await page.screenshot(path=str(path), full_page=True)
            print(f"📸 Screenshot saved (page only): {{path}}")
        except Exception as e:
            print(f"⚠️ Failed to take screenshot '{{description}}': {{e}}")
    except Exception as e:
        print(f"⚠️ Failed to take screenshot '{{description}}': {{e}}")
'''

    return f'''You are an expert Playwright automation engineer. Generate a fully **self-contained** Python script from the execution history below.

### ✅ WHAT TO GENERATE
Write a COMPLETE, STANDALONE Python script that:
1. Needs NO external project imports — only standard library + `playwright`.
2. Defines any helper functions it needs INLINE in the script.
3. Can be run from any directory with just `python script.py`.

### 🛡️ GOLDEN RULES
1.  **Self-contained** — The script MUST be fully standalone. Do NOT import from
    `src.utils`, `script_helpers`, or any project-internal module.
    Define all functions you need directly in the script.

2.  **Credentials** — Read from the project vault using `vault.get_credentials(VAULT_PREFIX)`.
    The script header automatically attempts to find the correct `VAULT_PREFIX` from available keys. 
    NEVER hardcode actual credentials.

3.  **Login handling** — If the history shows a login form, write a login function EXACTLY like this:
    ```python
    async def login(page):
        creds = vault.get_credentials(VAULT_PREFIX)
        if not creds or "username" not in creds or "password" not in creds:
            return
        username, password = creds["username"], creds["password"]

        # Fill username
        for sel in LOGIN_USER_SELECTORS:
            try:
                await page.locator(sel).fill(username, timeout=TIMEOUT_LOGIN_MS)
                break
            except Exception: continue
        # Fill password
        for sel in LOGIN_PASS_SELECTORS:
            try:
                await page.locator(sel).fill(password, timeout=TIMEOUT_LOGIN_MS)
                break
            except Exception: continue
        # Submit
        for sel in LOGIN_SUBMIT_SELECTORS:
            try:
                await page.locator(sel).click(timeout=TIMEOUT_LOGIN_MS)
                await page.wait_for_load_state("networkidle")
                return
            except Exception: continue
    ```

        async def extract_table(page, selector, column_names):
            print(f"DEBUG: Extracting table with selector: {{selector}}")
            try:
                await page.wait_for_selector(selector, timeout=TIMEOUT_TABLE_WAIT_MS)
            except Exception as e:
                if selector.strip() != "table":
                    try:
                        print(f"DEBUG: Specific selector failed, falling back to generic 'table'")
                        await page.wait_for_selector("table", timeout=10000)
                        selector = "table"
                    except: raise e
                else: raise e

            tables = await page.locator(selector).all()
            print(f"DEBUG: Found {{len(tables)}} tables matching selector.")

            best_table_data = []
            best_score = -1
            best_exact_count = -1

            for table_idx, table in enumerate(tables):
                # Skip layout tables that contain nested tables
                nested_count = await table.locator("table").count()
                if nested_count > 0:
                    print(f"DEBUG: Table {{table_idx}} has {{nested_count}} nested tables, skipping (layout table).")
                    continue

                # Use direct-child row selectors to avoid inheriting nested table rows
                rows = await table.locator(":scope > tbody > tr, :scope > thead > tr, :scope > tr").all()
                if len(rows) < 2: continue

                table_best_indices = {{}}
                table_best_header_idx = -1
                table_best_score = -1
                table_best_exact = -1

                search_limit = min(len(rows), 8)
                for r_idx in range(search_limit):
                    cells = await rows[r_idx].locator(":scope > th, :scope > td").all()
                    current_headers = [clean_text(await c.text_content()) for c in cells]

                    exact_count = 0
                    fuzzy_count = 0
                    indices = {{}}

                    for col in column_names:
                        for i, h in enumerate(current_headers):
                            if col.lower() == h.lower():
                                if col not in indices and i not in indices.values():
                                    indices[col] = i
                                    exact_count += 1
                                    break

                    for col in column_names:
                        if col in indices: continue
                        for i, h in enumerate(current_headers):
                            if i in indices.values(): continue
                            if col.lower() in h.lower() and len(h) < 50:
                                indices[col] = i
                                fuzzy_count += 1
                                break

                    score = exact_count * 10 + fuzzy_count
                    if score > table_best_score or (score == table_best_score and exact_count > table_best_exact):
                        table_best_score = score
                        table_best_exact = exact_count
                        table_best_indices = indices
                        table_best_header_idx = r_idx

                if table_best_score > 0 and table_best_header_idx >= 0:
                    data = []
                    for row in rows[table_best_header_idx + 1:]:
                        cells = await row.locator(":scope > th, :scope > td").all()
                        if len(cells) < len(table_best_indices): continue
                        row_data = {{}}
                        found_any = False
                        for col, idx in table_best_indices.items():
                            if idx < len(cells):
                                val = clean_text(await cells[idx].text_content())
                                row_data[col] = val
                                if val: found_any = True
                        if found_any:
                            data.append(row_data)

                    if len(data) > 0 and (table_best_score > best_score or (table_best_score == best_score and table_best_exact > best_exact_count)):
                        best_score = table_best_score
                        best_exact_count = table_best_exact
                        best_table_data = data
                        print(f"DEBUG: Table {{table_idx}} scored {{table_best_score}} (exact={{table_best_exact}}), found {{len(data)}} rows.")

            return best_table_data

    5.  **Selectors** — Use selectors from `dom_elements.selectors` in the history
        (`preferred` > `css` > `recommended_selector` > `xpath`).
        If `dom_elements` is empty, use scoped selectors like `form#id table`.
        NEVER hallucinate selectors. NEVER use bare `"table"` — always scope it.

    6.  **Report output** — Print results in a clear format. At the END of the script, call `generate_script_report()` to save a Word (.docx) report. NEVER save results to a `.txt` file.

    7.  **URL handling** — CRITICAL: Use the BASE application URL (e.g. `http://host/app/`),
        NOT login/auth redirect URLs. The base URL auto-redirects to login if needed.
        Use `timeout=TIMEOUT_TABLE_WAIT_MS` for `page.goto`.

    8.  **Post-navigation waits** — ALWAYS add:
        - `await page.wait_for_load_state("domcontentloaded")` after `page.goto(...)`
        - `await page.wait_for_load_state("networkidle", timeout=TIMEOUT_NETWORKIDLE_MS)` after login

9.  **Loops** — ALWAYS `elements = await locator.all()` then `for el in elements:`.
    Never `async for`.

10. **No noise** — Skip failed/retried actions from history. Only generate the successful path.

11. **Empty DOM** — If a step has empty `dom_elements`, do NOT invent selectors.
    Use scoped selectors from form IDs or container IDs.

    12. **Column hints** — If a step has `required_columns_hint`, extract those EXACT
        columns. Include ALL columns listed in the hint, plus any identity column
        (Server, Name) if not already present.
        CRITICAL: NEVER use URLs, paths, or credentials as column names.
        If a hint contains a URL, ignore that item. Column names MUST be simple strings like 'Status' or 'Worker ID'.

    13. **Resilience & JSON** — 
        - ALWAYS use `.get('Column', '')` when accessing row dictionaries (e.g., `row.get('ID', 'N/A')`). NEVER use direct access like `row['ID']`. This is MANDATORY to prevent `KeyError`.
        - Keep column lists SEPARATE for different logical tables. Do not merge all columns into one list if the history shows distinct extraction goals.
        - NEVER generate conversational summaries in the report script based on your own knowledge. 
        - ONLY use the data returned by `extract_table`.
        - If the extraction result is empty, handle it gracefully (e.g., `print("No data found")`).

    14. **Full-Page Screenshots** — ALWAYS call `await take_full_screenshot(page, "Descriptive Name")` at these points:
        - After a successful login.
        - After every major navigation to a new URL.
        - Immediately after extracting data from a table (to verify findings).
        The name MUST be relevant to the page content (e.g., "login_success", "agent_status_table").

    15. **Scrolling Resilience** — If the execution history shows `scroll_to_text(...)` failing (text not found or visible), or if you need to reach the bottom of a long page (like a Distributed Agent Status page with many workers):
        - Prefer `await page.mouse.wheel(0, 800)` or `scroll_page(direction='down', amount=800)` logic if text-based scrolling is brittle.
        - ALWAYS add `await page.wait_for_timeout(1000)` after a large scroll to allow lazy-loaded content to render.

    16. **Word Report** — At the END of the script (before `await browser.close()`), call:
        ```python
        steps = []  # Build this as you go: steps.append({"action": "action_name", "output": "captured text"})
        generate_script_report(
            script_name=os.path.basename(__file__),
            steps=steps,
            screenshots_dir=str(SCREENSHOT_DIR),
            output_path=str(SCREENSHOT_DIR.parent / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"),
            status="Execution Complete",
            captured_outputs="<combined text output>",
        )
        ```
        Build the `steps` list incrementally: after each action or extraction, append a dict with `{"action": ..., "output": ...}`.
        The `screenshots_dir` is SCREENSHOT_DIR, which already holds the numbered screenshots.

### 📝 STRUCTURE
Use this header, then write helper functions and `async def run()`:

```python
{script_header}

# Define helper functions here (login, extract_table, etc.) as needed

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        context = await browser.new_context(no_viewport=True)
        page = await context.new_page()

        # ... navigation, login, extraction, report ...

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
```

Write helper functions ABOVE `async def run()` or inline inside it.
Only import additional standard library modules if needed (e.g. `import re`, `from datetime import datetime`).
'''

def columns_from_results(results: list) -> list[str]:
    """Parse leaf-row keys from extracted_content fenced JSON in agent results.
    
    Strategy (per spec §3.A):
      1. Parse fenced ```json``` blocks from extracted_content.
      2. Walk nested objects/lists and collect all leaf-dict keys.
      3. Return deduplicated ordered list of column names found.
    """
    if not results:
        return []
    for res in results:
        if not isinstance(res, dict):
            continue
        content = res.get("extracted_content") or ""
        # Accept both fenced and raw JSON blobs
        json_candidates = []
        m = re.search(r'```(?:json)?\s*([\[\{].*?)\s*```', content, re.DOTALL)
        if m:
            json_candidates.append(m.group(1))
        else:
            # Try the entire content as JSON
            stripped = content.strip()
            if stripped.startswith(('[', '{')):
                json_candidates.append(stripped)

        for candidate in json_candidates:
            try:
                data = json.loads(candidate)
            except Exception:
                continue
            cols = []
            seen_cols = set()

            def walk(obj, depth=0):
                if depth > 10: return
                if isinstance(obj, dict):
                    if obj and all(not isinstance(v, (dict, list)) for v in obj.values()):
                        for k in obj.keys():
                            if k not in seen_cols:
                                seen_cols.add(k)
                                cols.append(k)
                    else:
                        for v in obj.values(): walk(v, depth + 1)
                elif isinstance(obj, list):
                    for v in obj: walk(v, depth + 1)

            walk(data)
            if cols: return cols
        
        # 3. FALLBACK: Parse Markdown tables or lists if JSON is missing
        md_cols = []
        table_match = re.search(r'\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|', content)
        if table_match and "---" in content:
            header_line = table_match.group(0)
            md_cols = [c.strip() for c in header_line.split("|") if c.strip()]
        
        if not md_cols:
            bullet_matches = re.findall(r'^[ \t]*[-*+]\s*\*\*?([^\*:]+)\*\*?:\s*', content, re.MULTILINE)
            if bullet_matches:
                md_cols = list(dict.fromkeys([b.strip() for b in bullet_matches]))
        
        if md_cols:
            filtered_md = [c for c in md_cols if len(c) < 40 and not any(n in c.lower() for n in ['http', '/', 'click', 'page'])]
            if filtered_md: return filtered_md
    return []

def infer_required_columns(text: str) -> list[str]:
    """Infer required columns from the user's objective/goal."""
    if not text: return []
    candidates = []
    candidates += re.findall(r'"([^"]{2,30})"', text)
    candidates += re.findall(r"'([^']{2,30})'", text)
    for line in re.split(r'[\r\n]+', text):
        if ":" in line or " - " in line:
            tail = re.split(r'[:\-]\s*', line, maxsplit=1)[-1]
            parts = re.split(r',| and ', tail, flags=re.IGNORECASE)
            candidates += [p.strip() for p in parts if p.strip()]
    for m in re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})', text):
        if len(m) > 40: continue
        if "http" in m.lower() or "/" in m: continue
        candidates.append(m)

    common_noisy_words = {
        'navigate', 'click', 'http', 'https', 'login', 'server', 'secret', 'password', 'username', 'check', 'extract', 
        'read', 'done', 'log', 'in', 'do', 'find', 'locate', 'entire', 'table', 'once', 'see', 'results', 'consolidated',
        'report', 'summarizing', 'findings', 'key', 'points', 'distributed', 'system', 'agent', 'dashboard'
    }
    cleaned = []
    seen = set()
    for c in candidates:
        c = c.strip().strip(".:'\"[]()")
        c_lower = c.lower()
        if not c or len(c) < 2 or len(c) > 40: continue
        words = c_lower.split()
        noisy_count = sum(1 for w in words if w in common_noisy_words)
        if noisy_count >= len(words) / 2: continue
        if "://" in c or "/" in c or "\\" in c: continue
        if c not in seen:
            seen.add(c)
            cleaned.append(c)
    return cleaned

SYSTEM_PROMPT = _build_system_prompt()

def clean_history_json(history_data: Dict[str, Any], objective: str = None, history_path: str = None) -> str:
    """Optimize JSON payload by pruning redundant actions, failed attempts, and duplicate navigations.
    
    Args:
        history_data: Raw history dict
        objective: Task objective string
        history_path: Path to history JSON file (used to resolve DOM snapshots)
    """
    # ── Snapshot resolution: find DOM snapshots directory for this run ──
    _snapshots_by_step = {}  # step_index -> snapshot dict
    if history_path:
        try:
            hp = Path(history_path).resolve()
            # Walk up the directory tree to find a sibling "dom_snapshots" folder
            dom_root = None
            for parent in hp.parents:
                candidate = parent / "dom_snapshots"
                if candidate.exists() and candidate.is_dir():
                    dom_root = candidate
                    break
            # Try multiple candidate directory names
            if dom_root:
                for candidate in [hp.stem, hp.parent.name]:
                    snap_dir = dom_root / candidate
                    if snap_dir.exists():
                        for snap_file in sorted(snap_dir.glob("*.json"), key=lambda p: int(p.stem)):
                            try:
                                with snap_file.open("r", encoding="utf-8") as sf:
                                    snap_data = json.load(sf)
                                step_idx = snap_data.get("metadata", {}).get("step_index")
                                if step_idx is not None:
                                    _snapshots_by_step[step_idx] = snap_data
                            except Exception:
                                continue
                        if _snapshots_by_step:
                            logger.info(f"Loaded {len(_snapshots_by_step)} DOM snapshots from {snap_dir}")
                            break
                # Also try orchestration-style path (snapshot dir name includes process name)
                if not _snapshots_by_step:
                    for entry in dom_root.iterdir():
                        if entry.is_dir() and hp.stem in entry.name:
                            for snap_file in sorted(entry.glob("*.json"), key=lambda p: int(p.stem)):
                                try:
                                    with snap_file.open("r", encoding="utf-8") as sf:
                                        snap_data = json.load(sf)
                                    step_idx = snap_data.get("metadata", {}).get("step_index")
                                    if step_idx is not None:
                                        _snapshots_by_step[step_idx] = snap_data
                                except Exception:
                                    continue
                            if _snapshots_by_step:
                                logger.info(f"Loaded {len(_snapshots_by_step)} DOM snapshots from {entry}")
                                break
        except Exception as e:
            logger.debug(f"Could not load DOM snapshots: {e}")
    simplified = []
    seen_urls = []
    
    for item in history_data.get('history', []):
        state = item.get('state', {})
        model_out = item.get('model_output', {})
        action = model_out.get('action', [])
        evaluation = model_out.get('current_state', {}).get('evaluation_previous_goal', '')
        
        current_url = state.get('url', '')
        if current_url:
            # Simple URL deduplication: if the URL is logically the same as the last one, mark as redundant
            # (ignoring minor query param diffs or typos that redirect)
            if seen_urls and seen_urls[-1].split('?')[0] == current_url.split('?')[0]:
                is_redundant_nav = True
            else:
                is_redundant_nav = False
                seen_urls.append(current_url)
        else:
            is_redundant_nav = False

        step_info = {
            "url": current_url,
            "title": state.get('title'),
            "evaluation": evaluation,
            "goal": model_out.get('current_state', {}).get('next_goal'),
            "actions": [],
            "dom_elements": []
        }
        
        # Extract actions and their context
        for act_idx, act in enumerate(action):
            # PRUNING: Only skip if there was an actual error AND it wasn't a core interaction
            # Sometimes 'timeouts' happen but the action actually worked (SPA redirects)
            # We keep 'click', 'input', 'upload' unless they are followed by a successful retry
            results = item.get('result', [])
            if act_idx < len(results):
                res = results[act_idx]
                if isinstance(res, dict) and res.get('error'):
                    error_msg = res.get('error', '').lower()
                    action_type = next(iter(act.keys())) if act else "unknown"
                    
                    # If it's a timeout for a core action, keep it - the LLM can decide
                    if "timeout" in error_msg and action_type in ['click_element', 'input_text', 'upload_file', 'click_element_by_text']:
                        logger.info(f"Keeping core action {action_type} despite timeout: {error_msg}")
                    else:
                        logger.info(f"Pruning action {act_idx} due to error: {res.get('error')}")
                        continue

            # Skip redundant navigations if they were just retries of the same URL
            if is_redundant_nav and "go_to_url" in act:
                continue

            # Create a shallow copy of action to exclude dom_context from the 'actions' list
            act_copy = {k: v for k, v in act.items() if k != 'dom_context'}
            step_info['actions'].append(act_copy)
            
            if 'dom_context' in act:
                ctx = act['dom_context']
                selectors = ctx.get('selectors', {})
                attrs = ctx.get('attributes', {})
                comp = ctx.get('comprehensive', {})
                
                # ENRICHED ELEMENT DATA: Pass all critical selectors to the LLM
                # Also look into 'comprehensive' data which often has better labels
                comp_attrs = comp.get('all_attributes', {})
                comp_selectors = {s['type']: s['selector'] for s in comp.get('all_selectors', [])}
                
                element_data = {
                    "tag": ctx.get('tagName') or comp.get('tag'),
                    "text": selectors.get('text', '') or attrs.get('textContent', '') or comp_attrs.get('title', '') or comp_attrs.get('textContent', ''),
                    "title": attrs.get('title', '') or comp_attrs.get('title', ''),
                    "id": attrs.get('id', '') or comp_attrs.get('id', ''),
                    "name": attrs.get('name', '') or comp_attrs.get('name', ''),
                    "type": attrs.get('type', '') or comp_attrs.get('type', ''),
                    "role": attrs.get('role', '') or comp_attrs.get('role', ''),
                    "aria-label": attrs.get('aria-label', '') or comp_attrs.get('aria-label', ''),
                    "selectors": {**selectors, **comp_selectors}, # Merge basic and comprehensive selectors
                }
                
                # Clean up empty selectors
                element_data["selectors"] = {k: v for k, v in element_data["selectors"].items() if v}

                # Include recommended selector if available
                rec_sel = comp.get('recommended_selector', {}).get('selector', '')
                if rec_sel:
                    element_data["recommended_selector"] = rec_sel
                    
                step_info['dom_elements'].append(element_data)

            if isinstance(act, dict) and "extract_content" in act:
                goal = act.get("extract_content", {}).get("goal", "") if isinstance(act.get("extract_content", {}), dict) else ""
                cols = columns_from_results(results)
                if cols:
                    goal_text = f"{objective or ''} {goal}".lower()
                    # Singular/plural-tolerant matching (per spec §3.A)
                    def _goal_match(col_name):
                        c_l = col_name.lower()
                        c_singular = c_l[:-1] if c_l.endswith("s") else c_l
                        c_plural = c_l + "s" if not c_l.endswith("s") else c_l
                        return c_l in goal_text or c_singular in goal_text or c_plural in goal_text

                    filtered = [c for c in cols if _goal_match(c)]
                    # Fall back to ALL cols if objective filter is too narrow
                    required = filtered if filtered else cols

                    # Identity-key fallbacks (per spec §3.A)
                    goal_mentions_server = any(w in goal_text for w in ("server", "entity", "host", "node"))
                    has_server_col = any("server" in c.lower() for c in required)
                    has_name_col = any(c.lower() == "name" for c in required)
                    if goal_mentions_server and not has_server_col and not has_name_col:
                        required = list(required) + ["Server"]
                else:
                    required = infer_required_columns(f"{objective or ''} {goal}")
                if required:
                    step_info["required_columns_hint"] = required

                # ── Snapshot enrichment: inject real table selectors ──────
                # If dom_elements is empty for this extract step, try to
                # find the best matching table from the DOM snapshot.
                if not step_info['dom_elements'] and _snapshots_by_step:
                    # Get extracted content text for matching
                    extracted_text = ""
                    for r in results:
                        if isinstance(r, dict) and r.get("extracted_content"):
                            extracted_text = r["extracted_content"]
                            break

                    # Find the closest snapshot (same step or nearest before)
                    snap = None
                    step_meta = item.get('metadata', {})
                    current_step_num = step_meta.get('step_number', step_idx + 1)
                    for try_idx in [current_step_num, current_step_num - 1, current_step_num + 1]:
                        if try_idx in _snapshots_by_step:
                            snap = _snapshots_by_step[try_idx]
                            break
                    # Fallback: use the last available snapshot
                    if not snap and _snapshots_by_step:
                        snap = _snapshots_by_step[max(_snapshots_by_step.keys())]

                    if snap:
                        snap_elements = snap.get("elements", [])
                        tables = [
                            e for e in snap_elements
                            if e.get("identity", {}).get("tagName", "").lower() == "table"
                        ]
                        if tables:
                            # Score each table by text overlap with extracted content
                            # + specificity bonus (prefer deeper/narrower tables)
                            extracted_lower = extracted_text.lower() if extracted_text else ""
                            best_table = None
                            best_score = -1
                            for tbl in tables:
                                sp = tbl.get("selector_provenance", {})
                                tbl_text = sp.get("text", "").lower()
                                if not tbl_text or len(tbl_text) < 10:
                                    continue
                                # Base score: word overlap with extracted content
                                ext_words = set(re.findall(r'\w{3,}', extracted_lower))
                                tbl_words = set(re.findall(r'\w{3,}', tbl_text))
                                overlap = len(ext_words & tbl_words)
                                # Bonus: column name matches (weighted higher)
                                if required:
                                    col_matches = sum(1 for c in required if c.lower() in tbl_text)
                                    overlap += col_matches * 3
                                # Specificity bonus: prefer deeper tables (longer selector path)
                                # This avoids picking broad outer containers
                                preferred = sp.get("preferred", "")
                                depth = preferred.count(">")
                                overlap += min(depth, 5)  # Cap bonus at 5
                                # Penalty: tables whose text is identical to a parent/sibling
                                # (container tables that just inherit child text) — penalize
                                # tables with very long text relative to actual column headers
                                if required and len(tbl_text) > 500 and col_matches < len(required):
                                    overlap -= 2  # Likely a container, not the data table
                                if overlap > best_score:
                                    best_score = overlap
                                    best_table = tbl

                            if best_table and best_score > 0:
                                sp = best_table.get("selector_provenance", {})
                                af = best_table.get("attribute_fingerprint", {})
                                table_element = {
                                    "tag": "table",
                                    "id": af.get("id", ""),
                                    "text": sp.get("text", "")[:200],
                                    "selectors": {
                                        k: v for k, v in {
                                            "preferred": sp.get("preferred", ""),
                                            "css": sp.get("css", ""),
                                            "xpath": sp.get("xpath", ""),
                                        }.items() if v
                                    },
                                    "class_list": af.get("class_list", []),
                                    "_source": "dom_snapshot",
                                }
                                step_info['dom_elements'].append(table_element)
                                logger.info(f"Step {step_idx}: Enriched extract_content with table selector from DOM snapshot (score={best_score})")
                 
        # Final Pruning: Skip steps that contain no successful actions unless it's a Terminal step
        if not step_info['actions'] and "done" not in str(step_info['goal']).lower():
            continue
                
        simplified.append(step_info)
    
    result_data = simplified
    if objective:
        result_data = {"objective": objective, "steps": simplified}
        
    return json.dumps(result_data, indent=2)

def _extract_table_hints(cleaned_history_json: str):
    """Extract the best table selector, required columns, and target URL from cleaned history.
    
    Selector priority per spec §3.B:
      preferred > css > recommended_selector > xpath > (first available)
    Prefers steps that already carry required_columns_hint to maximise coherence.

    Returns: (selector, required_columns, target_url)
    """
    try:
        data = json.loads(cleaned_history_json)
    except Exception:
        return None, None, None

    steps = data.get("steps") if isinstance(data, dict) else data
    if not isinstance(steps, list):
        return None, None, None

    SELECTOR_PRIORITY = ["preferred", "css", "recommended_selector", "xpath", "aria", "text"]

    def _best_selector_from_element(el: dict) -> str | None:
        if not isinstance(el, dict):
            return None
        selectors = el.get("selectors") or {}
        # Check recommended_selector as a standalone field too
        rec = el.get("recommended_selector", "")
        if isinstance(rec, dict):
            rec = rec.get("selector", "")
        for key in SELECTOR_PRIORITY:
            if key == "recommended_selector":
                val = rec
            else:
                val = selectors.get(key)
            if val and isinstance(val, str) and val.strip() and val.strip().lower() != "table":
                return val.strip()
        return None

    best_selector = None
    best_required = None
    target_url = None

    # Extract target URL: prefer first step URL that looks like a process page
    # (has query params or is not the bare login page)
    for step in steps:
        if not isinstance(step, dict):
            continue
        url = step.get("url", "")
        if url and "?" in url and not any(
            login_kw in url.lower()
            for login_kw in ("login", "otdsws", "signin", "auth/login")
        ):
            target_url = url
            break
    # Fallback: any URL from the steps
    if not target_url:
        for step in steps:
            if isinstance(step, dict):
                url = step.get("url", "")
                if url and not any(kw in url.lower() for kw in ("login", "otdsws", "signin")):
                    target_url = url
                    break

    # First pass: prefer steps that carry required_columns_hint (most coherent)
    for step in steps:
        if not isinstance(step, dict):
            continue
        required = step.get("required_columns_hint")
        selector = None
        for el in step.get("dom_elements", []) or []:
            sel = _best_selector_from_element(el)
            if sel:
                selector = sel
                break
        if required and selector:
            return selector, required, target_url
        # Keep partial results as fallback
        if required and best_required is None:
            best_required = required
        if selector and best_selector is None:
            best_selector = selector

    # Second pass: accept any step with at least a selector
    if best_selector or best_required:
        return best_selector, best_required, target_url

    return None, None, target_url

def _is_login_selector(sel: str) -> bool:
    """Return True if a selector looks like a login-form field, not a data table."""
    login_patterns = [
        r'#otds_username', r'#otds_password', r'#username', r'#password',
        r'input\[name=["\']?(?:username|password|user|otds_username|otds_password)',  # detection patterns — intentional
        r'input\[type=["\']?(?:password|text)',
        r'input#(?:username|password)',
    ]
    sel_lower = sel.lower()
    return any(re.search(p, sel_lower) for p in login_patterns)


def _postprocess_generated_code(code: str, cleaned_history_json: str) -> str:
    """Enforce correct structure in LLM-generated scripts.

    The LLM is instructed to only write async def run(), importing all helpers
    from src.utils.script_helpers. This function enforces that contract:

      0. Merge duplicate run() blocks (mandatory + process concatenated).
      1. Inject 'from src.utils.script_helpers import ...' if missing.
      2. Strip any helper function bodies the LLM hallucinated (get_secret,
         find_in_frames, resilient_*, maybe_login, generate_report, etc.).
      3. Rewrite wrong/login-field table selectors to the resolved DOM selector.
      4. Inject required columns (with identity key) into extraction calls.
      5. Ensure maybe_login(page) is called after page.goto.
      6. Fix post-login wait_for_selector if it targets a login field.
      7. Normalise row identity access to safe .get() fallback chain.
      8. Strip unused 'import re' and 'from docx import Document'.
      9. Replace bare 'except:' with 'except Exception as _e:'.
     10. Normalise hardcoded vault key names to use VAULT_CREDENTIAL_PREFIX.
     11. Strip redundant manual login steps that duplicate maybe_login.
    """
    selector, required, process_url = _extract_table_hints(cleaned_history_json)

    # ── 0: Merge duplicate run() blocks ──────────────────────────────────────
    # When mandatory + process scripts are concatenated, two full script blocks
    # appear separated by 'if __name__ == "__main__": asyncio.run(run())'.
    # Strategy: split on entrypoint boundaries, extract each run() body,
    # deduplicate browser setup, merge task steps, reconstruct one clean script.
    _entrypoint_pat = re.compile(
        r'if __name__\s*==\s*["\']__main__["\']\s*:\s*\n\s*asyncio\.run\(run\(\)\)'
    )
    entrypoint_positions = [m.start() for m in _entrypoint_pat.finditer(code)]

    if len(entrypoint_positions) > 1:
        logger.info(f"[POST] Merging {len(entrypoint_positions)} script blocks into one")

        # Split code into segments at each entrypoint
        segments = []
        prev = 0
        for pos in entrypoint_positions:
            end = _entrypoint_pat.search(code, pos).end()
            segments.append(code[prev:end])
            prev = end
        if prev < len(code):
            segments.append(code[prev:])

        # Extract the header (imports + bootstrap) from the LAST segment
        # (it has the most complete imports after postprocessing merges)
        def _split_header_body(seg):
            """Return (header_str, run_body_str) for a segment."""
            run_start = seg.find("async def run():\n")
            if run_start == -1:
                return seg, ""
            header = seg[:run_start]
            after_def = seg[run_start + len("async def run():\n"):]
            # Body = everything from first indented line up to (not including) if __name__
            ep = _entrypoint_pat.search(after_def)
            body = after_def[:ep.start()].rstrip("\n") + "\n" if ep else after_def
            return header, body

        # Use last segment's header (most up to date imports)
        best_header, _ = _split_header_body(segments[-1])

        # Collect all run() bodies in order
        all_bodies = []
        for seg in segments:
            _, body = _split_header_body(seg)
            if body.strip():
                all_bodies.append(body)

        # Merge bodies: keep browser setup from first body only
        _browser_setup_kws = (
            "async with async_playwright",
            "p.chromium.launch",
            "browser.new_context",
            "context.new_page",
        )
        merged_lines = []
        browser_setup_seen = False
        close_line = None

        for body_idx, body in enumerate(all_bodies):
            body_lines = body.splitlines(keepends=True)
            in_browser_block = False
            for line in body_lines:
                stripped = line.strip()

                # Defer browser.close() to very end
                if "await browser.close()" in line:
                    close_line = line
                    continue

                # Track/skip duplicate browser setup lines
                is_setup = any(kw in stripped for kw in _browser_setup_kws)
                if is_setup:
                    if browser_setup_seen:
                        in_browser_block = True
                        continue
                    else:
                        in_browser_block = False  # first occurrence: keep
                elif in_browser_block and stripped and not stripped.startswith("#"):
                    # Once we exit setup keywords, we're back in task logic
                    in_browser_block = False

                # Collapse excessive blank lines between merged blocks
                if not stripped and merged_lines and not merged_lines[-1].strip():
                    continue

                merged_lines.append(line)

            browser_setup_seen = True
            # Add a blank line separator between merged sections
            if merged_lines and merged_lines[-1].strip() and body_idx < len(all_bodies) - 1:
                merged_lines.append("\n")

        if close_line:
            if merged_lines and merged_lines[-1].strip():
                merged_lines.append("\n")
            merged_lines.append(close_line)

        merged_body = "".join(merged_lines)

        # Deduplicate maybe_login: only the FIRST call is needed (session persists)
        first_login_seen = False
        deduped = []
        for line in merged_body.splitlines(keepends=True):
            if "await maybe_login(page)" in line:
                if first_login_seen:
                    logger.info("[POST] Removed duplicate maybe_login(page) call")
                    continue
                first_login_seen = True
            deduped.append(line)
        merged_body = "".join(deduped)

        # Build best header: use imports/bootstrap from last segment
        # Find everything before 'async def run():' in the last segment
        last_seg = segments[-1]
        run_def_idx = last_seg.find("async def run():\n")
        best_header = last_seg[:run_def_idx] if run_def_idx != -1 else ""

        # If header is empty/short (bootstrap missing), fall back to first segment
        if best_header.count("\n") < 5:
            first_seg = segments[0]
            run_def_idx0 = first_seg.find("async def run():\n")
            if run_def_idx0 != -1:
                best_header = first_seg[:run_def_idx0]

        code = (
            best_header.rstrip("\n") + "\n"
            + "async def run():\n"
            + merged_body
            + '\nif __name__ == "__main__":\n'
            + "    asyncio.run(run())\n"
        )


    # ── 1 & 2: REMOVED — scripts are now fully self-contained ─────────────────
    # The LLM writes all functions inline. No script_helpers import is injected,
    # and no helper functions are stripped. The LLM decides what functions to write.
    #
    # Strip any accidental project-internal imports the LLM may still emit:
    # EXCEPT for src.utils.vault which we now explicitly use.
    code = re.sub(r'^from src\.utils\.script_helpers import[^\n]*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'^(?!from src\.utils\.vault)from src\.utils import[^\n]*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'^from src\.utils\.config import[^\n]*\n', '', code, flags=re.MULTILINE)
    # Safety net: replace any hallucinated maybe_login with login
    code = code.replace('await maybe_login(page)', 'await login(page)')
    code = code.replace('maybe_login(page)', 'login(page)')
    # Strip redundant path bootstrap blocks if the LLM hallucinated its own
    code = re.sub(
        r'(?m)^# Path bootstrap[^\n]*\n'
        r'(?:current_dir[^\n]*\n)?'
        r'(?:if[^\n]*web-ui[^\n]*\n'
        r'[ \t]+sys\.path\.append[^\n]*\n'
        r'else:\n'
        r'[ \t]+sys\.path\.append[^\n]*\n)?',
        '', code,
    )

    # ── 3: Rewrite wrong/login-field table selectors (ONLY if generic/wrong) ──
    def _should_replace(sel_str: str) -> bool:
        s = sel_str.strip().strip('"\'')
        # Only replace if the LLM used a very generic 'table' or a login field.
        # If it found a specific ID like '#workerTable', leave it as the primary intent.
        return s == "table" or _is_login_selector(s)

    if selector:
        sel_literal = json.dumps(selector)

        def _fix_wait(m):
            return f'page.wait_for_selector({sel_literal}' if _should_replace(m.group(1)) else m.group(0)
        code = re.sub(r'page\.wait_for_selector\((["\'][^"\']+["\'])', _fix_wait, code)

        def _fix_extract(m):
            # Only fix the FIRST extract_table call if it's generic, 
            # as different tables likely need different selectors.
            return f'extract_table(page, {sel_literal}' if _should_replace(m.group(1)) else m.group(0)
        code = re.sub(r'extract_table\(\s*page\s*,\s*(["\'][^"\']+["\'])', _fix_extract, code, count=1)

        def _fix_table_var(m):
            return f'table_selector = {sel_literal}' if _should_replace(m.group(1)) else m.group(0)
        code = re.sub(r'table_selector\s*=\s*(["\'][^"\']+["\'])', _fix_table_var, code)

        def _fix_locator(m):
            return f'page.locator({sel_literal}' if _should_replace(m.group(1)) else m.group(0)
        code = re.sub(r'page\.locator\((["\'][^"\']+["\'])', _fix_locator, code)

    # ── 4: SAFETY: Always ensure each extract_table has its own column list ──
    # The post-processor previously merged all hints. We now ensure
    # the LLM's original specific lists survive unless they are bare hallucinations.
    # We only inject 'Server' if it's completely missing from the script.
    if required and "Server" not in code:
        # Check if the code has a generic empty list or obviously wrong list
        code = re.sub(
            r'(extract_table\(\s*page\s*,\s*[^,]+,\s*)(\[\])',
            rf'\1{json.dumps(required)}', code
        )

    # ── 5: Ensure login(page) is called after first page.goto ────────────────
    if "login(page)" not in code:
        code = re.sub(
            r'(await page\.goto\([^\n]+\)\n)',
            r'\1        await login(page)\n',
            code, count=1,
        )

    # ── 6: Fix post-login wait if it targets a login-field selector ──────────
    if selector and "await login(page)" in code:
        sel_literal = json.dumps(selector)
        lines = code.splitlines(keepends=True)
        past_login, fixed_lines = False, []
        for line in lines:
            if "await login(page)" in line:
                past_login = True
            if past_login and "wait_for_selector" in line:
                m = re.search(r'wait_for_selector\((["\'][^"\']+["\'])', line)
                if m and _is_login_selector(m.group(1).strip("'\"")):
                    line = re.sub(
                        r'wait_for_selector\(["\'][^"\']+["\']',
                        f'wait_for_selector({sel_literal}', line,
                    )
                    logger.info(f"[POST] Fixed post-login wait → {selector}")
            fixed_lines.append(line)
        code = "".join(fixed_lines)

    # ── 7: Normalise row identity access ─────────────────────────────────────
    _id = "(row.get('Server') or row.get('Name') or 'Unknown')"
    _core = re.escape("row.get('Server') or row.get('Name') or 'Unknown'")

    def _collapse_redundant(src):
        # Matches the full outer expression including any extra wrapping parens:
        #   ((row.get('Server') or ... or 'Unknown') or row.get('Name') or 'Unknown')
        # The outer \(? and \)? handle the optional extra paren that wraps the whole thing.
        return re.sub(
            r'\(?'    # optional outer open paren
            r'\(+'    # one or more inner open parens
            + _core +
            r'\)+'    # one or more inner close parens
            r'\s*or\s*row\.get\([\'"]Name[\'"]\)\s*or\s*[\'"]Unknown[\'"]'
            r'\)?',   # optional outer close paren
            _id, src,
        )

    code = _collapse_redundant(code)           # pre-existing redundant chains
    code = re.sub(r"row\[['\"]Server['\"]\]", _id, code)
    code = re.sub(r"row\.get\(['\"]Server['\"](?:,\s*['\"][^'\"]*['\"])?\)", _id, code)
    code = re.sub(r"row\[['\"]Status['\"]\]", "row.get('Status', '')", code)
    code = re.sub(r"row\[['\"]Errors['\"]\]", "row.get('Errors', '')", code)
    code = re.sub(r"row\[['\"]Name['\"]\]",   "row.get('Name', '')",   code)
    code = _collapse_redundant(code)           # collapse again after substitutions

    # ── 8: Strip unused 'import re' ───────────────────────────────────────────
    if re.search(r'^import re\s*$', code, re.MULTILINE):
        without = re.sub(r'^import re\s*$', '', code, flags=re.MULTILINE)
        if not re.search(r'\bre\.', without):
            code = re.sub(r'^import re\s*\n', '', code, flags=re.MULTILINE)
            logger.info("[POST] Removed unused 'import re'")

    # ── 9: Replace bare 'except:' ────────────────────────────────────────────
    code = re.sub(
        r'(?m)^(\s+)except:\s*$',
        r'\1except Exception as _e:\n\1    print(f"[-] Error: {_e}")',
        code,
    )
    code = re.sub(
        r'(?m)^(\s+)except:\s*(pass|continue)\s*$',
        r'\1except Exception as _e:\n\1    \2',
        code,
    )

    # ── +2: Strip unused 're' import when re is not referenced in script body ─
    # Run AFTER all other transforms so we don't strip re that's actually needed.
    if re.search(r'^import re\s*$', code, re.MULTILINE):
        body_without_import = re.sub(r'^import re\s*$', '', code, flags=re.MULTILINE)
        if not re.search(r'\bre\.', body_without_import):
            code = re.sub(r'^import re\s*\n', '', code, flags=re.MULTILINE)
            logger.info("[POST] Removed unused 'import re'")

    # ── +5: Remove _timeout unused variable from find_in_frames copies ────────
    code = re.sub(
        r'[ \t]+_timeout\s*=\s*timeout\s+or\s+\w+\s*\n',
        '',
        code,
    )

    # ── 10: Normalise hardcoded vault key names → VAULT_CREDENTIAL_PREFIX ─────
    # LLM sometimes emits get_secret("OTCS_USERNAME") with a literal prefix.
    # Replace with get_secret(f"{VAULT_CREDENTIAL_PREFIX}_USERNAME") so the
    # script works for any system without env-var credentials.
    # Also handles PASSWORD, USER, PWD variants.
    _vault_key_pat = re.compile(
        r'get_secret\(\s*["\']([A-Z][A-Z0-9]*)_(USERNAME|PASSWORD|USER|PWD)["\']'
        r'\s*\)'
    )
    def _normalize_vault_key(m):
        suffix = m.group(2)  # USERNAME / PASSWORD / USER / PWD
        return f'get_secret(f"{{VAULT_CREDENTIAL_PREFIX}}_{suffix}")'

    if _vault_key_pat.search(code):
        # Only replace if the prefix is not already VAULT_CREDENTIAL_PREFIX
        def _maybe_replace(m):
            full_key = m.group(1) + "_" + m.group(2)
            # Already uses config: skip
            if "VAULT_CREDENTIAL_PREFIX" in m.group(0):
                return m.group(0)
            logger.info(f"[POST] Normalised vault key: {full_key!r} → VAULT_CREDENTIAL_PREFIX_{m.group(2)}")
            return _normalize_vault_key(m)
        code = _vault_key_pat.sub(_maybe_replace, code)
        # Ensure VAULT_CREDENTIAL_PREFIX is imported
        if "VAULT_CREDENTIAL_PREFIX" in code and "VAULT_CREDENTIAL_PREFIX" not in code.split("async def run")[0]:
            # Only touch the import line, not occurrences inside run()
            code = re.sub(
                r'(from src\.utils\.config import[^\n]*TIMEOUT[^\n]*)',
                lambda m: m.group(0) + ', VAULT_CREDENTIAL_PREFIX'
                    if 'VAULT_CREDENTIAL_PREFIX' not in m.group(0) else m.group(0),
                code, count=1,
            )
            # Ensure newline after the modified import line
            code = re.sub(
                r'(from src\.utils\.config import[^\n]*VAULT_CREDENTIAL_PREFIX)([^\n])',
                r'\1\n\2',
                code, count=1,
            )

    # ── 11: Strip redundant manual login steps after login() ──────────────────
    # If the script calls login(page) AND then also manually does
    # page.fill(<login_sel>, ...) + page.click(<login_sel>),
    # the manual steps are redundant and must be removed.
    if "await login(page)" in code:
        lines = code.splitlines(keepends=True)
        past_login, out_lines = False, []
        i = 0
        while i < len(lines):
            line = lines[i]
            if "await login(page)" in line:
                past_login = True
                out_lines.append(line)
                i += 1
                continue
            if past_login:
                stripped = line.strip()
                # Detect manual login: page.fill/page.click targeting a login-field selector
                is_login_fill = bool(re.search(
                    r'(?:resilient_fill|page\.fill)\s*\(\s*(?:page\s*,\s*)?(["\'][^"\']+["\'])',
                    stripped
                )) and bool(re.search(
                    r'(?:resilient_fill|page\.fill)\s*\(\s*(?:page\s*,\s*)?(["\'][^"\']+["\'])',
                    stripped
                ) and _is_login_selector(
                    re.search(r'(?:resilient_fill|page\.fill)\s*\(\s*(?:page\s*,\s*)?(["\'][^"\']+["\'])', stripped).group(1).strip("'\"")
                ))
                is_login_click = bool(re.search(
                    r'(?:resilient_click|page\.click)\s*\(\s*(?:page\s*,\s*)?(["\'][^"\']+["\'])',
                    stripped
                )) and _is_login_selector(
                    re.search(r'(?:resilient_click|page\.click)\s*\(\s*(?:page\s*,\s*)?(["\'][^"\']+["\'])', stripped).group(1).strip("'\"")
                    if re.search(r'(?:resilient_click|page\.click)\s*\(\s*(?:page\s*,\s*)?(["\'][^"\']+["\'])', stripped)
                    else ""
                )
                # Also catch login button patterns
                is_login_submit = bool(re.search(
                    r'(?:resilient_click|page\.click)\s*\(\s*(?:page\s*,\s*)?["\']'
                    r'(?:#login(?:button|btn|submit)|button\[type=["\']submit["\']|input\[type=["\']submit["\'])',
                    stripped
                ))
                if is_login_fill or is_login_click or is_login_submit:
                    logger.info(f"[POST] Stripped redundant manual login step: {stripped[:80]}")
                    i += 1
                    continue
            out_lines.append(line)
            i += 1
        code = "".join(out_lines)

    # ── 12: Replace hardcoded integer timeouts ───────────────────────────────
    # LLM sometimes emits wait_for_selector(..., timeout=30000) instead of
    # using TIMEOUT_TABLE_WAIT_MS. Replace all such raw integers.
    def _fix_timeout(m):
        val = int(m.group(1))
        # Map to the right config constant by magnitude
        if val >= 20000:
            return f'timeout=TIMEOUT_TABLE_WAIT_MS'
        elif val >= 5000:
            return f'timeout=TIMEOUT_FILL_MS'
        else:
            return m.group(0)  # leave small timeouts alone
    code = re.sub(r'\btimeout=(\d{4,})\b', _fix_timeout, code)
    # Ensure the constants are imported if we just introduced them
    for const in ("TIMEOUT_TABLE_WAIT_MS", "TIMEOUT_FILL_MS"):
        if const in code and const not in code.split("async def run")[0]:
            code = re.sub(
                r'(from src\.utils\.config import[^\n]*)',
                lambda m, c=const: m.group(0) + f', {c}'
                    if c not in m.group(0) else m.group(0),
                code, count=1,
            )

    # ── 13: Inject missing process URL navigation ─────────────────────────────
    # If history has a target URL (with query params, not a login page) and the
    # script never navigates there, inject a goto BEFORE the first table wait.
    if process_url and "await login(page)" in code:
        # Check if the process URL is already in a goto call
        if process_url not in code:
            # Find the first wait_for_selector AFTER maybe_login and insert goto before it
            lines = code.splitlines(keepends=True)
            past_login, injected = False, False
            new_lines = []
            for line in lines:
                if "await login(page)" in line:
                    past_login = True
                if (past_login and not injected
                        and "wait_for_selector" in line
                        and "await page.wait_for_selector" in line):
                    indent = re.match(r'^(\s*)', line).group(1)
                    new_lines.append(f'\n{indent}# Navigate to process target URL\n')
                    new_lines.append(f'{indent}await page.goto({json.dumps(process_url)})\n')
                    logger.info(f"[POST] Injected missing process URL navigation: {process_url}")
                    injected = True
                new_lines.append(line)
            if injected:
                code = "".join(new_lines)

    # ── 14: Replace login/auth redirect URLs with base app URL ────────────────
    # When the LLM uses a login redirect URL (e.g. /login?RFA=..., /otdsws/...,
    # /signin?...) as the first page.goto, replace it with the base app URL.
    # The base URL auto-redirects to login if needed, and maybe_login handles it.
    _login_url_kws = ("login", "otdsws", "signin", "auth/login", "auth/realms")

    def _is_login_url(url: str) -> bool:
        return any(kw in url.lower() for kw in _login_url_kws)

    # Find a base URL from the process_url or infer from the login URL
    goto_matches = list(re.finditer(r'await page\.goto\((["\'])([^"\']+)\1\)', code))
    if goto_matches:
        first_goto_url = goto_matches[0].group(2)
        if _is_login_url(first_goto_url):
            # Try to derive base URL: strip everything after the path
            # e.g. http://host:port/otdsws/login?... → http://host:port/OTCS/cs.exe
            # Use process_url if available, otherwise strip query params from first non-login URL
            replacement_url = None
            if process_url and not _is_login_url(process_url):
                # Use the base portion: scheme+host+path (no query)
                from urllib.parse import urlparse, urlunparse
                parsed = urlparse(process_url)
                replacement_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
            if not replacement_url:
                # Fallback: any non-login URL in the goto list
                for gm in goto_matches[1:]:
                    if not _is_login_url(gm.group(2)):
                        replacement_url = gm.group(2)
                        break
            if replacement_url:
                old_goto = goto_matches[0].group(0)
                new_goto = f'await page.goto({json.dumps(replacement_url)})'
                code = code.replace(old_goto, new_goto, 1)
                logger.info(f"[POST] Replaced login URL with base URL: {replacement_url}")

    # ── 15: Inject wait_for_load_state after page.goto and maybe_login ────────
    # Enterprise apps need networkidle waits after navigation and login.
    lines = code.splitlines(keepends=True)
    injected_lines = []
    for i, line in enumerate(lines):
        injected_lines.append(line)
        stripped = line.strip()
        # Check next line (if exists) to see if wait_for_load_state already follows
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        already_has_wait = "wait_for_load_state" in next_line

        if already_has_wait:
            continue

        indent = re.match(r'^(\s*)', line).group(1)

        # After page.goto(...) — inject domcontentloaded wait
        if re.search(r'await page\.goto\(', stripped) and not stripped.startswith("#"):
            injected_lines.append(f'{indent}await page.wait_for_load_state("domcontentloaded")\n')
            logger.info("[POST] Injected wait_for_load_state after page.goto")

        # After maybe_login(page) — inject networkidle wait
        if "await maybe_login(page)" in stripped:
            injected_lines.append(f'{indent}await page.wait_for_load_state("networkidle", timeout=TIMEOUT_NETWORKIDLE_MS)\n')
            logger.info("[POST] Injected wait_for_load_state after maybe_login")

    code = "".join(injected_lines)

    # Ensure TIMEOUT_NETWORKIDLE_MS is imported if we just introduced it
    if "TIMEOUT_NETWORKIDLE_MS" in code and "TIMEOUT_NETWORKIDLE_MS" not in code.split("async def run")[0]:
        code = re.sub(
            r'(from src\.utils\.config import[^\n]*)',
            lambda m: m.group(0) + ', TIMEOUT_NETWORKIDLE_MS'
                if 'TIMEOUT_NETWORKIDLE_MS' not in m.group(0) else m.group(0),
            code, count=1,
        )

    # ── Quality gate ─────────────────────────────────────────────────────────
    issues = []
    if re.search(r'resilient_extract_table\(\s*page\s*,\s*["\']table["\']', code):
        issues.append("resilient_extract_table still uses bare 'table' selector")
    if re.search(r'page\.wait_for_selector\(\s*["\']table["\']', code):
        issues.append("wait_for_selector still uses bare 'table' selector")
    if re.search(r'table_selector\s*=\s*["\']table["\']', code):
        issues.append("table_selector still assigned literal 'table'")
    if re.search(r'page\.locator\(\s*["\']tr["\']', code):
        issues.append("global page.locator('tr') without table scoping")
    # Check for login-field selectors used in extraction/wait calls
    for fn in ("extract_table", "wait_for_selector"):
        m = re.search(rf'{fn}\(\s*(?:page\s*,\s*)?(["\'][^"\']+["\'])', code)
        if m and _is_login_selector(m.group(1).strip("'\"")):
            issues.append(f"{fn} still uses a login-field selector: {m.group(1)!r}")
    # Warn if script still imports from project internals
    if any("from src.utils" in line and "vault" not in line for line in code.splitlines()):
        issues.append("script still imports from src.utils — should be self-contained")

    if issues:
        warning = "# ⚠️  POST-PROCESSING WARNINGS:\n" + "\n".join(f"# - {i}" for i in issues) + "\n\n"
        logger.warning("[QUALITY GATE]\n  " + "\n  ".join(issues))
        if selector is None and any("selector" in i for i in issues):
            raise RuntimeError(f"[QUALITY GATE] Cannot auto-fix — no DOM selector resolved. Issues: {issues}")
        code = warning + code

    return code

def generate_script(history_path: str, output_path: str = None, model_name: str = None, provider: str = None, objective: str = None, mandatory_history_path: str = None):
    """
    Generate script from history file.
    
    Args:
        history_path: Path to the process history JSON file
        output_path: Where to save the generated script
        model_name: LLM model to use (default: LLM_MODEL_NAME from config/env)
        provider: LLM provider (default: LLM_PROVIDER from config/env)
        objective: The task objective
        mandatory_history_path: Optional path to mandatory/login history to prepend
    """
    from src.utils.config import SCRIPT_GEN_MODEL, SCRIPT_GEN_PROVIDER
    if model_name is None:
        model_name = SCRIPT_GEN_MODEL
    if provider is None:
        provider = SCRIPT_GEN_PROVIDER
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
            
        run_id = Path(history_path).stem
        logger.info(f"Loaded history for {run_id}")
        
        # If mandatory history is provided, merge it with process history
        # This ensures non-mandatory process scripts can run standalone with login
        if mandatory_history_path and os.path.exists(mandatory_history_path):
            try:
                with open(mandatory_history_path, 'r', encoding='utf-8') as f:
                    mandatory_data = json.load(f)
                
                mandatory_steps = mandatory_data.get('history', [])
                process_steps = history_data.get('history', [])
                
                # Prepend mandatory steps to process steps
                history_data['history'] = mandatory_steps + process_steps
                
                # Include mandatory objective in the combined objective
                mandatory_objective = mandatory_data.get('task', '')
                if mandatory_objective:
                    objective = f"PREREQUISITES (Login/Session Setup):\n{mandatory_objective}\n\nMAIN TASK:\n{objective or history_data.get('task', '')}"
                
                logger.info(f"Merged {len(mandatory_steps)} mandatory steps with {len(process_steps)} process steps")
            except Exception as merge_err:
                logger.warning(f"Failed to merge mandatory history: {merge_err}. Proceeding with process-only history.")
        
        # Prepare content
        cleaned_history = clean_history_json(history_data, objective=objective, history_path=history_path)
        logger.info(f"Cleaned History Context (First 500 chars): {cleaned_history[:500]}...")
        
        # Save a debug copy of the cleaned history for inspection
        try:
            debug_path = history_path.replace('.json', '_cleaned.json')
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_history)
            logger.info(f"Debug cleaned history saved to: {debug_path}")
        except: pass

        # Prepare user prompt
        # Initialize LLM
        logger.info(f"Initializing LLM: {provider}/{model_name} (num_predict={SCRIPT_GEN_NUM_PREDICT}, num_ctx={SCRIPT_GEN_NUM_CTX})")
        llm = get_llm_model(
            provider=provider, 
            model_name=model_name, 
            temperature=SCRIPT_GEN_TEMPERATURE,
            num_predict=SCRIPT_GEN_NUM_PREDICT,
            num_ctx=SCRIPT_GEN_NUM_CTX,
        )
        
        # Create Prompt
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Task Objective: {objective or 'Complete the automation task'}\n\nGenerate a Playwright script for the following execution history (Run ID: {run_id}):\n\n{cleaned_history}")
        ]
        
        # Generate with OOM retry fallback (spec §3.G)
        logger.info("Sending request to LLM (this may take a minute)...")
        # Build fallback chain: primary model first, then config-defined fallbacks
        _oom_fallback_models = [model_name] + OOM_FALLBACK_MODELS
        # ctx sizes: primary uses SCRIPT_GEN_NUM_CTX, fallbacks use OOM_FALLBACK_CTX values
        _oom_fallback_ctx    = [SCRIPT_GEN_NUM_CTX] + list(OOM_FALLBACK_CTX)
        response = None
        last_err = None
        for _attempt, (_fb_model, _fb_ctx) in enumerate(zip(_oom_fallback_models, _oom_fallback_ctx)):
            try:
                if _attempt > 0:
                    logger.warning(f"[OOM] Retrying with fallback model={_fb_model}, num_ctx={_fb_ctx} (attempt {_attempt+1})")
                    llm = get_llm_model(
                        provider=provider,
                        model_name=_fb_model,
                        temperature=SCRIPT_GEN_TEMPERATURE,
                        num_predict=OOM_RETRY_NUM_PREDICT,
                        num_ctx=_fb_ctx,
                    )
                response = llm.invoke(messages)
                break
            except Exception as invoke_err:
                last_err = invoke_err
                err_lower = str(invoke_err).lower()
                if any(kw in err_lower for kw in ("cuda out of memory", "oom", "out of memory", "failed to allocate")):
                    logger.warning(f"[OOM] CUDA OOM detected on attempt {_attempt+1}: {invoke_err}")
                    continue
                raise  # Non-OOM errors are re-raised immediately
        if response is None:
            raise RuntimeError(f"[OOM] All fallback models exhausted. Last error: {last_err}")
        code = response.content
        
        # Robust Extraction (Handle extra text, markdown blocks, etc.)
        import re
        code = response.content
        
        # Try to find the largest python code block
        code_blocks = re.findall(r'```python\n(.*?)\n```', code, re.DOTALL)
        if not code_blocks:
            code_blocks = re.findall(r'```\n(.*?)\n```', code, re.DOTALL)
            
        if code_blocks:
            # Use the longest block (highest probability of being the actual script)
            code = max(code_blocks, key=len).strip()
        else:
            # Fallback: strip markers manually but keep the whole content if no blocks found
            code = code.replace("```python", "").replace("```", "").strip()

        # Enforce required DOM scoping, columns, and login hook
        # May raise RuntimeError if quality gate cannot auto-fix
        try:
            code = _postprocess_generated_code(code, cleaned_history)
        except RuntimeError as qg_err:
            logger.error(f"[QUALITY GATE] Refusing to save script: {qg_err}")
            raise
        
        # Final safety check: if it STILL doesn't look like code, log it
        if "async def run" not in code:
            logger.warning("Generated content might not be a valid script! Checking for generic text...")
            # If there's an 'async filter' or similar, we might have a nested structure
            
        # Verify imports (Basic check)
        if "from playwright.async_api" not in code:
            logger.warning("Generated code might be missing imports!")

        # Determine output path
        if not output_path:
            output_path = str(Path(history_path).parent / f"{run_id}_LLM.py")
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
            
        logger.info(f"✅ Script saved to: {output_path}")
        return output_path, code

    except Exception as e:
        logger.error(f"Failed to generate script: {e}")
        raise

if __name__ == "__main__":
    from src.utils.config import SCRIPT_GEN_MODEL, SCRIPT_GEN_PROVIDER
    parser = argparse.ArgumentParser(description="Generate Playwright script via LLM")
    parser.add_argument("history_file", help="Path to agent history JSON file")
    parser.add_argument("--output", help="Output path for .py file")
    parser.add_argument("--provider", default=SCRIPT_GEN_PROVIDER, help=f"LLM Provider (default: {SCRIPT_GEN_PROVIDER}, set via SCRIPT_GEN_PROVIDER env)")
    parser.add_argument("--model",    default=SCRIPT_GEN_MODEL,    help=f"Model name (default: {SCRIPT_GEN_MODEL}, set via SCRIPT_GEN_MODEL env)")

    args = parser.parse_args()
    generate_script(args.history_file, args.output, args.model, args.provider)