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
    TIMEOUT_TABLE_WAIT_MS,
)
from langchain_core.messages import SystemMessage, HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _build_system_prompt() -> str:
    """Build the LLM system prompt. All runtime helpers live in script_helpers.py —
    the LLM must only write the async def run() body."""

    pw_sel          = LOGIN_PASS_SELECTORS[0] if LOGIN_PASS_SELECTORS else "input[type='password']"
    vault_prefix    = VAULT_CREDENTIAL_PREFIX
    table_wait_ms   = TIMEOUT_TABLE_WAIT_MS
    file_chooser_ms = TIMEOUT_FILE_CHOOSER_MS

    # Minimal script header the LLM should reproduce verbatim
    script_header = '''\
import asyncio
import sys
import os
from playwright.async_api import async_playwright

# Path bootstrap — keeps the script runnable from any working directory
current_dir = os.getcwd()
if "web-ui" not in current_dir and os.path.exists(os.path.join(current_dir, "web-ui")):
    sys.path.append(os.path.join(current_dir, "web-ui"))
else:
    sys.path.append(current_dir)

from src.utils.script_helpers import (
    get_secret, find_in_frames,
    resilient_fill, resilient_click, click_and_upload_file,
    maybe_login, resilient_extract_table, generate_report,
    check_certificate,
)
from src.utils.config import TIMEOUT_TABLE_WAIT_MS, TIMEOUT_FILL_MS

# Reporting state
steps = []
script_dir = os.path.dirname(os.path.abspath(__file__))

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        context = await browser.new_context(no_viewport=True)
        page = await context.new_page()

        async def capture_step(action, output):
            idx = len(steps) + 1
            shot_path = os.path.join(script_dir, f"shot_{idx}.png")
            await page.screenshot(path=shot_path)
            steps.append({"action": action, "output": output, "shot": f"shot_{idx}.png"})

        # [START OF YOUR LOGIC]
        pass

if __name__ == "__main__":
    asyncio.run(run())
'''

    return f'''You are an expert Playwright automation engineer. Generate a MINIMAL, CORRECT Python script from the execution history below.

### ✅ WHAT TO GENERATE
Only write `async def run()` and the standard header shown in the STRUCTURE section.
Do NOT redefine any helper functions — they all come pre-built from `src.utils.script_helpers`.

### 🛡️ GOLDEN RULES
1.  **Helpers are pre-built** — NEVER redefine `get_secret`, `find_in_frames`, `resilient_fill`,
    `resilient_click`, `click_and_upload_file`, `maybe_login`, `resilient_extract_table`,
    `generate_report`, or `check_certificate`. Import and call them; do not copy their implementation.
2.  **Vault** — Use `await get_secret("KEY_USERNAME")` / `"KEY_PASSWORD"`. Never hardcode credentials.
    The vault prefix is currently `"{vault_prefix}"`.
3.  **Login hook** — Call `await maybe_login(page)` once, right after the FIRST `page.goto(...)`.
    Do NOT repeat it after subsequent navigations — the session persists.
4.  **Table extraction per page**:
    - Use `await resilient_extract_table(page, table_selector, required_columns)`.
    - `table_selector` MUST come from the NAVIGATION PLAN for THAT specific page.
    - If no selector is known for a page, use `await page.wait_for_load_state("networkidle")`
      and pass `"table"` to `resilient_extract_table` — it will auto-discover the best table.
    - `required_columns` MUST be EXACTLY the columns listed in the NAVIGATION PLAN for THAT table.
      NEVER add, remove, or rename columns. NEVER add "Server". NEVER blend columns from two tables.
5.  **One selector per page, one call per table** — Every `page.goto(url)` takes you to a
    DIFFERENT page with its own DOM. NEVER reuse the same selector across pages.
    If a page has MULTIPLE tables (listed in the NAVIGATION PLAN), call `resilient_extract_table`
    ONCE per table, each with its own selector and EXACT column list from the plan:
    ```python
    # Page B has two tables — each uses ONLY ITS OWN columns from the NAVIGATION PLAN
    table_a = await resilient_extract_table(page, "#summary table",  ["ID","Status"])
    table_b = await resilient_extract_table(page, "#details table",  ["Metric","Value"])
    # WRONG — never add "Server" or copy columns from another table:
    # table_b = await resilient_extract_table(page, "#details table", ["ID","Status","Server"])
    ```
6.  **Selectors** — Use ONLY selectors from the NAVIGATION PLAN or SELECTOR REFERENCE TABLE.
    NEVER invent IDs like `"#view_1011"`. NEVER use `nth-child` or `html > body > table`.
    If no selector is listed for a wait, use `await page.wait_for_load_state("networkidle")`.
7.  **Action/Menu hotspots** — NEVER use `"#x1"`, `"#x2"` etc. (row-position IDs that change).
    Always use `page.evaluate()` to find a hotspot next to the named row:
    ```python
    await page.evaluate(\'\'\'(text) => {{
        const link = [...document.querySelectorAll("a")].find(a => a.textContent.trim() === text);
        if (link) {{
            const td = link.closest("td");
            const hotspot = td && td.querySelector("a[class*=\'hotspot\'], a[title*=\'Action\']");
            if (hotspot) hotspot.click();
        }}
    }}\'\'\', "ItemNameHere")
    ```
8.  **Configure / action pages** — When the task says "configure" or "save settings", click the
    named link then look for `input[value="Save"]` or `button:has-text("Save")`.
    NEVER click login-field selectors like `input[name="username"]` during a configure flow.
9.  **Uploads** — Use `await click_and_upload_file(page, label, path)` with `TIMEOUT_FILE_CHOOSER_MS`
    (currently {file_chooser_ms}ms).
10. **Loops** — ALWAYS `elements = await locator.all()` then `for el in elements:`. Never `async for`.
11. **Comprehensive Reporting** — ALWAYS `print()` extracted data to the terminal.
    Call `await capture_step(action, output)` after every key interaction.
    Combine ALL steps and ALL extracted tables into a SINGLE `generate_report` call at the end.
    The report renders Markdown natively:
    - `# Heading` → Word Heading
    - `| col | col |` → Word Table
    - `![alt](path.png)` → Embedded Word Image
    NEVER use a bare loop variable in the report — always iterate over the collected list.
12. **No noise** — Skip failed/retried actions. Only generate the successful path.
13. **Imports** — Only import what `run()` actually uses beyond the standard header.
14. **Never invent element IDs** — If an ID is not in the NAVIGATION PLAN, don't use it.
    Use `wait_for_load_state("networkidle")` as the wait instead.
15. **Certificate / security checks** — NEVER click the browser lock icon.
    Instead call `await check_certificate(page)`.
16. **Indentation** — Use 4 spaces. NEVER misalign triple-quote closings.
17. **Identity & Status Logic** — In reports, use
    `ident = r.get("Name") or r.get("Identity") or next(iter(r.values()), "Unknown")`.
    For status: `stat = r.get("Status") or r.get("Running") or r.get("State") or "N/A"`.
18. **Triple Quotes for Reports** — ALWAYS use `f\'\'\'...\'\'\'` for multi-line report blocks.
19. **Multi-Table Reporting** — Each extracted table MUST have its own dedicated Markdown
    section (`### Table Name`) and Markdown table in the final report. Do NOT merge them.
20. **Robust Interactions** — For hotspots/dropdowns use broad selectors:
    `"a[class*=\'hotspot\'], a[title*=\'Action\'], .dropdown-toggle, [aria-haspopup]"`.
21. **Loop Reporting** — Capture a screenshot inside interaction loops for the first few rows.
22. **Selector Resilience** — For OTCS browse views, prefer `#browseViewCoreTable` or specific
    table IDs found in the NAVIGATION PLAN.

### 📝 STRUCTURE
Reproduce this header exactly, then write only `async def run()`.

For tasks that navigate to **multiple pages** (the common case), follow this exact pattern:

```python
{script_header}
```

#### Multi-page / Multi-table example (READ THIS CAREFULLY):
```python
async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        context = await browser.new_context(no_viewport=True)
        page = await context.new_page()

        async def capture_step(action, output):
            idx = len(steps) + 1
            shot_path = os.path.join(script_dir, f"shot_{{idx}}.png")
            await page.screenshot(path=shot_path)
            steps.append({{"action": action, "output": output, "shot": f"shot_{{idx}}.png"}})

        # ── Step 1: Login ──────────────────────────────────────────────────
        await page.goto("http://host/login", timeout=60000)
        await maybe_login(page)
        await capture_step("Login", "Logged in successfully")

        # ── Step 2: Page A — may have TWO tables ───────────────────────────
        await page.goto("http://host/page-a", timeout=60000)
        await page.wait_for_load_state("networkidle")
        # Table 1 on this page (use exact selector from NAVIGATION PLAN)
        node_rows = await resilient_extract_table(page, "table", ["NodeID", "Name", "Status"])
        print(f"[*] Page A / Table 1: {{len(node_rows)}} rows")
        for r in node_rows: print(f"  {{r}}")
        await capture_step("Extract Node Table", f"{{len(node_rows)}} nodes found")

        # Table 2 on the same page (different selector from NAVIGATION PLAN)
        metric_rows = await resilient_extract_table(page, "#metrics-panel table", ["Metric","Value"])
        print(f"[*] Page A / Table 2: {{len(metric_rows)}} rows")
        await capture_step("Extract Metrics", f"{{len(metric_rows)}} metrics found")

        not_healthy = [r for r in node_rows if r.get("Status","").lower() != "healthy"]
        if not_healthy:
            await resilient_click(page, "a:has-text(\'Batch Heal\')")
            await page.wait_for_load_state("networkidle")
            await capture_step("Heal Nodes", f"Applied fixes to {{len(not_healthy)}} nodes")

        # ── Step 3: Page B — known selector from NAVIGATION PLAN ───────────
        await page.goto("http://host/page-b", timeout=60000)
        await page.wait_for_selector("#admin-servers table", timeout=TIMEOUT_TABLE_WAIT_MS)
        admin_rows = await resilient_extract_table(page, "#admin-servers table", ["Name","Status","Errors"])
        print(f"[*] Page B: {{len(admin_rows)}} admin servers")
        for r in admin_rows: print(f"  {{r}}")
        await capture_step("Admin Servers", f"{{len(admin_rows)}} servers found")

        # Open function menu for a named row — NEVER use #x1/#x2
        for r in admin_rows:
            sname = r.get("Server") or r.get("Name") or ""
            if sname:
                await page.evaluate(\'\'\'(text) => {{
                    const link = [...document.querySelectorAll("a")].find(a => a.textContent.trim() === text);
                    if (link) {{
                        const td = link.closest("td");
                        const h = td && td.querySelector("a[class*=\'hotspot\']");
                        if(h) h.click();
                    }}
                }}\'\'\', sname)
                await page.wait_for_timeout(500)
                await capture_step(f"Menu: {{sname}}", f"Opened function menu for {{sname}}")

        # ── Step 4: Single consolidated report ─────────────────────────────
        summary = \'\'\'### Execution Summary
| Step | Action | Output |
| :--- | :--- | :--- |
\'\'\'
        for i, s in enumerate(steps, 1):
            summary += f\'\'\'| {{i}} | {{s[\'action\']}} | {{s[\'output\']}} |\\n\'\'\'

        node_section = \'\'\'### Node Status (Page A — Table 1)
| Identity | Status |
| :--- | :--- |
\'\'\'
        for r in node_rows:
            ident = r.get("NodeID") or r.get("Name") or next(iter(r.values()), "Unknown")
            stat  = r.get("Status") or r.get("State") or "N/A"
            node_section += f\'\'\'| {{ident}} | {{stat}} |\\n\'\'\'

        metric_section = \'\'\'### Metrics (Page A — Table 2)
| Metric | Value |
| :--- | :--- |
\'\'\'
        for r in metric_rows:
            m = r.get("Metric") or next(iter(r.values()), "—")
            v = r.get("Value") or "N/A"
            metric_section += f\'\'\'| {{m}} | {{v}} |\\n\'\'\'

        admin_section = \'\'\'### Admin Servers (Page B)
| Server | Status | Errors |
| :--- | :--- | :--- |
\'\'\'
        for r in admin_rows:
            ident = r.get("Server") or r.get("Name") or next(iter(r.values()), "Unknown")
            stat  = r.get("Status") or "N/A"
            errs  = r.get("Errors") or "0"
            admin_section += f\'\'\'| {{ident}} | {{stat}} | {{errs}} |\\n\'\'\'

        details = \'\'\'## Step Screenshots\\n\'\'\'
        for i, s in enumerate(steps, 1):
            details += f\'\'\'### Step {{i}}: {{s[\'action\']}}\\n![Step {{i}}]({{s[\'shot\']}})\\n\'\'\'

        full_report = f\'\'\'# Automation Report\\n\\n{{summary}}\\n{{node_section}}\\n{{metric_section}}\\n{{admin_section}}\\n{{details}}\'\'\'
        await generate_report("Consolidated Report", True, output=full_report, print_terminal=False)
        await browser.close()
```

The header imports EVERYTHING from `script_helpers`. Only add extra imports if `run()` needs them.
'''

SYSTEM_PROMPT = _build_system_prompt()


# ── Known OTCS page selectors & columns ─────────────────────────────────────
# Maps URL query-string fragment → { selector, columns }.
# Used in three places:
#   1. _build_page_table_registry() — fallback when agent didn't capture tables_on_page
#   2. _postprocess_generated_code() Rule 16 — replace wrong/bare selectors per page
#   3. Navigation plan builder — inject exact selector+columns into LLM prompt
#
# Format: { "url_fragment": {"selector": "<css>", "columns": ["Col1", ...]} }
# The fragment is matched with `fragment in url` so partial matches work.
_KNOWN_PAGE_SELECTORS: dict[str, dict] = {
    # Note: AgentStatus and other pages are now 100% dynamic via Source 2b/2c (registry).
    # No hardcoded selectors here to ensure system works for any website.
}


def _known_page_info(url: str) -> dict | None:
    """Return {"selector": ..., "columns": ...} for a well-known OTCS page, or None."""
    if not url:
        return None
    for frag, info in _KNOWN_PAGE_SELECTORS.items():
        if frag in url:
            return info
    return None


def _derive_table_selector(selectors: dict, attrs: dict, comp: dict, comp_selectors: dict) -> str | None:
    """Derive a stable CSS table-container selector from dom_context captured by the agent.

    Strategy (in priority order):
    1. XPath contains @id="something" → CSS = #something table
       e.g. //*[@id="admin-servers"]/tbody/tr[2]/td[2]/a  → "#admin-servers table"
    2. preferred/css selector already contains a table ref → use as-is
    3. Ancestor ID from comprehensive capture → #id table
    4. None — caller should fall back to bare "table" with page-wide search

    This uses data the browser agent ACTUALLY captured during the live run,
    so it's always specific to the real page structure, never hardcoded.
    """
    _ignore_ids = {"otds_username", "otds_password", "username", "password",
                   "loginbutton", "login", "header", "nav", "sidebar", "footer",
                   "toolbar", "menu", "topnav"}

    # 1. XPath — extract @id from path like //*[@id="admin-servers"]/...
    xpath = selectors.get('xpath') or comp_selectors.get('xpath', '')
    if xpath:
        import re as _re
        m = _re.search(r'/\*\[@id=["\']([^"\']+)["\']\]', xpath)
        if m:
            container_id = m.group(1)
            if container_id.lower() not in _ignore_ids:
                return f'#{container_id} table'

    # 2. preferred/css selector that already scopes a table
    for key in ('preferred', 'css'):
        val = selectors.get(key) or comp_selectors.get(key, '')
        if val and isinstance(val, str):
            v = val.strip()
            if 'table' in v.lower() and v.lower() not in ('table',):
                return v
            # e.g. "#admin-servers" → "#admin-servers table"
            if v.startswith('#') and ' ' not in v:
                if v.lower().lstrip('#') not in _ignore_ids:
                    return f'{v} table'

    # 3. Ancestor IDs from comprehensive capture (parentChain)
    comp_data = comp if isinstance(comp, dict) else {}
    for parent in comp_data.get('parentChain', []):
        pid = parent.get('id') if isinstance(parent, dict) else None
        if pid and pid.lower() not in _ignore_ids:
            return f'#{pid} table'

    # 4. attributes.id of the element itself (only useful if it's a table/container)
    el_id = attrs.get('id') or ''
    if el_id and el_id.lower() not in _ignore_ids:
        tag = selectors.get('tag', '').lower()
        if tag in ('table', 'tbody', 'div', 'section', 'article', ''):
            return f'#{el_id} table'

    return None


def clean_history_json(history_data: Dict[str, Any], objective: str = None) -> str:
    """Optimize JSON payload by pruning redundant actions, failed attempts, and duplicate navigations."""
    def columns_from_results(results):
        """Parse leaf-row keys from extracted_content fenced JSON in agent results.
        
        Strategy (per spec §3.A):
          1. Parse fenced ```json``` blocks from extracted_content.
          2. Walk nested objects/lists and collect all leaf-dict keys (dict where all
             values are scalars – i.e. an actual data row).
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
                    if depth > 10:
                        return
                    if isinstance(obj, dict):
                        # Leaf row: dict where ALL values are scalars
                        if obj and all(not isinstance(v, (dict, list)) for v in obj.values()):
                            for k in obj.keys():
                                if k not in seen_cols:
                                    seen_cols.add(k)
                                    cols.append(k)
                        else:
                            for v in obj.values():
                                walk(v, depth + 1)
                    elif isinstance(obj, list):
                        for v in obj:
                            walk(v, depth + 1)

                walk(data)
                if cols:
                    return cols
        return []
    simplified = []
    seen_urls = []

    def infer_required_columns(text: str) -> list[str]:
        """
        Infer required columns from the user's objective/goal without hardcoded domain terms.
        Strategy:
          1) Extract quoted phrases (single or double quotes).
          2) Extract list after keywords like "columns", "fields", "extract".
          3) Extract Title Case multi-word phrases as a fallback.
        """
        if not text:
            return []

        candidates = []

        # 1) Quoted phrases: "Status", 'Errors', etc.
        candidates += re.findall(r'"([^"]{2,})"', text)
        candidates += re.findall(r"'([^']{2,})'", text)

    # 2) List after delimiters (colon or dash), derived from the text itself
        for line in re.split(r'[\r\n]+', text):
            if ":" in line or " - " in line:
                tail = re.split(r'[:\-]\s*', line, maxsplit=1)[-1]
                parts = re.split(r',| and ', tail, flags=re.IGNORECASE)
                candidates += [p.strip() for p in parts if p.strip()]

        # 3) Title Case phrases (e.g., "Enterprise Update Distributor")
        candidates += re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)

        # Normalize and filter noise using dynamic stopwords from the text
        tokens = re.findall(r'[A-Za-z]+', text.lower())
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        common = sorted(freq, key=freq.get, reverse=True)[:15]
        stopwords = set(common)
        cleaned = []
        seen = set()
        for c in candidates:
            c = c.strip().strip(".:")
            if not c or c in stopwords:
                continue
            if len(c) < 2:
                continue
            if c not in seen:
                seen.add(c)
                cleaned.append(c)

        return cleaned
    
    for item in history_data.get('history', []):
        state = item.get('state', {})
        model_out = item.get('model_output', {})
        action = model_out.get('action', [])
        evaluation = model_out.get('current_state', {}).get('evaluation_previous_goal', '')
        
        current_url = state.get('url', '')
        if current_url:
            # Simple URL deduplication: if the URL is logically the same as the last one, mark as redundant
            # (ignoring minor query param diffs or typos that redirect)
            # FIX: compare full URLs (including query string), not just base.
            # OTCS pages all share the same base (cs.exe) but are DIFFERENT pages
            # when the query string differs (func=distributedagent vs func=ll&objtype=148).
            # Using split('?')[0] made every OTCS page look like a retry of the first.
            if seen_urls and seen_urls[-1] == current_url:
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
                    "selectors": {**selectors, **comp_selectors},
                }
                element_data["selectors"] = {k: v for k, v in element_data["selectors"].items() if v}

                rec_sel = comp.get('recommended_selector', {}).get('selector', '')
                if rec_sel:
                    element_data["recommended_selector"] = rec_sel
                    
                step_info['dom_elements'].append(element_data)

                # ── Derive table container selector from captured XPath/CSS ──────
                # The agent captures the XPath of every element it interacts with.
                # XPaths rooted at an ancestor with an id look like:
                #   //*[@id="admin-servers"]/tbody/tr[2]/td[2]/a
                # From this we can derive the stable CSS selector: #admin-servers table
                # We store this as table_selector_hint on the step so the postprocessor
                # and NAVIGATION PLAN can use the REAL selector from the live run,
                # not a hardcoded guess or a fragile scoring heuristic.
                if 'table_selector_hint' not in step_info:
                    _derived = _derive_table_selector(selectors, attrs, comp, comp_selectors)
                    if _derived:
                        step_info['table_selector_hint'] = _derived
                        logger.debug(f"[TABLE-HINT] Derived table selector: {_derived!r} from dom_context")

            if isinstance(act, dict) and "extract_content" in act:
                goal = act.get("extract_content", {}).get("goal", "") if isinstance(act.get("extract_content", {}), dict) else ""
                goal_text = f"{objective or ''} {goal}".lower()

                # ── GROUND-TRUTH TABLE REGISTRY ────────────────────────────────────
                # extract_page_content stores tables_on_page in extracted_content:
                #   [{selector: "#admin-servers table", headers: [...], row_count: N}, ...]
                #
                # MULTI-TABLE SUPPORT: We keep ALL tables found on this page (not just
                # the best one). Each table gets its own entry in tables_registry with
                # its exact selector, headers, and derived required columns.
                #
                # The navigation plan then lists every table per page so the LLM
                # generates one resilient_extract_table call per table — never merging
                # tables from different containers or missing a secondary table.
                _tables_on_page = []
                for res in results:
                    if not isinstance(res, dict):
                        continue
                    ec = res.get("extracted_content")
                    if isinstance(ec, dict):
                        tops = ec.get("tables_on_page")
                        if isinstance(tops, list) and tops:
                            _tables_on_page.extend(tops)
                        break  # only one extract_content result per step

                if _tables_on_page:
                    # Score each table by header overlap with goal + objective
                    def _tbl_score(tbl_entry):
                        hdrs = [h.lower() for h in (tbl_entry.get("headers") or [])]
                        score = sum(1 for h in hdrs if h in goal_text or
                                    any(h in w or w in h for w in goal_text.split()))
                        # Bonus for tables with more data rows (prefer data over layout)
                        score += min(tbl_entry.get("row_count", 0), 5) * 0.1
                        return score

                    # Sort all tables: best match first
                    scored_tables = sorted(_tables_on_page,
                                           key=_tbl_score, reverse=True)

                    # Build per-table registry entries
                    registry = []
                    for tbl in scored_tables:
                        tbl_sel  = tbl.get("selector", "")
                        tbl_hdrs = [h for h in (tbl.get("headers") or []) if h and h.strip()]
                        if not tbl_sel or not tbl_hdrs:
                            continue

                        # Derive required columns for this specific table
                        relevant = [h for h in tbl_hdrs
                                    if h.lower() in goal_text or
                                       any(w in h.lower() or h.lower() in w
                                           for w in goal_text.split())]
                        tbl_required = relevant if relevant else tbl_hdrs

                        registry.append({
                            "selector":  tbl_sel,
                            "headers":   tbl_hdrs,
                            "required":  tbl_required,
                            "row_count": tbl.get("row_count", 0),
                        })

                    if registry:
                        step_info["tables_registry"] = registry
                        # Backward-compat single-table fields (primary = highest score)
                        primary = registry[0]
                        step_info["table_selector_hint"] = primary["selector"]
                        step_info["exact_headers"]       = primary["headers"]
                        step_info["required_columns_hint"] = primary["required"]
                        logger.info(
                            f"[TABLE-REGISTRY] {len(registry)} table(s) captured: "
                            + ", ".join(
                                f"{t['selector']!r}({len(t['headers'])} cols,"
                                f"{t['row_count']} rows)"
                                for t in registry
                            )
                        )
                    else:
                        # tables_on_page was present but no usable entries
                        step_info["required_columns_hint"] = (
                            infer_required_columns(goal_text) or []
                        )

                else:
                    # ── Legacy path: no tables_on_page in history ─────────────────
                    # Fall back to parsing extracted_content JSON for column names
                    # and deriving the selector from dom_context XPath/CSS.
                    cols = columns_from_results(results)
                    if cols:
                        def _goal_match(col_name):
                            c_l = col_name.lower()
                            c_singular = c_l[:-1] if c_l.endswith("s") else c_l
                            c_plural   = c_l + "s" if not c_l.endswith("s") else c_l
                            return (c_l in goal_text or c_singular in goal_text
                                    or c_plural in goal_text)
                        filtered = [c for c in cols if _goal_match(c)]
                        required = filtered if filtered else cols

                    else:
                        required = infer_required_columns(f"{objective or ''} {goal}")
                    if required:
                        step_info["required_columns_hint"] = required


                 
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

    Priority order (highest → lowest):
      1. table_selector_hint + exact_headers  — ground-truth from tables_on_page capture
      2. table_selector_hint + required_columns_hint  — derived from XPath / dom_context
      3. best dom_element selector + required_columns_hint  — legacy heuristic
      4. _scan_element_data_files()  — element capture fallback

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
        rec = el.get("recommended_selector", "")
        if isinstance(rec, dict):
            rec = rec.get("selector", "")
        for key in SELECTOR_PRIORITY:
            val = rec if key == "recommended_selector" else selectors.get(key)
            if val and isinstance(val, str) and val.strip() and val.strip().lower() != "table":
                return val.strip()
        return None

    best_selector = None
    best_required = None
    target_url = None

    # Extract target URL
    for step in steps:
        if not isinstance(step, dict):
            continue
        url = step.get("url", "")
        if url and "?" in url and not any(
            kw in url.lower() for kw in ("login", "otdsws", "signin", "auth/login")
        ):
            target_url = url
            break
    if not target_url:
        for step in steps:
            if isinstance(step, dict):
                url = step.get("url", "")
                if url and not any(kw in url.lower() for kw in ("login", "otdsws", "signin")):
                    target_url = url
                    break

    # Pass 1A: ground-truth tables_registry (multi-table, highest priority)
    # Find the best single table across all pages — prefer ground-truth registry
    # If caller needs all tables per page, use _build_page_table_registry() instead.
    for step in steps:
        if not isinstance(step, dict):
            continue
        tbls = step.get("tables_registry")
        if tbls and isinstance(tbls, list):
            # Pick the table most relevant to the objective/task
            # (first entry is already scored as best match by clean_history_json)
            primary = tbls[0]
            sel  = primary.get("selector", "")
            hdrs = primary.get("headers") or []
            req  = primary.get("required") or hdrs
            if sel and hdrs:
                logger.info(
                    f"[TABLE-HINTS] Ground-truth registry (primary): "
                    f"selector={sel!r} headers={hdrs} "
                    f"({len(tbls)} total tables on this page)"
                )
                return sel, req, target_url

    # Pass 1B: single-table ground-truth fingerprint (table_selector_hint + exact_headers)
    for step in steps:
        if not isinstance(step, dict):
            continue
        sel  = step.get("table_selector_hint")
        hdrs = step.get("exact_headers")
        req  = step.get("required_columns_hint")
        if sel and hdrs:
            logger.info(
                f"[TABLE-HINTS] Ground-truth fingerprint: selector={sel!r} headers={hdrs}"
            )
            return sel, req or hdrs, target_url

    # Pass 2: table_selector_hint + required_columns_hint (derived path)
    for step in steps:
        if not isinstance(step, dict):
            continue
        required = step.get("required_columns_hint")
        selector = step.get("table_selector_hint")
        if not selector:
            for el in step.get("dom_elements", []) or []:
                sel = _best_selector_from_element(el)
                if sel:
                    selector = sel
                    break
        if required and selector:
            return selector, required, target_url
        if required and best_required is None:
            best_required = required
        if selector and best_selector is None:
            best_selector = selector

    # Pass 3: any step with at least a selector
    if best_selector or best_required:
        return best_selector, best_required, target_url

    # Pass 4: element capture data files
    elem_selector, elem_url = _scan_element_data_files(target_url)
    if elem_selector:
        logger.info(f"[TABLE HINTS] Using selector from element capture data: {elem_selector}")
        return elem_selector, best_required, target_url or elem_url

    return None, None, target_url


def _build_page_table_registry(cleaned_history_json: str, run_id: str = None) -> dict:
    """Build a full URL → [table_entry, ...] registry from cleaned history.

    This is the authoritative multi-table, multi-page data structure.
    It is consumed by the navigation plan builder to tell the LLM exactly
    which tables exist on each page, what their CSS selectors are, and what
    columns to extract — so generated scripts never have to guess.

    Data sources in priority order per page:
      1. tables_registry  — ground-truth list from tables_on_page JS capture
         (set by clean_history_json when extract_page_content fires)
      2. table_selector_hint + exact_headers / required_columns_hint
         — derived from XPath / dom_context (single-table fallback)
      3. Legacy required_columns_hint alone (no selector)

    Returns:
        dict mapping full URL → list of:
          {
            "selector":  str,   # CSS selector, e.g. "#admin-servers table"
            "headers":   list,  # exact live headers, e.g. ["Name","Status","Errors"]
            "required":  list,  # columns to pass to resilient_extract_table
            "row_count": int,   # data row count (from live DOM, 0 if unknown)
          }

        Pages with no tables at all are omitted.
    """
    try:
        data = json.loads(cleaned_history_json)
    except Exception:
        return {}

    steps = data.get("steps") if isinstance(data, dict) else data
    if not isinstance(steps, list):
        return {}

    # Skip login/auth pages — they never carry data tables we care about
    _LOGIN_KWS = ("login", "otdsws", "signin", "auth/login", "about:blank")

    registry: dict = {}   # url → [table_entry, ...]
    seen_selectors_per_url: dict = {}  # url → set of selectors already added
    all_step_urls: list = []  # every non-login URL seen (for Source 3 fallback)

    # Pre-scan snapshots once if run_id is provided
    snap_map: dict = {}   # url → path
    snap_cache: dict = {} # path → json_data
    if run_id:
        import time as _time
        _t0 = _time.time()
        snap_dir = Path(f"tmp/dom_snapshots/{run_id}")
        if snap_dir.exists():
            _files = list(snap_dir.glob("*.json"))
            logger.info(f"[PERF] Found {len(_files)} snapshots in {snap_dir}")
            # Assume filename order (timestamp/index) is roughly chronological
            # to avoid expensive stat() calls on 700+ files
            for _sp in sorted(_files, reverse=True): 
                try:
                    with open(_sp, "r", encoding="utf-8") as _sf:
                        _meta = _sf.read(3000) # Read enough for URL
                        _um = re.search(r'"url":\s*"([^"]+)"', _meta)
                        if _um:
                            _u = _um.group(1)
                            if _u not in snap_map:
                                snap_map[_u] = _sp
                except Exception:
                    pass
        logger.info(f"[PERF] Pre-scan took {_time.time() - _t0:.2f}s (mapped {len(snap_map)} URLs)")

    # Pre-scan element_data once
    elem_map: dict = {} # url → [table_entry, ...]
    _ed_dir = Path("tmp/element_data")
    if _ed_dir.exists():
        _ed_files = list(_ed_dir.glob("*.json"))
        for _ef in _ed_files:
            try:
                with open(_ef, "r", encoding="utf-8") as _f:
                    _ed = json.load(_f)
                _u = _ed.get("metadata", {}).get("page_url", "")
                if not _u: continue
                # Check if this element data belongs to a table
                _chain = _ed.get("parentChain", [])
                _tbl_p = next((p for p in _chain if p.get("tag") == "table"), None)
                if _tbl_p:
                    _sel = f"#{_tbl_p.get('id')} table" if _tbl_p.get("id") else "table"
                    # Try to get headers from textContent or context if possible
                    # (Usually element_data is for a specific cell, but we can infer headers)
                    _hdrs = [] # Fallback to empty
                    _entry = {"selector": _sel, "headers": _hdrs, "required": _hdrs, "row_count": 0}
                    if _u not in elem_map: elem_map[_u] = []
                    # Avoid duplicates
                    if not any(e["selector"] == _sel for e in elem_map[_u]):
                        elem_map[_u].append(_entry)
            except Exception:
                pass

    _t_loop = _time.time()
    for step in steps:
        if not isinstance(step, dict):
            continue
        url = step.get("url", "")
        if not url or any(kw in url.lower() for kw in _LOGIN_KWS):
            continue

        # Track every URL for the Source-3 fallback (even if this step has no table data)
        if url not in all_step_urls:
            all_step_urls.append(url)

        # ── Source 2b: Background DOM snapshot extraction ─────────────────────
        # When a step has no tables_registry AND no selector hint, scan the
        # saved dom_snapshot for this URL to extract all table headers.
        if "tables_registry" not in step and not step.get("table_selector_hint"):
            snap_path = snap_map.get(url)
            if not snap_path:
                # Fuzzy URL match
                for _uk, _up in snap_map.items():
                    if _uk in url or url in _uk:
                        snap_path = _up
                        break

            if snap_path:
                try:
                    # Use cache if already loaded
                    if snap_path in snap_cache:
                        _snap = snap_cache[snap_path]
                    else:
                        with open(snap_path, "r", encoding="utf-8") as _f:
                            _snap = json.load(_f)
                        snap_cache[snap_path] = _snap

                    _bg_tables = []
                    for _el in _snap.get("elements", []):
                        if (_el.get("identity", {}).get("tagName") or "").lower() == "table":
                            _html = _el.get("integrity", {}).get("outerHTML", "")
                            _sel  = _el.get("selector_provenance", {}).get("preferred", "")
                            _th_matches = re.findall(r'<th[^>]*>(.*?)</th>', _html, re.IGNORECASE | re.DOTALL)
                            _headers = []
                            import html as _pyhtml # redundant but safe
                            for _th in _th_matches:
                                _txt = re.sub(r'<[^>]+>', '', _th).strip()
                                _txt = _pyhtml.unescape(_txt)
                                _txt = re.sub(r'\s+', ' ', _txt).strip()
                                if _txt:
                                    _headers.append(_txt)
                            if _headers and _sel:
                                _bg_tables.append({
                                    "selector": _sel,
                                    "headers":  _headers,
                                    "required": _headers,
                                    "row_count": max(0, _html.lower().count("<tr") - 1),
                                })
                    if _bg_tables:
                        logger.info(
                            f"[TABLE-BG] {url}: Extracted {len(_bg_tables)} tables from {snap_path.name}"
                        )
                        _seen_this_run = set()
                        for _bt in _bg_tables:
                            _bsel = _bt["selector"]
                            # If selector is duplicate on this URL, we must differentiate or we lose data.
                            # We'll use the headers as part of the "seen" check to allow multi-table pages.
                            # BUT the generated script needs a unique selector.
                            _header_key = "|".join(_bt["headers"])
                            _full_key = f"{_bsel}::{_header_key}"
                            
                            _seen = seen_selectors_per_url.setdefault(url, set())
                            if _full_key not in _seen:
                                _seen.add(_full_key)
                                # If the selector itself was seen, but with different headers, 
                                # we have a duplicate-selector issue. Fall back to n-th table?
                                # For now, we'll just allow it in the registry; Rule 16/17 will handle fixing.
                                registry.setdefault(url, []).append(_bt)
                                logger.debug(f"  - Registered table: selector={_bsel!r}, headers={len(_bt['headers'])}")
                            else:
                                logger.debug(f"  - Skipped duplicate table (same selector+headers)")
                except Exception as _bg_err:
                    logger.warning(f"[TABLE-BG] Failed for {url}: {_bg_err}")

        # ── Source 2c: Element Data extraction ────────────────────────────────
        # If we have any tables for this URL from element_data (Source 2c),
        # especially if Source 1/2b missed them.
        _etbls = elem_map.get(url, [])
        if _etbls:
            _added = 0
            _seen = seen_selectors_per_url.setdefault(url, set())
            for _et in _etbls:
                if _et["selector"] not in _seen:
                    _seen.add(_et["selector"])
                    registry.setdefault(url, []).append(_et)
                    _added += 1
            if _added > 0:
                logger.debug(f"[TABLE-ELEM] {url}: added {_added} tables from element_data")

        tbls = step.get("tables_registry")  # ground-truth multi-table list
        if tbls and isinstance(tbls, list):
            # Source 1: ground-truth tables_registry
            for tbl in tbls:
                sel  = tbl.get("selector", "")
                hdrs = [h for h in (tbl.get("headers") or []) if h and h.strip()]
                req  = [h for h in (tbl.get("required") or hdrs) if h and h.strip()]
                if not sel or not hdrs:
                    continue
                seen = seen_selectors_per_url.setdefault(url, set())
                if sel in seen:
                    continue
                seen.add(sel)
                registry.setdefault(url, []).append({
                    "selector":  sel,
                    "headers":   hdrs,
                    "required":  req or hdrs,
                    "row_count": tbl.get("row_count", 0),
                })
            logger.debug(
                f"[TABLE-REGISTRY] {url}: {len(tbls)} tables from ground-truth capture"
            )

        elif step.get("table_selector_hint") or step.get("required_columns_hint"):
            # Source 2: single-table hint (XPath-derived or legacy)
            sel  = step.get("table_selector_hint", "")
            hdrs = step.get("exact_headers") or []
            req  = step.get("required_columns_hint") or hdrs
            if sel:
                seen = seen_selectors_per_url.setdefault(url, set())
                if sel not in seen:
                    seen.add(sel)
                    registry.setdefault(url, []).append({
                        "selector":  sel,
                        "headers":   hdrs,
                        "required":  req or hdrs,
                        "row_count": 0,
                    })
            elif req:
                # No selector known — surface page so nav plan uses networkidle + auto-discover
                registry.setdefault(url, []).append({
                    "selector":  "",
                    "headers":   req,
                    "required":  req,
                    "row_count": 0,
                })

    # ── Source 3: _KNOWN_PAGE_SELECTORS fallback ─────────────────────────────
    # For pages where the agent either:
    #   a) Never called extract_content (agent took a different path), OR
    #   b) Called extract_content but the history predates tables_on_page capture,
    # inject the verified static selector + columns so the navigation plan always
    # has something definitive to show the LLM.
    #
    # This iterates ALL URLs seen in the steps (not just registry.keys()) so that
    # pages with no history data at all also get covered.
    for url in all_step_urls:
        kp = _known_page_info(url)
        if not kp:
            continue
        known_sel  = kp["selector"]
        known_cols = kp.get("columns") or []

        if url not in registry:
            # Page produced no history data at all → inject from known pages
            registry[url] = [{
                "selector":  known_sel,
                "headers":   known_cols,
                "required":  known_cols,
                "row_count": 0,
            }]
            logger.info(
                f"[TABLE-REGISTRY] {url}: "
                f"added from _KNOWN_PAGE_SELECTORS: {known_sel!r}"
            )
        else:
            # Page IS in registry but may have empty/wrong selector entries
            entries = registry[url]
            has_good_selector = any(
                e.get("selector") and e["selector"] != ""
                for e in entries
            )
            if not has_good_selector and known_sel:
                registry[url] = [{
                    "selector":  known_sel,
                    "headers":   known_cols,
                    "required":  known_cols,
                    "row_count": 0,
                }]
                logger.info(
                    f"[TABLE-REGISTRY] {url}: "
                    f"upgraded empty entry → {known_sel!r}"
                )

    logger.info(f"[PERF] Registry loop took {_time.time() - _t_loop:.2f}s")
    return registry


def _scan_element_data_files(target_url: str = None) -> tuple:
    """Scan tmp/element_data/*.json for real selectors captured during the agent run.
    
    Looks for elements inside table structures (td/tr/tbody ancestors) and
    builds a CSS selector from the nearest ancestor with an ID.
    
    Returns: (table_selector, page_url) or (None, None)
    """
    elem_dir = Path("tmp/element_data")
    if not elem_dir.exists():
        return None, None
    
    best_selector = None
    best_url = None
    best_stability = -1
    
    for fpath in sorted(elem_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                edata = json.load(f)
        except Exception:
            continue
        
        page_url = edata.get("metadata", {}).get("page_url", "")
        
        # If target_url is known, prefer elements from matching pages
        if target_url and page_url:
            from urllib.parse import urlparse
            t_host = urlparse(target_url).hostname or ""
            e_host = urlparse(page_url).hostname or ""
            if t_host and e_host and t_host != e_host:
                continue
        
        # Look for table-related ancestors in parentChain
        parent_chain = edata.get("parentChain", [])
        has_table_ancestor = any(
            p.get("tag") in ("td", "tr", "tbody", "table", "thead")
            for p in parent_chain
        )
        if not has_table_ancestor:
            continue
        
        # Find nearest ancestor with an ID → build table selector
        file_candidate = None
        file_stability = -1

        for parent in parent_chain:
            pid = parent.get("id")
            if pid:
                # Skip login and layout-related containers
                pid_lower = pid.lower()
                ignore_kws = ("login", "signin", "sign-in", "otds", "auth", "password", "menu", "nav", "header", "sidebar", "footer", "toolbar")
                if any(kw in pid_lower for kw in ignore_kws):
                    continue
                file_candidate = f"#{pid} table"
                # Use stability of the element's recommended selector as tie-breaker
                rec = edata.get("recommendedSelector", {})
                file_stability = rec.get("stability", 50) if isinstance(rec, dict) else 50
                break
        
        # Also check xpath for table container IDs if parent chain didn't yield a good candidate
        if not file_candidate:
            for sel_entry in edata.get("selectors", []):
                if sel_entry.get("type") == "xpath":
                    xpath = sel_entry.get("selector", "")
                    # Extract @id from xpath like //*[@id="admin-servers"]/table[1]/...
                    id_match = re.search(r'@id="([^"]+)"', xpath)
                    if id_match:
                        pid = id_match.group(1)
                        pid_lower = pid.lower()
                        ignore_kws = ("login", "signin", "sign-in", "otds", "auth", "password", "menu", "nav", "header", "sidebar", "footer", "toolbar")
                        if not any(kw in pid_lower for kw in ignore_kws):
                            file_candidate = f"#{pid} table"
                            file_stability = 45  # Slightly lower confidence than a direct parent ID
                        break

        # Apply a massive stability boost if this element was captured during a data extraction action
        # This prevents generic navigation menus (which the agent clicks) from outranking the actual data table
        action_context = edata.get("metadata", {}).get("action_context", "").lower()
        if "retrieve" in action_context or "extract" in action_context:
            file_stability += 100
            
        # Penalize empty structural tables
        if not edata.get("textContent", "").strip():
            file_stability -= 50

        # If this file yielded a candidate, compare it against the global best
        if file_candidate and file_stability > best_stability:
            best_selector = file_candidate
            best_url = page_url
            best_stability = file_stability
    
    return best_selector, best_url


def _build_action_selector_map(cleaned_history_json: str) -> str:
    """Build a compact action→selector reference table from cleaned history.

    The LLM is instructed to use ONLY these selectors, which prevents hallucination
    far more reliably than prose rules alone.

    Returns a formatted string like:
        STEP 1 | URL: https://...
          [click_element]   selector: #real-btn  | text: "Submit" | aria-label: "Submit"
          [input_text]      selector: input[name='q'] | type: "text"
    """
    try:
        data = json.loads(cleaned_history_json)
    except Exception:
        return ""

    steps = data.get("steps") if isinstance(data, dict) else data
    if not isinstance(steps, list) or not steps:
        return ""

    SELECTOR_PRIORITY = ["preferred", "css", "recommended_selector", "xpath", "aria", "text"]

    def _best_sel(el: dict) -> str:
        if not isinstance(el, dict):
            return ""
        sels = el.get("selectors") or {}
        rec = el.get("recommended_selector", "")
        if isinstance(rec, dict):
            rec = rec.get("selector", "")
        for key in SELECTOR_PRIORITY:
            val = rec if key == "recommended_selector" else sels.get(key, "")
            if val and str(val).strip() and str(val).strip().lower() not in ("table", ""):
                return str(val).strip()
        return ""

    lines = ["=== SELECTOR REFERENCE TABLE (use ONLY these selectors) ==="]
    has_any = False

    for step_idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        actions = step.get("actions", []) or []
        dom_els = step.get("dom_elements", []) or []
        url     = step.get("url", "")

        step_hdr  = f"\nSTEP {step_idx + 1}"
        if url:
            step_hdr += f" | URL: {url}"
        step_lines = []

        for act_idx, act in enumerate(actions):
            if not isinstance(act, dict):
                continue
            action_type = next(iter(act.keys()), "unknown")
            if action_type in ("go_to_url", "done", "check_ssl_certificate",
                               "scroll", "wait", "extract_content"):
                continue

            el   = dom_els[act_idx] if act_idx < len(dom_els) else (dom_els[-1] if dom_els else {})
            sel  = _best_sel(el)
            text = str(el.get("text", "") or "")[:60]
            aria = str(el.get("aria-label", "") or "")[:60]
            el_type = str(el.get("type", "") or "")
            tag  = str(el.get("tag", "") or "")

            parts = [f"[{action_type}]  selector: {sel or '(none — use aria/text)'}"]
            if text:
                parts.append(f'text: "{text}"')
            if aria:
                parts.append(f'aria-label: "{aria}"')
            if el_type:
                parts.append(f'type: "{el_type}"')
            if tag:
                parts.append(f'tag: <{tag}>')

            step_lines.append("  " + " | ".join(parts))
            has_any = True

        if step_lines:
            lines.append(step_hdr)
            lines.extend(step_lines)

    if not has_any:
        return ""

    lines.append("\n=== END SELECTOR REFERENCE TABLE ===")
    return "\n".join(lines)


def _build_element_data_context() -> str:
    """Build a concise summary of captured element data for the LLM prompt.
    
    Returns a formatted string summarizing real selectors, parent structure,
    and page URLs from tmp/element_data/*.json files.
    """
    elem_dir = Path("tmp/element_data")
    if not elem_dir.exists():
        return ""
    
    summaries = []
    seen_hashes = set()
    
    for fpath in sorted(elem_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                edata = json.load(f)
        except Exception:
            continue
        
        ehash = edata.get("elementHash", "")
        if ehash in seen_hashes:
            continue
        seen_hashes.add(ehash)
        
        tag = edata.get("tagName", "?")
        text = (edata.get("textContent") or "")[:60]
        url = edata.get("metadata", {}).get("page_url", "")
        action = edata.get("metadata", {}).get("action_context", "")
        
        # Get best selector
        rec = edata.get("recommendedSelector", {})
        sel = rec.get("selector", "") if isinstance(rec, dict) else ""
        
        # Get parent container with ID
        container = ""
        for parent in edata.get("parentChain", []):
            pid = parent.get("id")
            if pid:
                container = f"#{pid}"
                break
        
        parts = [f'- <{tag}> "{text}"']
        if sel:
            parts.append(f"  Selector: {sel}")
        if container:
            parts.append(f"  Container: {container} table")
        if url:
            parts.append(f"  Page: {url}")
        if action:
            parts.append(f"  Action: {action}")
        
        summaries.append("\n".join(parts))
    
    if not summaries:
        return ""
    
    return "\n\nCAPTURED ELEMENT DATA (real selectors from the page — use these instead of guessing):\n" + "\n".join(summaries)


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


def _postprocess_generated_code(code: str, cleaned_history_json: str, tbl_registry: dict | None = None) -> str:
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
    elem_selector, _ = _scan_element_data_files(process_url)
    # _KNOWN_PAGE_SELECTORS and _known_page_info() are defined at module level
    # (above SYSTEM_PROMPT) so Rule 16 and other postprocessor rules reference them directly.

    # ── -1: Repair garbled string literals before any other processing ────────
    # LLM occasionally emits broken strings like:
    #   table_selector = "some/selector"ExtraGarbage\")"
    # This is a syntax error. Fix by truncating at the closing quote.
    def _repair_broken_strings(src: str) -> str:
        """Fix lines where a Python string literal has trailing garbage after its closing quote."""
        fixed_lines = []
        changed = False
        for line in src.splitlines(keepends=True):
            # Pattern: variable = "..." followed by non-whitespace/non-comment junk
            m = re.match(
                r'^(\s*\w+\s*=\s*)(")([^"\\]*(?:\\.[^"\\]*)*)"(.+)$',
                line.rstrip('\n\r')
            )
            if m:
                pre, q, content, junk = m.group(1), m.group(2), m.group(3), m.group(4)
                # Only repair if junk looks wrong (not a comment, not a comma/paren)
                junk_stripped = junk.strip()
                if junk_stripped and not junk_stripped.startswith(('#', ',', ')', ']', '+')):
                    fixed = f'{pre}"{content}"\n'
                    logger.warning(f"[POST] Repaired garbled string literal. Removed: {junk_stripped!r}")
                    fixed_lines.append(fixed)
                    changed = True
                    continue
            fixed_lines.append(line)
        return "".join(fixed_lines) if changed else src

    def _repair_broken_join(src: str) -> str:
        """Repair the common LLM mistake of dropping '.join([' from the output line.

        The LLM emits:
            output = "\\n"
                    f"Identity: ..."
                    for row in rows
                    ])

        But should emit:
            output = "\\n".join([
                f"Identity: ..."
                for row in rows
            ])
        """
        # Match: output = "\\n" (newline, no .join) followed by an f-string continuation
        broken = re.compile(
            r'(\s*)(output\s*=\s*"\\n")\s*\n'   # output = "\n"  ← missing .join([
            r'(\s+)(f["\'].+?["\'])\s*\n'         # f"..." line
            r'(\s+for\s+\w+\s+in\s+\w+)\s*\n'    # for row in rows
            r'(\s*\]\))',                           # ])
            re.DOTALL
        )
        def _fix_join(m):
            indent  = m.group(1)
            inner   = m.group(3)
            fstring = m.group(4)
            for_clause = m.group(5).strip()
            logger.warning("[POST] Repaired broken output=\\n join expression")
            return (
                f'{indent}output = "\\n".join([\n'
                f'{inner}{fstring}\n'
                f'{inner}{for_clause}\n'
                f'{indent}])'
            )
        fixed = broken.sub(_fix_join, src)

        # Second pattern: output = "\n" on its own line, f-string body, for clause, no bracket
        # output = "\n"
        # f"Identity..."
        # for row in rows
        # ← missing the wrapping entirely
        broken2 = re.compile(
            r'(\s*)(output\s*=\s*"\\n")\n'
            r'(\s+)(f["\'].+?["\'])\n'
            r'(\s+)(for\s+\w+\s+in\s+\w+)\n'
            r'(?!\s*\]\))',   # NOT followed by ])
        )
        def _fix_join2(m):
            indent     = m.group(1)
            inner      = m.group(3)
            fstring    = m.group(4)
            for_kw     = m.group(5)
            for_clause = m.group(6)
            logger.warning("[POST] Repaired broken output join (no brackets)")
            return (
                f'{indent}output = "\\n".join([\n'
                f'{inner}{fstring}\n'
                f'{for_kw}{for_clause}\n'
                f'{indent}])\n'
            )
        fixed = broken2.sub(_fix_join2, fixed)
        return fixed

    code = _repair_broken_strings(code)
    code = _repair_broken_join(code)

    # ── -0: Validate Python syntax — warn loudly if broken ────────────────────
    import ast as _ast
    try:
        _ast.parse(code)
    except SyntaxError as _syn:
        logger.error(f"[POST] Generated code has syntax error at line {_syn.lineno}: {_syn.msg}")
        # Inject a visible comment so the file doesn't silently fail
        code = f"# ❌ SYNTAX ERROR (line {_syn.lineno}): {_syn.msg}\n# Fix this before running.\n\n" + code

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


    # ── 1: Ensure script_helpers import is present ────────────────────────────
    helpers_import = (
        "from src.utils.script_helpers import (\n"
        "    get_secret, find_in_frames,\n"
        "    resilient_fill, resilient_click, click_and_upload_file,\n"
        "    maybe_login, resilient_extract_table, generate_report,\n"
        "    check_certificate,\n"
        ")"
    )
    if "from src.utils.script_helpers import" not in code:
        # Insert AFTER the entire if/else bootstrap block, not inside it.
        # The bootstrap always ends with the else-branch sys.path.append line.
        # Pattern: else:\n    sys.path.append(...)\n
        inserted = False
        if "sys.path.append" in code:
            # Match the full bootstrap block: if ...:\n    sys.path.append\nelse:\n    sys.path.append
            new_code = re.sub(
                r'((?:if[^\n]+\n[ \t]+sys\.path\.append\([^\n]+\)\n'   # if branch
                r'else:\n[ \t]+sys\.path\.append\([^\n]+\)\n))',         # else branch
                r'\1\n' + helpers_import + '\n',
                code, count=1,
            )
            if new_code != code:
                code = new_code
                inserted = True
            else:
                # Fallback: after the last sys.path.append line in the file header
                new_code = re.sub(
                    r'(sys\.path\.append\([^\n]+\)\n)(?!\n*sys\.path\.append)',
                    r'\1\n' + helpers_import + '\n',
                    code, count=1,
                )
                if new_code != code:
                    code = new_code
                    inserted = True

        if not inserted:
            code = re.sub(
                r'(from playwright\.async_api import async_playwright\n)',
                r'\1' + helpers_import + '\n',
                code, count=1,
            )
        if "from src.utils.script_helpers import" not in code:
            code = helpers_import + "\n" + code

    # Strip now-redundant imports that script_helpers already handles internally
    code = re.sub(r'^from docx import Document\s*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'^from src\.utils\.vault import vault\s*\n', '', code, flags=re.MULTILINE)
    # Strip old monolithic config import (script_helpers + generated header cover these)
    code = re.sub(
        r'^from src\.utils\.config import \([^)]+\)\n', '', code, flags=re.MULTILINE | re.DOTALL
    )

    # ── 2: Strip hallucinated helper function definitions ─────────────────────
    # Names that must NOT be redefined — they live in script_helpers.py
    _HELPER_NAMES = (
        "get_secret", "find_in_frames", "resilient_fill", "resilient_click",
        "click_and_upload_file", "maybe_login", "resilient_extract_table",
        "generate_report",
    )

    def _strip_async_def(src: str, fn_name: str) -> str:
        """Remove an async def block for fn_name from src."""
        pattern = rf'(?m)^async def {re.escape(fn_name)}\b.*?(?=\n(?:async def |def |class |\Z))'
        match = re.search(pattern, src, re.DOTALL)
        if match:
            logger.info(f"[POST] Stripped hallucinated helper: async def {fn_name}")
            src = src[:match.start()] + src[match.end():]
        return src

    for name in _HELPER_NAMES:
        if re.search(rf'^async def {re.escape(name)}\b', code, re.MULTILINE):
            code = _strip_async_def(code, name)

    # ── 3: Rewrite wrong/login-field table selectors ──────────────────────────
    # IMPORTANT: rewrites are URL-scoped.  A script that visits multiple pages
    # (e.g. a Distributed-Agent page AND an Admin-Servers page) has different
    # correct selectors per page.  Applying a single global regex replacement
    # would stamp "#admin-servers table" onto the wrong page's wait_for_selector
    # call and cause a timeout.
    #
    # Strategy: walk line-by-line, track the most recent page.goto() URL, and
    # only rewrite a selector call when we are "on" the target page (i.e. the
    # most recent goto URL matches or contains process_url).

    def _is_fragile_selector(s: str) -> bool:
        """Return True if the selector is positional/brittle and should be replaced."""
        if re.search(r':nth-(?:child|of-type)\(', s):
            return True
        if re.match(r'^(?:html\s*>?\s*)?body\s*(?:>?\s*\w+)*\s*>?\s*table\b', s.strip()):
            return True
        return False

    def _should_replace(sel_str: str, on_target_page: bool = True) -> bool:
        s = sel_str.strip().strip('"\'')
        # Always replace login selectors and bare "table" regardless of page
        if s == "table" or _is_login_selector(s):
            return True
        if _is_fragile_selector(s):
            logger.info(f"[POST] Replacing fragile positional selector: '{s}'")
            return True
        # Only replace container-less selectors when we are on the correct page
        if not on_target_page:
            return False
        if elem_selector and not any(tag in s.lower() for tag in ['table', 'tr', 'tbody', 'thead', '#']):
            logger.info(f"[POST] LLM selector '{s}' lacks container context. Forcing '{elem_selector}'.")
            return True
        return False

    final_selector = elem_selector if (elem_selector and selector and _should_replace(selector)) else selector

    if final_selector:
        from urllib.parse import urlparse as _up

        # Derive a URL "fingerprint" for the target page so we can match gotos.
        # Use the query-string portion of process_url (e.g. "func=ll&objtype=148")
        # which is stable regardless of host/port.
        _proc_qs = ""
        if process_url:
            _parsed = _up(process_url)
            _proc_qs = _parsed.query or _parsed.path  # fallback to path if no query

        def _url_is_target(url: str) -> bool:
            """Return True if url looks like the page where final_selector lives."""
            if not _proc_qs:
                return True   # no process URL known — allow rewrite everywhere
            return _proc_qs in url

        sel_literal = json.dumps(final_selector)
        _str_pat = r'(["\'])(?:(?=(\\?))\2.)*?\1'   # any Python string literal

        # Line-by-line rewrite with URL tracking
        new_lines = []
        current_url = ""   # most recent page.goto() target seen so far
        on_target   = not bool(_proc_qs)  # True from the start if we have no URL hint

        for line in code.splitlines(keepends=True):
            # Track page.goto(URL) to know which page we are on
            goto_m = re.search(r'page\.goto\(\s*(["\'][^"\']+["\'])', line)
            if goto_m:
                current_url = goto_m.group(1).strip("'\"")
                on_target   = _url_is_target(current_url)

            # wait_for_selector
            def _fix_wait(m, _on=on_target):
                return f'page.wait_for_selector({sel_literal}' if _should_replace(m.group(1), _on) else m.group(0)
            line = re.sub(rf'page\.wait_for_selector\({_str_pat}', _fix_wait, line)

            # resilient_extract_table
            def _fix_extract(m, _on=on_target):
                return f'resilient_extract_table(page, {sel_literal}' if _should_replace(m.group(1), _on) else m.group(0)
            line = re.sub(rf'resilient_extract_table\(\s*page\s*,\s*{_str_pat}', _fix_extract, line)

            # table_selector = "..."
            def _fix_table_var(m, _on=on_target):
                return f'table_selector = {sel_literal}' if _should_replace(m.group(1), _on) else m.group(0)
            line = re.sub(rf'table_selector\s*=\s*{_str_pat}', _fix_table_var, line)

            # page.locator("...")
            def _fix_locator(m, _on=on_target):
                return f'page.locator({sel_literal}' if _should_replace(m.group(1), _on) else m.group(0)
            line = re.sub(rf'page\.locator\({_str_pat}', _fix_locator, line)

            new_lines.append(line)

        code = "".join(new_lines)

    # ── 4: Inject required columns (with identity key) ────────────────────────
    if required:
        if not any(c.lower() in ('server', 'name') for c in required):
            required = list(required) + ['Server']
        req_literal = json.dumps(required)
        code = re.sub(
            r'(resilient_extract_table\(\s*page\s*,\s*[^,]+,\s*)(\[[^\]]*\])',
            rf'\1{req_literal}', code,
        )

    # ── 5: Ensure maybe_login(page) is called after first page.goto ──────────
    if "maybe_login(page)" not in code:
        code = re.sub(
            r'(await page\.goto\([^\n]+\)\n)',
            r'\1        await maybe_login(page)\n',
            code, count=1,
        )

    # ── 6: Fix post-login wait if it targets a login-field selector ──────────
    if selector and "await maybe_login(page)" in code:
        sel_literal = json.dumps(selector)
        lines = code.splitlines(keepends=True)
        past_login, fixed_lines = False, []
        for line in lines:
            if "await maybe_login(page)" in line:
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

    # ── 11: Strip redundant manual login steps after maybe_login ─────────────
    # If the script calls maybe_login(page) AND then also manually does
    # resilient_fill(page, <login_sel>, ...) + resilient_click(page, <login_sel>),
    # the manual steps are redundant and must be removed.
    if "await maybe_login(page)" in code:
        lines = code.splitlines(keepends=True)
        past_login, out_lines = False, []
        i = 0
        while i < len(lines):
            line = lines[i]
            if "await maybe_login(page)" in line:
                past_login = True
                out_lines.append(line)
                i += 1
                continue
            if past_login:
                stripped = line.strip()
                # Detect manual login: resilient_fill/click/page.fill/page.click
                # targeting a login-field selector
                is_login_fill = bool(re.search(
                    r'(?:resilient_fill|page\.fill)\s*\(\s*page\s*,\s*(["\'][^"\']+["\'])',
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
                # Also catch: await resilient_click(page, "#loginbutton")
                # and login button patterns
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
    if process_url and "await maybe_login(page)" in code:
        # Check if the process URL (or same host) is already in a goto call
        from urllib.parse import urlparse as _urlparse
        _proc_host = _urlparse(process_url).hostname or ""
        # Check if any existing goto targets the same host
        _existing_gotos = re.findall(r'page\.goto\(["\']([^"\']+)["\']', code)
        _already_navigates = any(
            _proc_host and _proc_host == (_urlparse(g).hostname or "")
            for g in _existing_gotos
        )
        if not _already_navigates and process_url not in code:
            # Find the first wait_for_selector AFTER maybe_login and insert goto before it
            lines = code.splitlines(keepends=True)
            past_login, injected = False, False
            new_lines = []
            for line in lines:
                if "await maybe_login(page)" in line:
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

    # ── 14: Replace hallucinated #view_<N> selectors in wait_for_selector ───────
    # LLM sometimes invents IDs like "#view_1011" that don't exist on the page.
    # Detection: any wait_for_selector using "#view_<digits>" that is NOT in the
    # known selector from history → replace with networkidle wait or the real selector.
    def _is_invented_view_id(sel: str) -> bool:
        s = sel.strip().strip("'\"")
        if not re.match(r'^#view_\d+$', s):
            return False
        # If it matches the known good selector from history, it's fine
        if selector and s == selector.strip():
            return False
        return True

    def _fix_invented_view_id(m):
        sel_str = m.group(1)
        if _is_invented_view_id(sel_str):
            # Replace with real selector if we have one, else networkidle approach
            replacement = selector if selector else None
            if replacement:
                logger.warning(f"[POST] Replaced hallucinated selector {sel_str!r} → {replacement!r}")
                return f'page.wait_for_selector({json.dumps(replacement)}'
            else:
                logger.warning(f"[POST] Removed hallucinated wait_for_selector({sel_str}) — using networkidle")
                return f'page.wait_for_load_state("networkidle"  # was: wait_for_selector({sel_str}'
        return m.group(0)

    code = re.sub(
        r'page\.wait_for_selector\((["\']#view_\d+["\'])',
        _fix_invented_view_id,
        code,
    )
    # Also fix resilient_extract_table using an invented #view_* selector
    def _fix_invented_extract(m):
        sel_str = m.group(1)
        if _is_invented_view_id(sel_str):
            replacement = selector if selector else '"#admin-servers table"'
            logger.warning(f"[POST] Replaced hallucinated extract selector {sel_str!r} → {replacement!r}")
            return f'resilient_extract_table(page, {json.dumps(replacement) if isinstance(replacement, str) else replacement}'
        return m.group(0)

    code = re.sub(
        r'resilient_extract_table\(\s*page\s*,\s*(["\']#view_\d+["\'])',
        _fix_invented_extract,
        code,
    )

    # ── 15: Replace hardcoded OTCS row function-menu IDs (#x1, #x2 …) ───────────
    # OTCS renders function menu hotspots as <a id="x1" class="functionMenuHotspot">
    # where the number is the row's position — it changes when rows are reordered.
    # Any resilient_click / page.click targeting "#x<digits>" is therefore fragile.
    # Replace with a page.evaluate() block that finds the hotspot relative to the
    # named link text. We extract the server name from the nearest string literal
    # in the same expression or fall back to a generic nearest-hotspot approach.
    _xN_pat = re.compile(
        r'([ \t]*)await\s+resilient_click\s*\(\s*page\s*,\s*["\']#x(\d+)["\'][^)]*\)'
        r'(?:\s*#[^\n]*)?'
    )
    def _replace_xN_click(m):
        indent = m.group(1)
        logger.warning(f"[POST] Replaced hardcoded function menu ID #x{m.group(2)} with JS hotspot click")
        return (
            f'{indent}# Click function menu hotspot — resolved dynamically (never use #x<N>)\n'
            f'{indent}_hotspot_clicked = await page.evaluate(\'\'\'\n'
            f'{indent}    () => {{\n'
            f'{indent}        for (const a of document.querySelectorAll("a.functionMenuHotspot")) {{\n'
            f'{indent}            const td = a.closest("td");\n'
            f'{indent}            if (td) {{ a.click(); return {{ clicked: true, id: a.id }}; }}\n'
            f'{indent}        }}\n'
            f'{indent}        return {{ clicked: false }};\n'
            f'{indent}    }}\n'
            f"{indent}''')\n"
            f'{indent}print(f"[*] Function menu clicked: {{_hotspot_clicked}}")'
        )

    code = _xN_pat.sub(_replace_xN_click, code)

    # Also catch page.locator("#x<N>") and page.click("#x<N>") patterns
    code = re.sub(
        r'await\s+page\.click\s*\(\s*["\']#x\d+["\']\s*\)',
        lambda m: (
            logger.warning("[POST] Replaced hardcoded page.click(#x<N>)") or
            'await page.evaluate(\'() => { const a = document.querySelector("a.functionMenuHotspot"); if(a) a.click(); }\')'
        ),
        code,
    )
    # Catch wait_for_selector("#x1") — just remove it, it's not a real wait target
    code = re.sub(
        r'await\s+page\.wait_for_selector\s*\(\s*["\']#x\d+["\'][^)]*\)\s*\n',
        '',
        code,
    )


    # ── 17: Deterministic column correction from DOM snapshot (tbl_registry) ─
    # Rule 16 fixes selectors using _KNOWN_PAGE_SELECTORS (a static map).
    # Rule 17 fixes columns using the live tbl_registry captured from the actual
    # DOM during the agent run — this has the EXACT headers per table per URL.
    #
    # For single-table pages: replace columns on every extract call for that URL.
    # For multi-table pages: match each call by its selector, then replace columns.
    # Unmatched calls (selector not in registry) are left untouched.
    if tbl_registry:
        _r17_extract_pat = re.compile(
            r'(resilient_extract_table\(\s*page\s*,\s*)(["\'][^"\']+["\'])(\s*,\s*)(\[[^\]]*\])'
        )
        _r17_url = ""
        r17_lines = []
        for line in code.splitlines(keepends=True):
            gm = re.search(r'page\.goto\(\s*["\']([^"\']+)["\']', line)
            if gm:
                _r17_url = gm.group(1)

            if "resilient_extract_table" in line and _r17_url:
                page_tables = tbl_registry.get(_r17_url) or []
                if page_tables:
                    def _r17_fix(m, _ptbls=page_tables):
                        sel_str  = m.group(2).strip().strip("\"'")
                        orig_cols = m.group(4)

                        # Multi-table: find the entry whose selector matches this call's selector
                        matched = next(
                            (t for t in _ptbls if t.get("selector") == sel_str),
                            None
                        )
                        # Single-table fallback: if there's exactly one table, use it
                        if matched is None and len(_ptbls) == 1:
                            matched = _ptbls[0]

                        if matched is None:
                            # Multiple tables, none matched — leave this call untouched
                            return m.group(0)

                        # Use exact_headers (ground-truth DOM) > required (scored subset)
                        exact = matched.get("exact_headers") or matched.get("headers") or []
                        required = matched.get("required") or exact
                        # Prefer required (the scored/relevant subset), fall back to all headers
                        cols = required if required else exact
                        if not cols:
                            return m.group(0)  # nothing to inject

                        new_cols = json.dumps(cols)
                        if new_cols != orig_cols:
                            logger.info(
                                "[POST Rule17] %s selector=%r: cols %s → %s",
                                _r17_url, sel_str, orig_cols, new_cols
                            )
                        return f"{m.group(1)}{m.group(2)}{m.group(3)}{new_cols}"

                    line = _r17_extract_pat.sub(_r17_fix, line)

            r17_lines.append(line)
        code = "".join(r17_lines)

    # ── 16: Replace wrong/bare table selectors using known-page selector map ──
    # Extended (was: only replaced bare "table").
    #
    # For EVERY page where we know the correct selector (_KNOWN_PAGE_SELECTORS),
    # we now replace ANY table selector that doesn't match — including hallucinated
    # ones like "#metrics-panel table", "#browseViewCoreTable", etc.
    # We also inject the correct columns for that page.
    #
    # Same line-by-line URL-tracking approach as Rule 3 so each page gets its
    # own selector and columns independently.
    _bare_table_pat = re.compile(
        r'(resilient_extract_table\(\s*page\s*,\s*)["\']table["\']'
    )
    _any_extract_pat = re.compile(
        r'(resilient_extract_table\(\s*page\s*,\s*)(["\'][^"\']+["\'])'
    )
    _bare_wait_pat = re.compile(
        r'(page\.wait_for_selector\(\s*)["\']table["\']'
    )
    # NEW: matches wait_for_selector with ANY selector string
    _any_wait_pat = re.compile(
        r'(page\.wait_for_selector\(\s*)(["\'][^"\']+["\'])'
    )
    _any_cols_pat = re.compile(
        r'(resilient_extract_table\(\s*page\s*,\s*[^,]+,\s*)(\[[^\]]*\])'
    )

    new_lines = []
    _kps_url = ""
    for line in code.splitlines(keepends=True):
        goto_m = re.search(r'page\.goto\(\s*["\']([^"\']+)["\']', line)
        if goto_m:
            _kps_url = goto_m.group(1)

        # Look up known selector + columns for current URL
        kp_entry = _known_page_info(_kps_url)
        _known_sel  = kp_entry["selector"]  if kp_entry else None
        _known_cols = kp_entry.get("columns") if kp_entry else None

        if _known_sel:
            if _known_sel != "table":
                # Replace ANY extract selector that doesn't match the known one.
                # This catches bare "table" AND hallucinated IDs like "#metrics-panel table".
                def _fix_any_extract(m, _ks=_known_sel):
                    cur = m.group(2).strip().strip('"\'')
                    if cur != _ks:
                        logger.info(
                            f"[POST Rule16] extract {cur!r} → {_ks!r} (URL: {_kps_url})"
                        )
                        return f'{m.group(1)}{json.dumps(_ks)}'
                    return m.group(0)
                line = _any_extract_pat.sub(_fix_any_extract, line)

                # Replace ANY wait_for_selector that doesn't match the known selector.
                # This catches "#x1 table", "#browseViewCoreTable", bare "table", etc.
                def _fix_any_wait(m, _ks=_known_sel):
                    cur = m.group(2).strip().strip('"\'')
                    if cur != _ks:
                        logger.info(
                            f"[POST Rule16] wait {cur!r} → {_ks!r} (URL: {_kps_url})"
                        )
                        return f'{m.group(1)}{json.dumps(_ks)}'
                    return m.group(0)
                line = _any_wait_pat.sub(_fix_any_wait, line)

            else:
                # For DA page (known_sel == "table"), only replace truly wrong non-table selectors.
                # We don't replace bare "table" since that is the correct selector.
                def _fix_non_table_extract(m, _ks="table"):
                    cur = m.group(2).strip().strip('"\'')
                    # Replace if it's a container selector (has # or .) but not bare "table"
                    if cur != "table" and (cur.startswith('#') or cur.startswith('.')):
                        logger.info(
                            f"[POST Rule16] extract {cur!r} → 'table' (DA page, URL: {_kps_url})"
                        )
                        return f'{m.group(1)}"table"'
                    return m.group(0)
                line = _any_extract_pat.sub(_fix_non_table_extract, line)

                def _fix_non_table_wait(m):
                    cur = m.group(2).strip().strip('"\'')
                    if cur != "table" and (cur.startswith('#') or cur.startswith('.')):
                        logger.info(
                            f"[POST Rule16] wait {cur!r} → 'table' (DA page, URL: {_kps_url})"
                        )
                        return f'{m.group(1)}"table"'
                    return m.group(0)
                line = _any_wait_pat.sub(_fix_non_table_wait, line)

            # Inject correct columns for this page if we know them.
            # Skip if tbl_registry already has live DOM data for this URL
            # (Rule 17 will have already set the right columns before Rule 16 runs).
            _r16_tbl_covered = bool(tbl_registry and tbl_registry.get(_kps_url))
            if _known_cols and not _r16_tbl_covered:
                def _fix_cols(m, _kc=_known_cols):
                    return f'{m.group(1)}{json.dumps(_kc)}'
                line = _any_cols_pat.sub(_fix_cols, line)

        new_lines.append(line)
    code = "".join(new_lines)

    # ── Quality gate ─────────────────────────────────────────────────────────
    issues = []
    if re.search(r'resilient_extract_table\(\s*page\s*,\s*["\']table["\']', code):
        # Only flag if there are URLs in the script where we had a known non-"table" selector
        # but still couldn't fix it (i.e. the known-page map had a match).
        _unfixed_urls = []
        _scan_url = ""
        for line in code.splitlines():
            gm = re.search(r'page\.goto\(\s*["\']([^"\']+)["\']', line)
            if gm:
                _scan_url = gm.group(1)
            if re.search(r'resilient_extract_table\(\s*page\s*,\s*["\']table["\']', line):
                kp = _known_page_info(_scan_url)
                if kp and kp.get("selector") not in (None, "table"):
                    _unfixed_urls.append(_scan_url)
        if _unfixed_urls:
            issues.append(f"resilient_extract_table still uses bare 'table' for known page(s): {_unfixed_urls}")
        else:
            logger.info("[POST] bare 'table' selector kept — no known replacement for this page")
    if re.search(r'page\.wait_for_selector\(\s*["\']table["\']', code):
        # Only flag if this is on a page where we know the real selector isn't bare "table"
        _wait_scan_url = ""
        _wait_is_issue = False
        for _wl in code.splitlines():
            gm = re.search(r'page\.goto\(\s*["\']([^"\']+)["\']', _wl)
            if gm:
                _wait_scan_url = gm.group(1)
            if re.search(r'page\.wait_for_selector\(\s*["\']table["\']', _wl):
                kp = _known_page_info(_wait_scan_url)
                if kp and kp.get("selector") not in (None, "table"):
                    _wait_is_issue = True
                    break
        if _wait_is_issue:
            issues.append("wait_for_selector still uses bare 'table' selector on known page")
    if re.search(r'table_selector\s*=\s*["\']table["\']', code):
        issues.append("table_selector still assigned literal 'table'")
    if re.search(r'page\.locator\(\s*["\']tr["\']', code):
        issues.append("global page.locator('tr') without table scoping")
    if "maybe_login(page)" not in code:
        issues.append("maybe_login(page) call missing after page.goto")
    for fn in ("resilient_extract_table", "wait_for_selector"):
        m = re.search(rf'{fn}\(\s*(?:page\s*,\s*)?(["\'][^"\']+["\'])', code)
        if m and _is_login_selector(m.group(1).strip("'\"")):
            issues.append(f"{fn} still uses a login-field selector: {m.group(1)!r}")
    if "from src.utils.script_helpers import" not in code:
        issues.append("script_helpers import missing")
    # Check for any surviving hardcoded row IDs or invented selectors
    if re.search(r'["\']#x\d+["\']', code):
        issues.append("hardcoded OTCS row function-menu ID (#x<N>) still present — use a.functionMenuHotspot instead")
    if re.search(r'["\']#view_\d+["\']', code):
        issues.append("invented #view_<N> selector still present — use a known selector from the SELECTOR REFERENCE TABLE")

    if issues:
        warning = "# ⚠️  POST-PROCESSING WARNINGS:\n" + "\n".join(f"# - {i}" for i in issues) + "\n\n"
        logger.warning("[QUALITY GATE]\n  " + "\n  ".join(issues))
        if selector is None and any("selector" in i for i in issues):
            raise RuntimeError(f"[QUALITY GATE] Cannot auto-fix — no DOM selector resolved. Issues: {issues}")
        code = warning + code

    # Trim imports to only what's actually used in the final script
    code = _minimal_imports(code)

    return code


# ── Helper: only import what the generated script actually calls ──────────────
_ALL_SCRIPT_HELPERS = [
    "get_secret", "find_in_frames", "resilient_fill", "resilient_click",
    "click_and_upload_file", "maybe_login", "resilient_extract_table",
    "generate_report", "check_certificate",
]

def _minimal_imports(code: str) -> str:
    """
    Replace the monolithic script_helpers import block with only the helpers
    that are actually called in the generated code.
    Also trims TIMEOUT_FILL_MS from the config import when unused.
    Applied to both fast-path and LLM-generated scripts.
    """
    used = [h for h in _ALL_SCRIPT_HELPERS if (h + "(") in code]

    _import_pat = re.compile(
        r"from src\.utils\.script_helpers import \([^)]+\)",
        re.DOTALL,
    )
    if used:
        if len(used) == 1:
            new_import = "from src.utils.script_helpers import " + used[0]
        else:
            sep = ",\n    "
            new_import = (
                "from src.utils.script_helpers import (\n    "
                + sep.join(used)
                + ",\n)"
            )
        code = _import_pat.sub(new_import, code)
    else:
        code = _import_pat.sub("", code)

    # Trim TIMEOUT_FILL_MS from config import if not actually used in the script body
    _config_line = "from src.utils.config import TIMEOUT_TABLE_WAIT_MS, TIMEOUT_FILL_MS"
    if _config_line in code:
        # Check if TIMEOUT_FILL_MS is used anywhere other than its own import line
        _code_without_import = code.replace(_config_line, "")
        if "TIMEOUT_FILL_MS" not in _code_without_import:
            code = code.replace(_config_line, "from src.utils.config import TIMEOUT_TABLE_WAIT_MS")
    return code


# ── Fast-path: pure navigation+extraction tasks, no LLM needed ───────────────

_COMPLEX_ACTIONS = {
    "click_element",
    "click_element_by_text",
    "input_text",
    "upload_file",
    "select_dropdown_option",
    "drag_drop",
}


def _try_template_fast_path(
    cleaned_history: str,
    all_urls_ordered: list,
    tbl_registry: dict,
    objective: str,
    run_id: str,
) -> "str | None":
    """
    Fast-path script generator — bypasses LLM entirely when the task is purely
    navigation + table extraction (no clicks, fills, or conditional logic).

    Returns a ready-to-run Python script string, or None if the history contains
    any non-trivial actions that require the LLM.

    Fast-path fires when:
      - Every action in history is ONLY go_to_url / extract_content / scroll / wait / done
      - Every page with table data has a fully-resolved CSS selector
    """
    # ── 1. Detect non-trivial actions in history ─────────────────────────────
    try:
        _h = json.loads(cleaned_history)
        _steps = _h.get("steps") if isinstance(_h, dict) else _h
        for _s in (_steps or []):
            for _act in (_s.get("actions") or []):
                if isinstance(_act, dict):
                    _atype = next(iter(_act.keys()), "")
                    if _atype in _COMPLEX_ACTIONS:
                        logger.info(
                            "[FAST-PATH] Action %r in history — falling back to LLM", _atype
                        )
                        return None
    except Exception:
        return None

    # ── 1b. Scan objective text for interaction verbs ─────────────────────────
    # If the task description mentions clicks, saves, selects, etc., the history
    # may be incomplete (e.g. crashed before those actions) but the LLM still
    # needs to generate them. Fall back rather than produce an incomplete script.
    _INTERACTION_VERBS = (
        " click", "select ", " radio", " save", " submit", " fill",
        " upload", " drag", " dropdown", " configure", " open ",
        " check ", " uncheck", " toggle", " press ", " type ",
    )
    _obj_lower = (objective or "").lower()
    if any(v in _obj_lower for v in _INTERACTION_VERBS):
        logger.info("[FAST-PATH] Objective has interaction verbs — falling back to LLM")
        return None

    # ── 2. Filter URLs to only those relevant to the objective ────────────────
    # The agent may have visited pages (e.g. Admin Servers) that are NOT part of
    # this task. Include a URL only if:
    #   a) the objective text contains a word from the URL's query string, OR
    #   b) the URL has table data in the registry AND those headers appear in
    #      the objective text, OR
    #   c) there is only one non-login URL (nothing to filter against)
    #
    # This prevents stale agent history pages from appearing as extra steps.
    _LOGIN_FRAGS = ("login", "otdsws", "signin", "auth/login", "llworkspace", "about:blank")
    _content_urls = [u for u in all_urls_ordered
                     if not any(f in u.lower() for f in _LOGIN_FRAGS)]

    # Column names so generic they appear in nearly every table — not useful
    # for distinguishing which page an objective is about.
    _GENERIC_COLS = {
        "status", "name", "id", "type", "description", "date", "actions",
        "errors", "last update", "primary", "title", "size", "owner",
        "created", "modified", "version", "state", "enabled", "running",
    }

    def _url_relevant(url: str) -> bool:
        """Return True if this URL is relevant to the objective."""
        if not url:
            return False
        # Always include if it's the only non-login URL
        if len(_content_urls) == 1:
            return True
        _obj_lc = (objective or "").lower()
        # ── Check 1: URL query-string fragment appears in objective ──────────
        # e.g. "distributedagent" in objective → DA URL is relevant
        # Also matches full key=value: "objtype=148" in objective → Admin relevant
        from urllib.parse import urlparse as _up2
        qs = _up2(url).query.lower()
        for frag in qs.split("&"):
            val = frag.split("=")[-1]          # e.g. "distributedagent", "148"
            # Match on the value OR the full key=value pair (e.g. "objtype=148")
            if (len(val) > 4 and val in _obj_lc) or (len(frag) > 4 and frag in _obj_lc):
                return True
        # ── Check 2: a DISTINCTIVE (non-generic) table header is in objective ─
        # e.g. "Worker ID" in objective → Worker Agents table URL is relevant
        # Generic cols like "Status", "Name" are excluded — they match too broadly
        page_tbls = tbl_registry.get(url) or []
        for tbl in page_tbls:
            for hdr in (tbl.get("headers") or tbl.get("required") or []):
                if (len(hdr) > 4
                        and hdr.lower() not in _GENERIC_COLS
                        and hdr.lower() in _obj_lc):
                    return True
        return False

    # Only filter when there are multiple content pages — if 0 or 1, pass through
    if len(_content_urls) > 1:
        filtered_urls = [u for u in all_urls_ordered if
                         any(f in u.lower() for f in _LOGIN_FRAGS) or _url_relevant(u)]
        n_dropped = len(all_urls_ordered) - len(filtered_urls)
        if n_dropped:
            logger.info("[FAST-PATH] Filtered %d URL(s) not relevant to objective", n_dropped)
            for u in all_urls_ordered:
                if u not in filtered_urls:
                    logger.info("[FAST-PATH]   dropped: %s", u)
        all_urls_ordered = filtered_urls

    # ── 3. Resolve per-page table data ────────────────────────────────────────
    # Priority: registry tables with columns (dynamic) > _KNOWN_PAGE_SELECTORS (static fallback)
    pages = []  # [{url, tables: [{selector, columns}]}]
    for url in all_urls_ordered:
        # Check registry first — it's live data
        tbls = tbl_registry.get(url) or []
        
        # If registry is empty, maybe _KNOWN_PAGE_SELECTORS has something
        if not tbls:
            kp = _known_page_info(url)
            if kp and kp.get("selector"):
                tbls = [{"selector": kp["selector"], "headers": kp.get("columns") or [], "required": kp.get("columns") or []}]

        resolved = []
        for t in tbls:
            sel  = (t.get("selector") or "").strip()
            cols = list(t.get("required") or t.get("headers") or [])
            if not sel:
                # If we have no selector for this table, we can't do fast-path for the whole page
                logger.info("[FAST-PATH] %s: unresolved selector — falling back to LLM", url)
                return None
            if not cols:
                # Skip tables without columns (likely layout tables) instead of failing
                continue

            # ── Refinement: Filter columns and tables based on objective ───
            # User doesn't want "everything" if they only asked for one specific table.
            filtered = []
            
            # 1. Prepare word sets from objective
            import re as _re
            all_text = (objective or "").lower()
            obj_words = set(_re.findall(r'\w+', all_text))
            
            # 1b. Identify "Goal Words" specifically from extraction sentences
            # We split by line/period and look for "Extract", "Get", etc.
            goal_words = set()
            extraction_verbs = {"extract", "get", "read", "scrape", "table", "report", "columns"}
            sentences = _re.split(r'[\n.]', all_text)
            for s in sentences:
                s_words = set(_re.findall(r'\w+', s))
                if s_words & extraction_verbs:
                    # Ignore very common generic words even in goal sentences
                    goal_words.update(s_words - {"the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at", "by", "with", "columns", "table", "both"})

            # Identity markers frequently asked for implicitly as "Name" or "ID"
            id_markers = {"id", "name", "title", "agent", "worker", "server", "host"}
            # Generic words that often cause false positives across tables
            generic_words = {"id", "name", "status", "description", "last", "update", "value", "type", "date", "time", "both"}
            specific_goal_words = goal_words - generic_words
            
            has_specific_match = False
            for c in cols:
                c_lc = c.lower()
                c_words = set(_re.findall(r'\w+', c_lc))
                
                # Check 1: Overlap between column words and objective words?
                matched = bool(c_words & obj_words)
                
                # Check 1b: Is it a SPECIFIC match (non-generic)?
                if bool(c_words & specific_goal_words):
                    has_specific_match = True
                
                # Check 2: If user asked for generic "Name" or "ID", treat identity cols as matches.
                is_requested_id = (("name" in obj_words or "id" in obj_words) and 
                                 any(m in c_lc for m in id_markers))
                
                if matched or is_requested_id:
                    filtered.append(c)
            
            # Table-Level Strictness:
            # If we have SPECIFIC goal words (like "Worker"), and this table ONLY matched 
            # via generic words (like "Status" or "ID"), then it's probably NOT requested.
            is_noise = specific_goal_words and not has_specific_match
            
            if filtered and not is_noise:
                resolved.append({"selector": sel, "columns": filtered})
            elif filtered and not specific_goal_words:
                # Fallback: if user only used generic words in prompt, keep everything that matched.
                resolved.append({"selector": sel, "columns": filtered})

        pages.append({"url": url, "tables": resolved})

    if not pages:
        return None

    # ── 5. Derive login base URL ──────────────────────────────────────────────
    from urllib.parse import urlparse, urlunparse

    try:
        _h2 = json.loads(cleaned_history)
        _s2 = _h2.get("steps") if isinstance(_h2, dict) else _h2
        _first = next(
            (s["url"] for s in (_s2 or []) if s.get("url", "").startswith("http")),
            pages[0]["url"],
        )
    except Exception:
        _first = pages[0]["url"]
    _pu = urlparse(_first)
    login_url = urlunparse((_pu.scheme, _pu.netloc, _pu.path, "", "", ""))

    # ── 6. Build body lines ───────────────────────────────────────────────────
    # Use explicit string concatenation to avoid f-string/quote conflicts when
    # the generated lines themselves contain string literals.
    L = []
    all_row_vars = []  # [(var_name, section_title, columns)]

    # Login block
    L.append("        # ── Step 1: Login")
    L.append("        await page.goto(" + repr(login_url) + ", timeout=TIMEOUT_TABLE_WAIT_MS)")
    L.append("        await maybe_login(page)")
    L.append('        await capture_step("Login", "Logged in successfully")')
    L.append("")

    for pi, pg in enumerate(pages):
        url = pg["url"]
        tables = pg["tables"]
        step = pi + 2

        L.append("        # ── Step " + str(step) + ": " + url)
        L.append("        await page.goto(" + repr(url) + ", timeout=TIMEOUT_TABLE_WAIT_MS)")

        if not tables:
            L.append('        await page.wait_for_load_state("networkidle")')
            L.append('        await capture_step("Page ' + str(pi + 1) + '", "Navigated")')
        else:
            for ti, tbl in enumerate(tables):
                sel = tbl["selector"]
                cols = tbl["columns"]
                multi = len(tables) > 1
                var = ("rows_p" + str(pi + 1) + "_t" + str(ti + 1)) if multi else ("rows_p" + str(pi + 1))
                title = "Page " + str(pi + 1) + (" Table " + str(ti + 1) if multi else "")
                all_row_vars.append((var, title, cols))

                L.append(
                    "        await page.wait_for_selector("
                    + repr(sel)
                    + ", timeout=TIMEOUT_TABLE_WAIT_MS)"
                )
                L.append(
                    "        "
                    + var
                    + " = await resilient_extract_table(page, "
                    + repr(sel)
                    + ", "
                    + repr(cols)
                    + ")"
                )
                L.append('        print(f"[*] ' + title + ': {len(' + var + ')} rows")')
                L.append('        for _r in ' + var + ': print(f"  {_r}")')

            L.append(
                '        await capture_step("Step '
                + str(step)
                + '", f"'
                + str(len(tables))
                + ' table(s) from page '
                + str(pi + 1)
                + '")'
            )
        L.append("")

    # ── 7. Report block ───────────────────────────────────────────────────────
    L.append("        # ── Report")
    L.append(
        '        _summary = "### Execution Summary\\n'
        '| Step | Action | Output |\\n'
        '| :--- | :--- | :--- |\\n"'
    )
    L.append("        for _i, _s in enumerate(steps, 1):")
    L.append(
        "            _summary += f\"| {_i} | {_s['action']} | {_s['output']} |\\n\""
    )
    L.append("")

    for var, title, cols in all_row_vars:
        ncols = len(cols)
        # Header row: "| Col1 | Col2 |\n| :--- | :--- |\n"
        L.append(
            "        _"
            + var
            + '_hdr = "| " + " | ".join('
            + repr(cols)
            + ') + " |\\n| " + " | ".join([":---"] * '
            + str(ncols)
            + ') + " |\\n"'
        )
        # Data rows
        L.append(
            "        _"
            + var
            + '_body = "".join('
        )
        L.append(
            '            f"| {chr(32).join(str(_r.get(_c, chr(78)+chr(47)+chr(65))) for _c in '
            + repr(cols)
            + ")} |\\n\""
        )
        # Simpler: use explicit join
        L[-2] = (
            "        _"
            + var
            + "_body = \"\".join("
        )
        L[-1] = (
            "            \"| \" + \" | \".join(str(_r.get(_c, \"N/A\")) for _c in "
            + repr(cols)
            + ") + \" |\\n\""
        )
        L.append("            for _r in " + var)
        L.append("        )")
        L.append(
            "        "
            + var
            + '_section = "### '
            + title
            + '\\n" + _'
            + var
            + "_hdr + _"
            + var
            + "_body"
        )
        L.append("")

    if all_row_vars:
        sections_expr = ' + "\\n" + '.join(v + "_section" for v, _, _ in all_row_vars)
    else:
        sections_expr = '"(no data extracted)"'

    L.append('        _details = "## Step Screenshots\\n"')
    L.append("        for _i, _s in enumerate(steps, 1):")
    L.append(
        "            _details += f\"### Step {_i}: {_s['action']}\\n![Step {_i}]({_s['shot']})\\n\""
    )
    L.append("")
    L.append("        _report_sections = " + sections_expr)
    L.append(
        '        _full_report = ('
        ' "# Automation Report\\n\\n"'
        " + _summary + \"\\n\\n\""
        " + _report_sections + \"\\n\\n\""
        " + _details"
        " )"
    )
    L.append(
        '        await generate_report("Consolidated Report", True,'
        " output=_full_report, print_terminal=False)"
    )

    body = "\n".join(L)

    # ── 8. Assemble final script ──────────────────────────────────────────────
    # Fast-path scripts only use: maybe_login, resilient_extract_table, generate_report
    # Wrap objective as # comments (it may span multiple lines)
    _obj_lines = ["# " + ln for ln in (objective or "").splitlines()] or ["# (no objective)"]
    script_lines = [
        "# ⚡ FAST-PATH generated (no LLM) — selectors/columns from agent history",
        "# Run ID: " + run_id,
    ] + _obj_lines + [
        "",
        "import asyncio",
        "import sys",
        "import os",
        "from playwright.async_api import async_playwright",
        "",
        "current_dir = os.getcwd()",
        'if "web-ui" not in current_dir and os.path.exists(os.path.join(current_dir, "web-ui")):',
        '    sys.path.append(os.path.join(current_dir, "web-ui"))',
        "else:",
        "    sys.path.append(current_dir)",
        "",
        "from src.utils.script_helpers import (",
        "    maybe_login, resilient_extract_table, generate_report,",
        ")",
        "from src.utils.config import TIMEOUT_TABLE_WAIT_MS",
        "",
        "steps = []",
        "script_dir = os.path.dirname(os.path.abspath(__file__))",
        "",
        "",
        "async def run():",
        "    async with async_playwright() as p:",
        '        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])',
        "        context = await browser.new_context(no_viewport=True)",
        "        page = await context.new_page()",
        "",
        "        async def capture_step(action, output):",
        "            idx = len(steps) + 1",
        '            shot_path = os.path.join(script_dir, f"shot_{idx}.png")',
        "            await page.screenshot(path=shot_path)",
        '            steps.append({"action": action, "output": output, "shot": f"shot_{idx}.png"})',
        "",
        body,
        "        await browser.close()",
        "",
        "",
        'if __name__ == "__main__":',
        "    asyncio.run(run())",
        "",
    ]
    script = "\n".join(script_lines)

    logger.info(
        "[FAST-PATH] ⚡ Generated script for %d pages / %d tables — LLM skipped",
        len(pages),
        len(all_row_vars),
    )
    return script


def _build_cert_script(target_url: str, scenario: str) -> str:
    """Return a ready-to-run Playwright script that calls check_certificate(page)."""
    return f'''import asyncio
import sys
import os

current_dir = os.getcwd()
if "web-ui" not in current_dir and os.path.exists(os.path.join(current_dir, "web-ui")):
    sys.path.append(os.path.join(current_dir, "web-ui"))
else:
    sys.path.append(current_dir)

from playwright.async_api import async_playwright
from src.utils.script_helpers import (
    get_secret, maybe_login, generate_report, check_certificate,
)
from src.utils.config import TIMEOUT_TABLE_WAIT_MS


async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--start-maximized", "--remote-debugging-port=9223"],
        )
        context = await browser.new_context(no_viewport=True)
        page    = await context.new_page()

        await page.goto({repr(target_url)}, timeout=60000)
        await maybe_login(page)
        
        await page.bring_to_front()
        await asyncio.sleep(1.0)

        # check_certificate uses CDP + direct TLS handshake.
        # On Windows (headed) it also captures the native OS popup via
        # Windows UI Automation — no hardcoded coordinates.
        result = await check_certificate(page)

        cert = result["cert_info"]
        output = (
            f"Host: {{result['hostname']}}\\n"
            f"Valid: {{result['is_valid']}} | Secure: {{result['is_secure']}}\\n"
            f"Issued to: {{cert.get('subject_cn')}} ({{cert.get('subject_org')}})\\n"
            f"Issued by: {{cert.get('issuer_cn')}}\\n"
            f"Valid from {{cert.get('valid_from')}} to {{cert.get('valid_to')}} "
            f"({{cert.get('days_remaining')}}d remaining)\\n"
            f"Protocol: {{cert.get('protocol')}} | Cipher: {{cert.get('cipher')}}\\n"
            f"Expired: {{cert.get('is_expired')}}"
        )

        await generate_report(
            {repr(scenario)},
            result["is_valid"],
            screenshot=result["screenshot"] if not result.get("native_capture") else None,
            output=output,
            panels={{
                "panel1": result.get("screenshot_site_info"),
                "panel2": result.get("screenshot_security"),
                "panel3": result.get("screenshot_cert"),
            }} if result.get("native_capture") else None,
        )
        await browser.close()


if __name__ == "__main__":
    asyncio.run(run())
'''


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
        cleaned_history = clean_history_json(history_data, objective=objective)
        logger.info(f"Cleaned History Context (First 500 chars): {cleaned_history[:500]}...")

        # ── Cert task short-circuit: skip LLM, emit check_certificate script ──
        import re as _re
        _CERT_RE = r'\b(ssl|tls|certificate|https cert|cert detail|cert valid|lock icon|connection secure)\b'
        _obj_text = (objective or history_data.get('task', '')).lower()
        if _re.search(_CERT_RE, _obj_text, _re.IGNORECASE):
            # Extract target URL from history
            _url = None
            try:
                _h = json.loads(cleaned_history)
                _steps = _h.get('steps') if isinstance(_h, dict) else _h
                for _s in (_steps or []):
                    _u = _s.get('url', '')
                    if _u and _u.startswith('http') and 'about:blank' not in _u:
                        _url = _u
                        break
            except Exception:
                pass
            _url = _url or "# REPLACE_WITH_TARGET_URL"
            cert_script = _build_cert_script(_url, objective or "SSL Certificate Check")
            
            if not output_path:
                output_path = str(Path(history_path).parent / f"{run_id}_LLM.py")
                
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cert_script)
            logger.info(f"[cert-shortcircuit] Cert script written to: {output_path}")
            
            return output_path, cert_script


        try:
            debug_path = history_path.replace('.json', '_cleaned.json')
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_history)
            logger.info(f"Debug cleaned history saved to: {debug_path}")
        except: pass

        # Prepare user prompt
        # (LLM is initialized later, after dynamic num_ctx is computed from actual prompt size)

        # Build element data context from captured elements
        element_context = _build_element_data_context()

        # Build action→selector map from history (explicit lookup, prevents hallucination)
        selector_map = _build_action_selector_map(cleaned_history)
        if selector_map:
            logger.info(f"[selector-map] Built action→selector reference ({selector_map.count(chr(10))} lines)")

        # Create Prompt
        user_prompt = f"Task Objective: {objective or 'Complete the automation task'}\n\n"

        # ── Multi-page / Multi-table navigation plan ──────────────────────────
        # Uses _build_page_table_registry() to inject EVERY table on EVERY page
        # with its exact CSS selector and required columns.
        #
        # For each page visited in the history the LLM receives:
        #   Page N: <url>
        #     Table 1: selector="<css>" | headers=[...] | row_count=N
        #       → wait_for_selector("<css>", timeout=TIMEOUT_TABLE_WAIT_MS)
        #       → resilient_extract_table(page, "<css>", [<exact cols>])
        #     Table 2: ...
        #
        # This eliminates ALL column and selector guessing by the LLM.
        try:
            _tbl_registry = _build_page_table_registry(cleaned_history, run_id=run_id)
            _login_kws = ("login", "otdsws", "signin", "auth/login", "about:blank")

            # Also collect non-table pages (they still need a goto block)
            _h2 = json.loads(cleaned_history)
            _steps2 = _h2.get('steps') if isinstance(_h2, dict) else _h2
            _all_urls_ordered = []
            _seen_u2 = set()
            for _s2 in (_steps2 or []):
                # FIX: dedup by FULL URL, not base URL.
                # OTCS pages all share the same base (cs.exe) but differ only in query
                # string (func=distributedagent vs func=ll&objtype=148 etc.).
                # Using split('?')[0] collapsed them all into one entry → only the
                # first page ever appeared in the navigation plan.
                _full_u2 = _s2.get('url') or ''
                if not _full_u2 or _full_u2 in _seen_u2:
                    continue
                if any(kw in _full_u2.lower() for kw in _login_kws):
                    continue
                _seen_u2.add(_full_u2)
                _all_urls_ordered.append(_full_u2)

            if _all_urls_ordered:
                plan_lines = [
                    "## NAVIGATION PLAN",
                    "# ⚠️  CRITICAL RULES:",
                    "# 1. Generate ONE page.goto() block per page listed below.",
                    "# 2. Call resilient_extract_table() ONCE per table listed — "
                    "use the EXACT selector and EXACT columns shown.",
                    "# 3. NEVER mix selectors across pages.",
                    "# 4. For pages with NO TABLE listed: use wait_for_load_state('networkidle').",
                    "",
                ]

                for _pi, _purl in enumerate(_all_urls_ordered):
                    # Match URL to registry key (full URL first)
                    _ptbls = _tbl_registry.get(_purl) or []
                    # Fallback: match registry key whose query string is contained in purl
                    # (handles minor URL variations like extra params)
                    if not _ptbls:
                        for _rk, _rv in _tbl_registry.items():
                            if _rk in _purl or _purl in _rk:
                                _ptbls = _rv
                                break

                    # Fallback 2: _KNOWN_PAGE_SELECTORS — for pages with no history data
                    # (agent visited the page but didn't call extract_content, or old run)
                    if not _ptbls:
                        kp = _known_page_info(_purl)
                        if kp and kp.get("selector"):
                            _ptbls = [{
                                "selector":  kp["selector"],
                                "headers":   kp.get("columns") or [],
                                "required":  kp.get("columns") or [],
                                "row_count": 0,
                            }]
                            logger.info(
                                f"[NAV-PLAN] {_purl}: "
                                f"injected known-page selector {kp['selector']!r}"
                            )

                    plan_lines.append(f"  Page {_pi+1}: {_purl}")

                    if _ptbls:
                        for _ti, _tbl in enumerate(_ptbls):
                            _tsel = _tbl.get("selector", "")
                            _tcols = _tbl.get("required") or _tbl.get("headers") or []
                            _rows  = _tbl.get("row_count", 0)
                            _hdrs  = _tbl.get("headers") or []

                            if _tsel:
                                plan_lines.append(
                                    f"    Table {_ti+1}: selector={_tsel!r}"
                                    + (f" | headers={_hdrs}" if _hdrs else "")
                                    + (f" | row_count={_rows}" if _rows else "")
                                )
                                plan_lines.append(
                                    f"      → await page.wait_for_selector({_tsel!r}, "
                                    f"timeout=TIMEOUT_TABLE_WAIT_MS)"
                                )
                                plan_lines.append(
                                    f"      → rows_{_ti+1} = await resilient_extract_table"
                                    f"(page, {_tsel!r}, {_tcols!r})"
                                )
                            else:
                                # No selector — use networkidle + auto-discover
                                plan_lines.append(
                                    f"    Table {_ti+1}: selector=unknown"
                                    + (f" | columns={_tcols}" if _tcols else "")
                                )
                                plan_lines.append(
                                    f"      → await page.wait_for_load_state('networkidle')"
                                )
                                plan_lines.append(
                                    f"      → rows_{_ti+1} = await resilient_extract_table"
                                    f"(page, 'table', {_tcols!r})"
                                )
                    else:
                        plan_lines.append(
                            f"    (No table data captured — use wait_for_load_state"
                            f"('networkidle') and interact as needed)"
                        )
                    plan_lines.append("")  # blank line between pages

                user_prompt += "\n".join(plan_lines) + "\n\n"
                logger.info(
                    f"[PROMPT] Injected {len(_all_urls_ordered)}-page / "
                    f"{sum(len(v) for v in _tbl_registry.values())}-table navigation plan"
                )

                # ── ⚡ FAST-PATH: skip LLM if all selectors/columns are known ──
                # Attempt this BEFORE initializing the LLM to save the most time.
                _fast_code = _try_template_fast_path(
                    cleaned_history=cleaned_history,
                    all_urls_ordered=_all_urls_ordered,
                    tbl_registry=_tbl_registry,
                    objective=objective or "",
                    run_id=run_id,
                )
                if _fast_code is not None:
                    if not output_path:
                        output_path = str(Path(history_path).parent / f"{run_id}_LLM.py")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(_fast_code)
                    logger.info(f"[FAST-PATH] ⚡ Script written (no LLM used): {output_path}")
                    return output_path, _fast_code

        except Exception as _pe:
            logger.warning(f"[PROMPT] Could not build page/table plan: {_pe}")

        if selector_map:
            user_prompt += (
                f"{selector_map}\n\n"
                "⚠️  CRITICAL: For EVERY resilient_fill, resilient_click, wait_for_selector, "
                "and page.locator call you MUST use the exact selector from the "
                "SELECTOR REFERENCE TABLE above. Do NOT invent selectors.\n\n"
            )
        user_prompt += f"Generate a Playwright script for the following execution history (Run ID: {run_id}):\n\n{cleaned_history}"
        if element_context:
            user_prompt += element_context
            logger.info(f"[PROMPT] Injected {element_context.count(chr(10))} lines of captured element data into LLM prompt")

        # ── Dynamic num_ctx: actual prompt size + 2K headroom ──────────────
        # Avoids loading a 32K context window for a 5K prompt.
        # 1 token ≈ 4 chars; add 2048 tokens for the generated output.
        _prompt_chars  = len(SYSTEM_PROMPT) + len(user_prompt)
        _prompt_tokens = (_prompt_chars // 4) + 2048
        _dynamic_ctx   = max(4096, min(_prompt_tokens, SCRIPT_GEN_NUM_CTX))
        if _dynamic_ctx < SCRIPT_GEN_NUM_CTX:
            logger.info(
                f"[LLM] Dynamic num_ctx: {_dynamic_ctx} "
                f"(prompt ~{_prompt_chars//4} tok, cap was {SCRIPT_GEN_NUM_CTX})"
            )

        # Initialize LLM now that we know the actual context size needed
        logger.info(f"Initializing LLM: {provider}/{model_name} (num_predict={SCRIPT_GEN_NUM_PREDICT}, num_ctx={_dynamic_ctx})")
        llm = get_llm_model(
            provider=provider,
            model_name=model_name,
            temperature=SCRIPT_GEN_TEMPERATURE,
            num_predict=SCRIPT_GEN_NUM_PREDICT,
            num_ctx=_dynamic_ctx,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        # Generate with OOM retry fallback (spec §3.G)
        logger.info("Sending request to LLM (this may take a minute)...")
        # Build fallback chain: primary model first, then config-defined fallbacks
        _oom_fallback_models = [model_name] + OOM_FALLBACK_MODELS
        # ctx sizes: primary uses dynamic_ctx, fallbacks use OOM_FALLBACK_CTX values
        _oom_fallback_ctx    = [_dynamic_ctx] + list(OOM_FALLBACK_CTX)
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
            code = _postprocess_generated_code(code, cleaned_history, tbl_registry=_tbl_registry)
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