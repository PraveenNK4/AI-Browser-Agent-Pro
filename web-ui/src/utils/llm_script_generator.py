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

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        context = await browser.new_context(no_viewport=True)
        page = await context.new_page()

        # Navigate and handle login
        await page.goto("<URL from history>", timeout=60000)
        await maybe_login(page)

        # Wait for target element, then extract / interact
        table_selector = "<selector from dom_context>"
        await page.wait_for_selector(table_selector, timeout=TIMEOUT_TABLE_WAIT_MS)
        rows = await resilient_extract_table(page, table_selector, ["Col1", "Col2"])
        output = "\\n".join([
            f"Identity: {row.get('Server') or row.get('Name') or 'Unknown'}, "
            f"Status: {row.get('Status', '')}, Errors: {row.get('Errors', '')}"
            for row in rows
        ])
        await generate_report("Task Name", bool(rows), output=output)
        await browser.close()

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
3.  **Login hook** — After every `await page.goto(...)` for a protected URL, call `await maybe_login(page)`.
    `maybe_login` is smart: it does nothing if no login form is present.
4.  **Table extraction**:
    - ALWAYS use `await resilient_extract_table(page, table_selector, required_columns)`.
    - `table_selector` MUST come from `dom_elements.selectors` in the history (`preferred` > `css` > `xpath`).
      NEVER use a bare `"table"` or a login-field selector like `"input[type='password']"`.
    - `required_columns` MUST include the columns from `required_columns_hint` PLUS an identity
      key (`"Server"` or `"Name"`) if not already present.
    - After `await maybe_login(page)`, wait for `table_selector`, NOT a login field:
      `await page.wait_for_selector(table_selector, timeout=TIMEOUT_TABLE_WAIT_MS)`
5.  **Selectors** — Use `[aria-label]`, `[title]`, or descriptive attributes. Avoid brittle
    positional selectors. For exact-text menus: `re.compile(f"^{{target}}$", re.I)`.
6.  **Uploads** — Use `await click_and_upload_file(page, label, path)` with `TIMEOUT_FILE_CHOOSER_MS`
    (currently {file_chooser_ms}ms).
7.  **Loops** — ALWAYS `elements = await locator.all()` then `for el in elements:`.
    Never `async for`.
8.  **Report output** — Format rows as:
    `f"Identity: {{row.get('Server') or row.get('Name') or 'Unknown'}}, Status: {{row.get('Status', '')}}, ..."`
    Then call `await generate_report(scenario, bool(rows), output=output)`.
9.  **No noise** — Skip failed/retried actions from history. Only generate the successful path.
10. **Imports** — Only import what `run()` actually uses beyond the standard header.
    Common extras: `import re` (if using regex), `import time` (if using sleep).
11. **Certificate / security checks** — NEVER click the browser lock icon (it is above the page
    and unreachable from Playwright). Instead call:
    ```python
    result = await check_certificate(page)
    # result keys: screenshot (PNG bytes), cert_info (dict), is_valid (bool),
    #              is_secure (bool), hostname (str)
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
    await generate_report("Certificate Check", result["is_valid"],
                          screenshot=result["screenshot"], output=output)
    ```
    `check_certificate` uses CDP + direct TLS — no coordinates, no OS-specific UI automation.

### 📝 STRUCTURE
Reproduce this header exactly, then write only `async def run()`:

```python
{script_header}
```

The header imports EVERYTHING from `script_helpers`. Only add extra imports if `run()` needs them
(e.g. `import re` for regex matching, `import time` for explicit waits).
'''

SYSTEM_PROMPT = _build_system_prompt()

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
    def _should_replace(sel_str: str) -> bool:
        s = sel_str.strip().strip('"\'')
        return s == "table" or _is_login_selector(s)

    if selector:
        sel_literal = json.dumps(selector)

        def _fix_wait(m):
            return f'page.wait_for_selector({sel_literal}' if _should_replace(m.group(1)) else m.group(0)
        code = re.sub(r'page\.wait_for_selector\((["\'][^"\']+["\'])', _fix_wait, code)

        def _fix_extract(m):
            return f'resilient_extract_table(page, {sel_literal}' if _should_replace(m.group(1)) else m.group(0)
        code = re.sub(r'resilient_extract_table\(\s*page\s*,\s*(["\'][^"\']+["\'])', _fix_extract, code)

        def _fix_table_var(m):
            return f'table_selector = {sel_literal}' if _should_replace(m.group(1)) else m.group(0)
        code = re.sub(r'table_selector\s*=\s*(["\'][^"\']+["\'])', _fix_table_var, code)

        def _fix_locator(m):
            return f'page.locator({sel_literal}' if _should_replace(m.group(1)) else m.group(0)
        code = re.sub(r'page\.locator\((["\'][^"\']+["\'])', _fix_locator, code)

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
        # Check if the process URL is already in a goto call
        if process_url not in code:
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
    if "maybe_login(page)" not in code:
        issues.append("maybe_login(page) call missing after page.goto")
    for fn in ("resilient_extract_table", "wait_for_selector"):
        m = re.search(rf'{fn}\(\s*(?:page\s*,\s*)?(["\'][^"\']+["\'])', code)
        if m and _is_login_selector(m.group(1).strip("'\"")):
            issues.append(f"{fn} still uses a login-field selector: {m.group(1)!r}")
    if "from src.utils.script_helpers import" not in code:
        issues.append("script_helpers import missing")

    if issues:
        warning = "# ⚠️  POST-PROCESSING WARNINGS:\n" + "\n".join(f"# - {i}" for i in issues) + "\n\n"
        logger.warning("[QUALITY GATE]\n  " + "\n  ".join(issues))
        if selector is None and any("selector" in i for i in issues):
            raise RuntimeError(f"[QUALITY GATE] Cannot auto-fix — no DOM selector resolved. Issues: {issues}")
        code = warning + code

    return code

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
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        context = await browser.new_context(no_viewport=True)
        page    = await context.new_page()

        await page.goto({repr(target_url)}, timeout=60000)
        await maybe_login(page)

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
            screenshot=result["screenshot"],
            output=output,
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