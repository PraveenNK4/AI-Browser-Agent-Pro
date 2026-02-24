"""
Generate a Playwright automation script from agent history JSON with DOM context.
Auto-extracts table selectors from DOM snapshots for maximum accuracy.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import defaultdict
import logging

# Import table extraction compiler (Optional - handled if missing)
try:
    from .table_extraction_compiler import (
        generate_extraction_with_failure_semantics,
        classify_extraction_strategy
    )
    TABLE_COMPILER_AVAILABLE = True
except ImportError:
    TABLE_COMPILER_AVAILABLE = False
    # Define dummy functions to prevent NameError if used in conditionals check (though we guard usage)
    def generate_extraction_with_failure_semantics(*args, **kwargs): return None, {}
    def classify_extraction_strategy(*args, **kwargs): return "unknown"
    # logging.warning("table_extraction_compiler not found. Using fallback extraction logic.")

logger = logging.getLogger(__name__)


def is_env_variable(text: str) -> bool:
    """Check if text contains {{VAR}} or ${VAR} format."""
    if not text: return False
    return bool(re.search(r'{{[A-Za-z0-9_]+}}|^\$[A-Za-z0-9_]+$', text))

def get_env_variable_name(text: str) -> str:
    """Extract variable name from {{VAR}} or ${VAR}."""
    match = re.search(r'{{([A-Za-z0-9_]+)}}|^\$([A-Za-z0-9_]+)$', text)
    if match:
        return match.group(1) or match.group(2)
    return text

def is_positional_selector(selector: str) -> bool:
    """
    Check if a selector contains positional pseudo-classes that make it brittle.
    These should be avoided per the Script Guide unless absolutely necessary.
    """
    if not selector:
        return False
    positional_patterns = [
        r':nth-child\(',
        r':nth-of-type\(',
        r':first-child',
        r':last-child',
        r':nth-last-child\(',
        r':nth-last-of-type\(',
    ]
    return any(re.search(pattern, selector) for pattern in positional_patterns)


def get_selector_stability_score(selector: str, attrs: Dict[str, Any]) -> int:
    """
    Rate selector stability based on Script Guide principles.
    Higher score = more stable and preferred.
    """
    if not selector:
        return 0
    
    # ID selectors - highest priority
    if selector.startswith('#') and not is_positional_selector(selector):
        return 100
    
    # Data attributes and ARIA - very stable
    if '[data-' in selector or '[aria-' in selector:
        return 80
    
    # Semantic classes (common stable patterns)
    semantic_patterns = [
        r'\.btn-', r'\.button-', r'\.link-', r'\.nav-', r'\.header-', 
        r'\.footer-', r'\.content-', r'\.container-', r'\.form-', r'\.input-',
        r'\.submit', r'\.login', r'\.save', r'\.cancel', r'\.close'
    ]
    if any(re.search(pattern, selector) for pattern in semantic_patterns):
        if not is_positional_selector(selector):
            return 90
    
    # Text-based selectors (has-text, role, etc.)
    if ':has-text(' in selector or '[role=' in selector or '[type=' in selector:
        return 70
    
    # Check if it's a positional selector - lowest priority
    if is_positional_selector(selector):
        return 10
    
    # Structural selectors without nth-child
    return 50


def get_all_selector_candidates(dom: Dict[str, Any], is_visible: bool = False) -> List[Tuple[str, int, str]]:
    """
    Gather all plausible selector candidates from DOM context, sorted by stability score.
    """
    attrs = dom.get('attributes', {})
    selectors = dom.get('selectors', {})
    tag_name = dom.get('tagName', '').lower()
    text_content = dom.get('textContent', '').strip() or attrs.get('innerText', '').strip() or attrs.get('value', '').strip()
    
    candidates = []
    
    # ===== COMPREHENSIVE ELEMENT DATA INTEGRATION =====
    comprehensive = dom.get('comprehensive', {})
    if comprehensive and 'all_selectors' in comprehensive:
        for selector_info in comprehensive['all_selectors']:
            selector = selector_info.get('selector', '')
            stability = selector_info.get('stability', 0)
            sel_type = selector_info.get('type', '')
            
            if selector and stability > 0:
                candidates.append((selector, stability, sel_type))
    
    # ===== FALLBACK TO ORIGINAL LOGIC =====
    if is_visible and attrs.get('id'):
        candidates.append((f"#{attrs['id']}", 100, 'id-visible'))
    
    if attrs.get('id'):
        # Penalize if it looks like a dynamic ID (e.g. contains numbers/hashes)
        score = 95
        if any(c.isdigit() for c in attrs['id']) and len(attrs['id']) > 8:
            score = 60
        candidates.append((f"#{attrs['id']}", score, 'id'))
    
    for attr_name, attr_value in attrs.items():
        if not attr_value: continue
        if attr_name.startswith('data-') or attr_name.startswith('aria-') or attr_name == 'title':
            # Penalize cstabindex (often dynamic)
            score = 80
            if 'cstabindex' in attr_name: score = 50
            attr_selector = f'{tag_name}[{attr_name}="{attr_value}"]'
            candidates.append((attr_selector, score, f'{attr_name}-attribute'))
    
    if tag_name == 'input' and attrs.get('type'):
        if attrs.get('name'):
            type_selector = f'input[type="{attrs["type"]}"][name="{attrs["name"]}"]'
            candidates.append((type_selector, 85, 'type-name-combo'))
        else:
            type_selector = f'input[type="{attrs["type"]}"]'
            candidates.append((type_selector, 70, 'type-only'))
    
    # HIGH PRIORITY TEXT MATCHING for navigation/buttons
    if text_content and tag_name in ['button', 'a', 'span', 'div', 'label']:
        safe_text = text_content.replace('"', '\\"').replace("'", "\\'")[:50]
        if len(safe_text) > 2:
            # If it's a known interactive tag, boost text score
            score = 85 if tag_name in ['button', 'a'] else 70
            text_selector = f'{tag_name}:has-text("{safe_text}")'
            candidates.append((text_selector, score, 'text-content'))
            
            # Specific Playwright Locators
            loc_selector = f'text="{safe_text}"'
            candidates.append((loc_selector, score + 1, 'pw-text'))
    
    if attrs.get('class_list') or attrs.get('classList'):
        classes = attrs.get('class_list') or attrs.get('classList') or []
        semantic_classes = [c for c in classes if len(c) > 3 and c not in ['btn', 'button', 'active', 'selected']]
        if semantic_classes:
            class_selector = f'{tag_name}.{".".join(semantic_classes[:2])}'
            candidates.append((class_selector, 75, 'semantic-class'))
    
    if selectors.get('preferred'):
        pref_selector = selectors['preferred']
        score = get_selector_stability_score(pref_selector, attrs)
        candidates.append((pref_selector, score, 'preferred'))
    
    if selectors.get('css'):
        css_selector = selectors['css']
        score = get_selector_stability_score(css_selector, attrs)
        candidates.append((css_selector, score, 'css-path'))
    
    if selectors.get('xpath'):
        xpath_selector = selectors['xpath']
        candidates.append((xpath_selector, 30, 'xpath'))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c[0] not in seen:
            unique_candidates.append(c)
            seen.add(c[0])
            
    return unique_candidates

def select_best_selector(dom: Dict[str, Any], is_visible: bool = False) -> Optional[str]:
    """
    Select the best selector from DOM context following Script Guide principles.
    """
    candidates = get_all_selector_candidates(dom, is_visible)
    if candidates:
        return candidates[0][0]
    return dom.get('tagName', '').lower() or 'div'


def select_best_table_with_ollama(tables_info: List[Dict[str, Any]], extraction_goal: str) -> Optional[int]:
    """Use Ollama LLM to select the most relevant table based on extraction goal."""
    try:
        import httpx
        
        tables_summary = []
        for i, table in enumerate(tables_info):
            summary = f"Table {i+1}:\n"
            summary += f"  Selector: {table.get('selector', 'unknown')}\n"
            summary += f"  Headers: {', '.join(table.get('headers', [])[:10])}\n"
            summary += f"  Sample row: {', '.join(table.get('sample_row', [])[:10])}\n"
            summary += f"  Row count: {table.get('row_count', 0)}\n"
            tables_summary.append(summary)
        
        prompt = f"""You are analyzing a webpage to select the best table for data extraction.

Extraction Goal: {extraction_goal}

Available tables:
{chr(10).join(tables_summary)}

Which table best matches the extraction goal? Respond with ONLY the table number (1, 2, 3, etc.) or "NONE" if no table is relevant.
Your response:"""

        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            },
            timeout=10.0
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '').strip().upper()
            import re
            match = re.search(r'(\d+)', result)
            if match:
                table_num = int(match.group(1))
                if 1 <= table_num <= len(tables_info):
                    return table_num - 1
            return None
        return None
            
    except Exception as e:
        logger.debug(f"Ollama table selection unavailable: {e}")
        return None


def is_env_variable(value: str) -> bool:
    return bool(re.match(r'^\{\{[A-Z_]+\}\}$', value))


def get_env_variable_name(value: str) -> str:
    match = re.match(r'^\{\{([A-Z_]+)\}\}$', value)
    return match.group(1) if match else value


def load_dom_snapshot(snapshot_path: Path) -> Dict[str, Any]:
    try:
        with open(snapshot_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def extract_table_selectors(elements: List[Dict[str, Any]], selector_type: str = 'css') -> Dict[str, Any]:
    result = {
        'selector_type': selector_type,
        'table_root': None,
        'header_row': [],
        'data_rows': [],
        'column_selectors': [],
        'all_cells': [],
    }
    
    table_cells = []
    
    for el in elements:
        tag = el.get('identity', {}).get('tagName', '').lower()
        text = el.get('selector_provenance', {}).get('text', '').strip()
        selector = el.get('selector_provenance', {}).get(selector_type, '')
        attrs = el.get('attribute_fingerprint', {})
        classes = attrs.get('class_list', [])
        el_id = attrs.get('id', '')
        
        if tag in ['td', 'th']:
            cell_data = {
                'text': text[:100],
                'selector': selector,
                'id': el_id,
                'classes': classes,
                'tag': tag,
                'order': el.get('identity', {}).get('order', 0)
            }
            table_cells.append(cell_data)
            result['all_cells'].append({
                'text': text[:50],
                'selector': selector,
                'tag': tag,
                'order': el.get('identity', {}).get('order', 0)
            })
        
        if tag == 'table' and not result['table_root']:
            result['table_root'] = {
                'selector': selector,
                'id': el_id,
                'classes': classes
            }
    
    th_cells = [c for c in table_cells if c['tag'] == 'th']
    if th_cells:
        result['header_row'] = th_cells
    elif table_cells:
        result['header_row'] = [c for c in table_cells[:10] if len(c['text']) < 30]
    
    td_cells = [c for c in table_cells if c['tag'] == 'td']
    if td_cells:
        result['data_rows'] = td_cells
    
    num_cols = len(result['header_row']) if result['header_row'] else max(1, len(td_cells) // 5 if td_cells else 1)
    if num_cols > 0 and td_cells:
        for col_idx in range(num_cols):
            col_cells = [c for i, c in enumerate(td_cells) if i % num_cols == col_idx]
            result['column_selectors'].append({
                'column': col_idx + 1,
                'selectors': [c['selector'] for c in col_cells[:3]],
                'sample_values': [c['text'] for c in col_cells[:3]]
            })
    
    return result


def find_best_table_selector(snapshot_dir: Path, run_id: str) -> Optional[Dict[str, Any]]:
    # Fix: ensure we handle case with no snapshots gracefully
    if not snapshot_dir:
        return None
        
    snapshot_dir_path = snapshot_dir / run_id if run_id else snapshot_dir
    
    if not snapshot_dir_path.exists():
        return None
    
    snapshot_files = sorted(snapshot_dir_path.glob('*.json'), reverse=True)
    if not snapshot_files:
        return None
    
    snapshot = load_dom_snapshot(snapshot_files[0])
    elements = snapshot.get('elements', [])
    
    if not elements:
        return None
    
    selectors = extract_table_selectors(elements, 'css')
    return selectors if (selectors.get('table_root') or selectors.get('data_rows')) else None


def generate_action_code(action: Dict[str, Any], action_name: str, table_selectors: Optional[Dict[str, Any]] = None, snapshot_dir: Optional[Path] = None, run_id: Optional[str] = None, step_index: Optional[int] = None) -> Optional[str]:
    dom = action.get('dom_context', {})
    attrs = dom.get('attributes', {})
    tag = dom.get('tagName', '').lower()
    element_desc = tag
    if attrs.get('id'):
        element_desc += f"#{attrs['id']}"
    elif attrs.get('name'):
        element_desc += f"[name={attrs['name']}]"
    elif attrs.get('textContent'):
        element_desc += f" ({attrs['textContent'][:20]})"
    
    if action_name == 'go_to_url':
        params = action.get(action_name, {})
        url = params.get('url', '')
        
        # Clean URL (resolve secret if needed)
        if is_env_variable(url):
            var_name = get_env_variable_name(url)
            url_code = f"get_secret('{var_name}')"
            url_log = f"URL from {var_name}"
        else:
            url_code = f'"{url}"'
            url_log = url

        # Include Smart Login block only for the first navigation
        login_block = ""
        if step_index == 1:
            from src.utils.config import (
                VAULT_CREDENTIAL_PREFIX,
                LOGIN_USER_SELECTORS, LOGIN_PASS_SELECTORS, LOGIN_SUBMIT_SELECTORS,
            )
            user_sel_list  = repr(LOGIN_USER_SELECTORS)
            pass_sel_list  = repr(LOGIN_PASS_SELECTORS)
            submit_sel_list = repr(LOGIN_SUBMIT_SELECTORS)
            vault_pfx      = VAULT_CREDENTIAL_PREFIX
            login_block = f'''
            # Smart Login Check (Golden Standard)
            print("[*] Checking if login is required...")
            is_login_page = await page.evaluate('() => document.querySelectorAll("input[type=\\'password\\']").length > 0')
            if is_login_page:
                print("  [!] Login page detected. Performing Vault-based login...")
                user = get_secret("{vault_pfx}_USERNAME")
                passwd = get_secret("{vault_pfx}_PASSWORD")

                if user and passwd and not user.startswith("{{{{"):
                    # Fill User
                    for sel in {user_sel_list}:
                        if await page.locator(sel).count() > 0:
                            await page.fill(sel, user)
                            break
                    # Fill Password
                    for sel in {pass_sel_list}:
                        if await page.locator(sel).count() > 0:
                            await page.fill(sel, passwd)
                            break
                    # Submit
                    for sel in {submit_sel_list}:
                        if await page.locator(sel).count() > 0:
                            await page.click(sel)
                            break
                    print("  [*] Waiting for login redirect...")
                    await page.wait_for_load_state("networkidle")
                    await asyncio.sleep(2)
                else:
                    print("  [!] Skip Login: Credentials missing or placeholder in Vault")
            else:
                print("  [*] Already logged in or no password field detected.")
'''
        
        return f'            print("[+] Navigating to: {url_log}")\n            await page.goto({url_code})\n            await page.wait_for_load_state("domcontentloaded"){login_block}\n            print("[+] Page loaded successfully")'
    
    elif action_name == 'input_text':
        params = action.get(action_name, {})
        text = params.get('text', '').replace('<secret>', '').replace('</secret>', '')
        is_visible = dom.get('state', {}).get('visible', False)
        selector = select_best_selector(dom, is_visible)
        
        if not selector:
            return None

        # Escape quotes and backslashes in selector for generated code
        safe_selector = selector.replace('\\', '\\\\').replace('"', '\\"')

        # Robustness: Filter invalid elements for input_text
        valid_input_tags = ['input', 'textarea', 'select']
        is_content_editable = attrs.get('contenteditable') in ['true', ''] or attrs.get('role') in ['textbox']
        
        # If it's a generic tag (div, span) and NOT editable, skip it or warn
        tag = dom.get('tagName', '').lower()
        if tag not in valid_input_tags and not is_content_editable:
            return f"            # Skipped input_text on non-editable {tag} ({safe_selector})"

        candidates = get_all_selector_candidates(dom, is_visible)
        selectors_json = json.dumps([c[0] for c in candidates])
        
        if is_env_variable(text):
            var_name = get_env_variable_name(text)
            return f'''            # Resilient input from Vault
            try:
                cred = get_secret("{var_name}")
                if cred:
                    await resilient_fill(page, {selectors_json}, cred, desc="{element_desc}")
                else:
                    print("[!] Warning: {var_name} not set in vault or environment")
            except Exception as e:
                print(f"[!] Skipping input: {element_desc} error: {{e}}")'''
        else:
            text_preview = text[:50] + "..." if len(text) > 50 else text
            escaped_text = text.replace('"', '\\"')
            return f'''            # Resilient input
            try:
                await resilient_fill(page, {selectors_json}, "{escaped_text}", desc="{element_desc}")
            except Exception as e:
                print(f"[!] Skipping input: {element_desc} error: {{e}}")'''
    
    elif action_name in ['click_element', 'click_element_by_index']:
        attrs = dom.get('attributes', {})
        selectors = dom.get('selectors', {})
        is_visible = dom.get('state', {}).get('visible', False)
        tag_name = dom.get('tagName', '').lower()
        element_type = attrs.get('type', '').lower()

        candidates = get_all_selector_candidates(dom, is_visible)
        
        # Check if it's likely a login/submission field (input/password)
        if tag_name == 'input' and element_type in ['password', 'text']:
            selectors_json = json.dumps([c[0] for c in candidates])
            return f'''            try:
                success = await resilient_click(page, {selectors_json}, desc="submission field")
                if success:
                    await page.keyboard.press("Enter")
                    await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception as e:
                print(f"[!] Skipping submission: {{e}}")'''
        
        # Standard resilient click
        selectors_json = json.dumps([c[0] for c in candidates])
        return f'''            try:
                await resilient_click(page, {selectors_json}, desc="{element_desc}")
            except Exception as e:
                print(f"[!] Skipping click: {element_desc} error: {{e}}")'''
    
    elif action_name == 'click_element_by_text':
        params = action.get(action_name, {})
        text = params.get('text', '')
        escaped_text = text.replace('"', '\\"')
        
        # Use text-based matching directly (clean path)
        return f'''            try:
                print("[+] Clicking element with text: {escaped_text}")
                await page.get_by_text("{escaped_text}", exact=False).first.click(timeout=5000)
                await page.wait_for_load_state("networkidle")
            except Exception as e:
                print(f"[!] Skipping text-click: {escaped_text} not found (may already be logged in)")'''

        # Heuristic for "Add Item" removed (kept generic)
    
    elif action_name == 'extract_content':
        params = action.get(action_name, {})
        goal_text = params.get('goal', '') if isinstance(params, dict) else ''
        requires_hover = isinstance(goal_text, str) and 'hover' in goal_text.lower()
        
        if dom and TABLE_COMPILER_AVAILABLE:
            try:
                code, schema = generate_extraction_with_failure_semantics(dom, should_raise=False)
                strategy = schema.get('type', 'unknown')
                if strategy not in ['fallback', 'error', 'unknown']:
                    hover_block = ''
                    if requires_hover:
                        attrs = dom.get('attributes', {})
                        selectors = dom.get('selectors', {})
                        selector = None
                        if attrs.get('id'): selector = f"#{attrs['id']}"
                        elif selectors.get('preferred'): selector = selectors['preferred']
                        elif selectors.get('css'): selector = selectors['css']
                        if selector:
                            safe_selector = selector.replace('\\', '\\\\').replace('"', '\\"')
                            hover_block = f'''            await page.wait_for_selector("{safe_selector}")
            hover_el = await page.query_selector("{safe_selector}")
            if hover_el:
                await hover_el.hover()
                await page.wait_for_timeout(2000)'''
                    
                    schema_json = json.dumps(schema, indent=2)
                    schema_commented = '\n'.join(f'            # {line}' for line in schema_json.split('\n'))
                    return f'''            # Extraction strategy: {strategy}
{schema_commented}
{hover_block}
{code}'''
            except Exception:
                pass
        
        extraction_goal = action.get(action_name, {}).get('goal', 'extract data') if isinstance(action.get(action_name, {}), dict) else 'extract data'
        dom_elements = []
        tables_info = []
        
        if snapshot_dir and run_id:
            try:
                run_snapshots_dir = Path(snapshot_dir) / run_id
                if run_snapshots_dir.exists():
                    snapshot_files = sorted(run_snapshots_dir.glob('*.json'))
                    for snapshot_file in snapshot_files:
                        try:
                            snapshot_data = load_dom_snapshot(snapshot_file)
                            meta = snapshot_data.get('metadata', {})
                            if meta.get('reason', '') == 'pre_action:extract_content':
                                dom_elements = snapshot_data.get('elements', [])
                                break
                        except: continue
            except: pass
        
        if not dom_elements:
            dom_elements = dom.get('elements', []) if isinstance(dom, dict) else []
        
        for idx, el in enumerate(dom_elements):
            tag = el.get('identity', {}).get('tagName', '').lower() if isinstance(el.get('identity', {}), dict) else el.get('tagName', '').lower()
            if tag == 'table':
                # Simplified robust logic for extraction
                outerhtml = el.get('integrity', {}).get('outerHTML', '') if isinstance(el.get('integrity', {}), dict) else el.get('outerHTML', '')
                selectors = el.get('selector_provenance', {}) if isinstance(el.get('selector_provenance', {}), dict) else el.get('selectors', {})
                selector = selectors.get('css') or selectors.get('preferred') or f'table:nth-of-type({idx+1})'
                
                # Mock analysis
                import re as re_module
                headers = [h.strip()[:50] for h in re_module.findall(r'<th[^>]*>([^<]+)</th>', outerhtml)[:10]]
                tables_info.append({
                    'selector': selector,
                    'headers': headers,
                    'row_count': len(re_module.findall(r'<tr[^>]*>', outerhtml))
                })

        best_selector = None
        if tables_info:
            best_idx = select_best_table_with_ollama(tables_info, extraction_goal)
            if best_idx is not None:
                best_selector = tables_info[best_idx]['selector']

        if best_selector:
            return f'''            print("[+] Extracting content from table: {best_selector}")
            table_element = await page.query_selector('{best_selector}')
            if table_element:
                extracted_content = await parse_table_data(table_element)
            else:
                all_tables = await page.query_selector_all('table')
                extracted_content = await parse_table_data(all_tables[0]) if all_tables else await page.evaluate('() => document.body.innerText')'''
        else:
            return f'''            print("[*] Fallback extraction")
            all_tables = await page.query_selector_all('table')
            if all_tables:
                extracted_content = await parse_table_data(all_tables[0])
            else:
                extracted_content = await page.evaluate('() => document.body.innerText')'''
    
    elif action_name == 'safe_check_and_click':
        params = action.get(action_name, {})
        index = params.get('index', 0)
        is_visible = dom.get('state', {}).get('visible', False)
        candidates_raw = get_all_selector_candidates(dom, is_visible)
        selectors_json = json.dumps([c[0] for c in candidates_raw])
        
        return f'''            try:
                selectors = {selectors_json}
                element = None
                found_sel = None
                for sel in selectors:
                    try:
                        # Playwright's locator handles XPath automatically if it starts with // or xpath=
                        # We explicitly check for / to ensure it's treated as XPath
                        if sel.startswith("/") or sel.startswith("xpath="):
                            locator = page.locator(sel).first
                        else:
                            locator = page.locator(sel).first
                        
                        if await locator.count() > 0:
                            element = locator # Use the locator directly
                            found_sel = sel
                            break
                    except Exception:
                        # Continue to next selector if this one fails
                        continue
                
                if element:
                    is_checked = await element.evaluate("el => el.checked || el.getAttribute('aria-checked') === 'true'")
                    if is_checked:
                        print(f"[+] Element {{found_sel}} already checked")
                    else:
                        await element.click()
                        await asyncio.sleep(0.3)
                else:
                    print("[!] Skipping check-and-click: No candidate selectors found")
            except Exception as e:
                print(f"[!] Skipping check-and-click error: {{e}}")'''

    elif action_name == 'upload_file':
        params = action.get(action_name, {})
        path = params.get('path', '')
        index = params.get('index', 0)
        is_visible = dom.get('state', {}).get('visible', False)
        
        # Robust Upload Logic
        escaped_path = path.replace('\\', '\\\\').replace('"', '\\"')
        
        candidates_raw = get_all_selector_candidates(dom, is_visible)
        candidates = [c[0] for c in candidates_raw]
        
        # Enhanced text fallbacks for generic uploads
        text_content = dom.get('textContent', '').strip() or dom.get('attributes', {}).get('innerText', '').strip()
        if text_content and len(text_content) < 30:
            safe_text = text_content.replace('"', '\\"').replace("'", "\\'")
            candidates.append(f"text='{safe_text}'")
            candidates.append(f"a:has-text('{safe_text}')")
            candidates.append(f".csui-toolitem:has-text('{safe_text}')")
            # Dropdown specific (Content Server pattern)
            candidates.append(f"li[data-csui-command='add'] a:has-text('{safe_text}')")
            safe_text = text_content.replace('"', '\\"').replace("'", "\\'")
            candidates.append(f"text='{safe_text}'")
            candidates.append(f"a:has-text('{safe_text}')")
            candidates.append(f".csui-toolitem:has-text('{safe_text}')")
            # Dropdown specific (Content Server pattern)
            candidates.append(f"li[data-csui-command='add'] a:has-text('{safe_text}')")
            
        attr_title = dom.get('attributes', {}).get('title')
        if attr_title:
            candidates.append(f"[title='{attr_title}']")
            
        # Deduplicate candidates
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                unique_candidates.append(c)
                seen.add(c)
                
        # Convert list to string representation for the generated code
        candidates_repr = json.dumps(unique_candidates)
        
        return f'''            # Robust Upload using helper
            print("[*] Attempting upload...")
            upload_selectors = {candidates_repr}
            file_path = "{escaped_path}"
            
            success = await click_and_upload_file(page, upload_selectors, file_path)
            if success:
                print("[+] Upload triggered successfully. Waiting for commit dialog...")
                await asyncio.sleep(5)
                # Check for typical OpenText "Add" buttons
                add_btn = page.locator("button.binf-btn-primary:has-text('Add')")
                if await add_btn.count() > 0 and await add_btn.is_visible():
                    print("  [+] Found 'Add' button. Clicking to commit upload...")
                    await add_btn.click()
                    await page.wait_for_load_state("networkidle")
                await asyncio.sleep(2)
            else:
                print("[!] Upload failed - selectors not found")'''

    elif action_name == 'done':
        params = action.get(action_name, {})
        text = params.get('text', '')
        success = params.get('success', True)
        status = "Success" if success else "Failed"
        escaped_text = text.replace('"', '\\"').replace("\n", " ")
        return f'            print("[+] Task Finished: {status}")\n            print("[*] Result: {escaped_text}")'

    # Generic fallback
    if not dom: return None
    candidates_raw = get_all_selector_candidates(dom, is_visible=False)
    selectors_json = json.dumps([c[0] for c in candidates_raw])
    
    return f'''            try:
                await resilient_click(page, {selectors_json}, desc="fallback element")
            except Exception as e:
                print(f"[*] Fallback click failed: {{e}}")'''


def generate_playwright_script(history_data: Union[Path, List, Dict], snapshot_dir: Optional[Path] = None, auto_extract: bool = True) -> str:
    """
    Generate a standalone Playwright script.
    Accepts Path to history JSON, or raw history list/dict key.
    """
    # Load data if Path, otherwise use directly
    if isinstance(history_data, Path):
        run_id = history_data.stem
        with history_data.open('r', encoding='utf-8') as f:
            data = json.load(f)
    elif isinstance(history_data, str) and os.path.exists(history_data):
        run_id = Path(history_data).stem
        with open(history_data, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # Raw data passed
        run_id = "generated_script"
        data = history_data

    steps = data.get('history', []) if isinstance(data, dict) else data
    
    if snapshot_dir and auto_extract:
        table_selectors = find_best_table_selector(snapshot_dir, run_id)
    else:
        table_selectors = None
    
    all_actions = []
    # Collect results to filter out failed steps
    for step_idx, step in enumerate(steps):
        if not isinstance(step, dict): continue
        
        # Handle results
        results = step.get('result', [])
        
        # Handle actions
        actions = []
        if 'model_output' in step:
            actions = step['model_output'].get('action', [])
        elif hasattr(step, 'model_output'):
            actions = step.model_output.action
            
        for action_idx, action in enumerate(actions):
            # Check if this specific action failed
            has_error = False
            if action_idx < len(results):
                res = results[action_idx]
                if isinstance(res, dict) and res.get('error'):
                    has_error = True
            
            if has_error:
                # Aggressive filtering: Skip ANY action that resulted in an error.
                # The generated script should only follow the successful "golden path".
                continue

            # Handle ActionModel object or dict
            if hasattr(action, 'model_dump'):
                action_dict = action.model_dump(exclude_unset=True)
            else:
                action_dict = action
                
            if not isinstance(action_dict, dict):
                continue

            for action_name, params in action_dict.items():
                if params is None or action_name in ['dom_context', 'action_explanation']:
                    continue
                
                # Create a clean action dict for the generator
                full_action = {action_name: params}
                
                # Safely attach dom_context
                if hasattr(action, 'dom_context'):
                    full_action['dom_context'] = action.dom_context
                elif isinstance(action, dict) and 'dom_context' in action:
                     full_action['dom_context'] = action['dom_context']
                
                all_actions.append((full_action, action_name))

    script_lines = [
        'import asyncio', 'import os', 'import re', 'import sys', 'import time',
        'from playwright.async_api import async_playwright',
        'from docx import Document', 'from docx.shared import Inches', 'import io',
        '',
        '# Add project root to path for imports',
        'current_dir = os.path.dirname(os.path.abspath(__file__))',
        'project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))',
        'if project_root not in sys.path: sys.path.append(project_root)',
        '',
        'try:',
        '    from src.utils.vault import vault',
        'except ImportError:',
        '    vault = None',
        '    print("Warning: Could not import vault. Secrets loading may fail.")',
        '',
        '# Robust Helper functions',
        'def clean_text(raw: str) -> str:',
        '    """Remove script/style blocks and collapse whitespace for readability."""',
        '    if not raw: return ""',
        '    text = re.sub(r"<script.*?>.*?</script>", " ", raw, flags=re.IGNORECASE | re.DOTALL)',
        '    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)',
        '    text = re.sub(r"\\s+", " ", text).strip()',
        '    return text',
        '',
        'async def parse_table_data(element) -> str:',
        '    """',
        '    Parse table data dynamically - uses JavaScript for reliable structure detection.',
        '    Works with any table on any website. Can handle tables with title rows.',
        '    """',
        '    try:',
        '        # Use JavaScript to parse table structure directly in the browser',
        '        parse_script = """',
        '            (table) => {',
        '                const rows = Array.from(table.querySelectorAll("tr"));',
        '                if (rows.length < 2) return { rows: [] };',
        '                ',
        '                // DATA CLEANING: Filter out rows that are just titles or navigation',
        '                const candidateRows = rows.filter(row => {',
        '                    const cells = row.querySelectorAll("td, th");',
        '                    if (cells.length === 0) return false;',
        '                    ',
        '                    // Skip if it\'s a "title row" (single cell with colspan > 1)',
        '                    if (cells.length === 1 && cells[0].hasAttribute("colspan") && parseInt(cells[0].getAttribute("colspan")) > 1) {',
        '                        return false; ',
        '                    }',
        '                    ',
        '                    // Skip if it seems to be just a navigation bar (mostly links)',
        '                    const links = row.querySelectorAll("a");',
        '                    const text = row.innerText.trim();',
        '                    if (links.length > 2 && text.length < links.length * 20) {',
        '                        return false;',
        '                    }',
        '                    return true;',
        '                });',
        '                ',
        '                if (candidateRows.length < 2) return { headers: [], dataRows: [] };',
        '                ',
        '                // DETECT HEADER ROW',
        '                let headerRow = null;',
        '                let headerIdx = -1;',
        '                ',
        '                // Strategy 1: Look for <th> elements',
        '                for (let i = 0; i < Math.min(candidateRows.length, 3); i++) {',
        '                    const ths = candidateRows[i].querySelectorAll("th");',
        '                    if (ths.length >= 2) {',
        '                        headerRow = candidateRows[i];',
        '                        headerIdx = i;',
        '                        break;',
        '                    }',
        '                }',
        '                ',
        '                // Strategy 2: Look for header-like row',
        '                if (!headerRow) {',
        '                    for (let i = 0; i < Math.min(candidateRows.length, 3); i++) {',
        '                        const cells = candidateRows[i].querySelectorAll("td");',
        '                        if (cells.length < 2) continue;',
        '                        ',
        '                        const texts = Array.from(cells).map(c => c.innerText.trim());',
        '                        ',
        '                        const hasKeywords = texts.some(t => ',
        '                            ["name", "status", "type", "id", "date", "size", "count", "description", "processes", "running", "errors"]',
        '                            .some(kw => t.toLowerCase().includes(kw))',
        '                        );',
        '                        ',
        '                        const allShort = texts.every(t => t.length < 40);',
        '                        ',
        '                        if (allShort && hasKeywords) {',
        '                            headerRow = candidateRows[i];',
        '                            headerIdx = i;',
        '                            break;',
        '                        }',
        '                    }',
        '                }',
        '                ',
        '                // Strategy 3: Default to first candidate row',
        '                if (!headerRow && candidateRows.length > 0) {',
        '                    headerRow = candidateRows[0];',
        '                    headerIdx = 0;',
        '                }',
        '                ',
        '                // EXTRACT HEADERS',
        '                const headerCells = headerRow.querySelectorAll("th, td");',
        '                const headers = Array.from(headerCells).map(c => c.innerText.trim().split("\\\\n")[0]);',
        '                ',
        '                // EXTRACT DATA',
        '                const dataRows = [];',
        '                for (let i = headerIdx + 1; i < candidateRows.length; i++) {',
        '                    const cells = candidateRows[i].querySelectorAll("td, th");',
        '                    if (cells.length === 0) continue;',
        '                    ',
        '                    const values = Array.from(cells).map(c => c.innerText.trim().split("\\\\n")[0]);',
        '                    ',
        '                    if (values.every(v => !v)) continue;',
        '                    dataRows.push(values);',
        '                }',
        '                ',
        '                return { headers, dataRows };',
        '            }',
        '        """',
        '        ',
        '        table_data = await element.evaluate(parse_script)',
        '        ',
        '        headers = table_data.get("headers", [])',
        '        data_rows = table_data.get("dataRows", [])',
        '        ',
        '        if not data_rows:',
        '            text = await element.inner_text()',
        '            return text[:500] if len(text) > 500 else text',
        '        ',
        '        output_lines = []',
        '        output_lines.append("=" * 60)',
        '        output_lines.append(f"COLS: {\' | \'.join(headers)}")',
        '        output_lines.append("=" * 60)',
        '        ',
        '        for row in data_rows:',
        '            output_lines.append("")',
        '            for i, value in enumerate(row):',
        '                if value:',
        '                    col_name = headers[i] if i < len(headers) and headers[i] else f"Col{i+1}"',
        '                    output_lines.append(f"  {col_name}: {value}")',
        '        ',
        '        output_lines.append("")',
        '        output_lines.append(f"Total: {len(data_rows)} row(s)")',
        '        ',
        '        return \'\\\\n\'.join(output_lines)',
        '        ',
        '    except Exception as e:',
        '        try:',
        '            return await element.inner_text()',
        '        except:',
        '            return f"Parse error: {str(e)}"',
        '',
        'async def click_and_upload_file(page, selectors, file_path):',
        '    """',
        '    Robustly clicks an element and handles the resulting file chooser.',
        '    Includes scrolling, stability delays, and direct input fallbacks.',
        '    """',
        '    if not os.path.exists(file_path):',
        '        print(f"  ❌ File not found: {file_path}")',
        '        return False',
        '        ',
        '    for sel in selectors:',
        '        try:',
        '            target_loc = None',
        '            if sel.startswith("text="):',
        '                 target_loc = page.get_by_text(sel.replace("text=", "", 1), exact=False).first',
        '            elif sel.startswith("/") and not sel.startswith("//"):',
        '                 target_loc = page.locator("xpath=" + sel).first',
        '            else:',
        '                 target_loc = page.locator(sel).first',
        '',
        '            if await target_loc.count() > 0:',
        '                # Ensure element is visible and stable (Resilient 5s check)',
        '                await target_loc.scroll_into_view_if_needed(timeout=5000)',
        '                await asyncio.sleep(0.5)',
        '                ',
        '                async with page.expect_file_chooser(timeout=8000) as fc_info:',
        '                    await target_loc.click(timeout=5000)',
        '                file_chooser = await fc_info.value',
        '                await file_chooser.set_files(file_path)',
        '                print(f"  ✅ File uploaded via chooser: {sel}")',
        '                return True',
        '            except Exception as e:',
        '                print(f"[*] Choice {sel} failed: {e}")',
        '                ',
        '    # Fallback: Check for direct file inputs',
        '    try:',
        '        for sel in ["input[type=\'file\']", "input[name=\'file\']", "#file"]:',
        '            if await page.locator(sel).count() > 0:',
        '                print(f"  Fallback: Setting file directly on {sel}")',
        '                await page.locator(sel).first.set_input_files(file_path)',
        '                print("  ✅ File uploaded via direct input fallback!")',
        '                return True',
        '    except Exception as e2:',
        '        print(f"  ❌ Fallback failed: {e2}")',
        '        ',
        '    return False',
        '',
        'async def resilient_click(page, selectors, desc="element"):',
        '    """Try multiple selectors for a click until one works or all fail."""',
        '    for sel in selectors:',
        '        try:',
        '            # Check if selector is a text-based Playwright locator',
        '            target_loc = None',
        '            if sel.startswith("text="):',
        '                target_loc = page.get_by_text(sel.replace("text=", "", 1), exact=False).first',
        '            elif sel.startswith("/") and not sel.startswith("//"):',
        '                 target_loc = page.locator("xpath=" + sel).first',
        '            else:',
        '                target_loc = page.locator(sel).first',
        '            ',
        '            if await target_loc.count() > 0:',
        '                await target_loc.scroll_into_view_if_needed(timeout=5000)',
        '                await target_loc.click(timeout=5000)',
        '                await page.wait_for_load_state("networkidle", timeout=5000)',
        '                print(f"  ✅ Clicked {desc} via {sel}")',
        '                return True',
        '        except Exception:',
        '            continue',
        '    print(f"  ❌ Failed to click {desc} after trying {len(selectors)} selectors")',
        '    return False',
        '',
        'async def resilient_fill(page, selectors, text, desc="input"):',
        '    """Try multiple selectors to fill an input until one works or all fail."""',
        '    for sel in selectors:',
        '        try:',
        '            locator = page.locator(sel).first',
        '            if await locator.count() > 0:',
        '                await locator.scroll_into_view_if_needed(timeout=5000)',
        '                await locator.fill(text, timeout=5000)',
        '                await locator.dispatch_event("input")',
        '                await locator.dispatch_event("change")',
        '                print(f"  ✅ Filled {desc} via {sel}")',
        '                return True',
        '        except Exception:',
        '            continue',
        '    print(f"  ❌ Failed to fill {desc} after trying {len(selectors)} selectors")',
        '    return False',
        '',
        'def get_secret(key):',
        '    """Resolve {{KEY_FIELD}} strictly from Vault (no ENV fallback as per requirement)."""',
        '    # Strip braces if present',
        '    raw_key = key.replace("{{", "").replace("}}", "")',
        '    ',
        '    # Standard project vault resolution',
        '    if vault and "_" in raw_key:',
        '        parts = raw_key.split("_")',
        '        vault_key = parts[0]',
        '        field_key = "_".join(parts[1:]).lower()',
        '        ',
        '        # Try finding credentials for this key',
        '        creds = vault.get_credentials(vault_key) or vault.get_credentials(vault_key.lower())',
        '        if creds:',
        '            if field_key in creds: return creds[field_key]',
        '            if field_key.upper() in creds: return creds[field_key.upper()]',
        '            ',
        '    # If not found or vault unavailable, return placeholder (allows debugging without leak)',
        '    return f"{{{{MISSING_VAULT_KEY_{raw_key}}}}}"',
        '',
        'async def run_automation():',
        f'    print("Running automation: {run_id}")',
        '    async with async_playwright() as p:',
        '        browser = await p.chromium.launch(headless=False)',
        '        page = await browser.new_page()',
        '        steps_data = []',
        '        try:'
    ]
    
    for i, (action, name) in enumerate(all_actions):
        step_num = i + 1
        code = generate_action_code(action, name, table_selectors, snapshot_dir, run_id, step_index=step_num)
        if code:
            step_num = i + 1
            script_lines.append(f'            # Step {step_num}: {name}')
            script_lines.append(f'            print(f"Step {step_num}: {name}")')
            script_lines.append(code)
            script_lines.append('            time.sleep(1)')
            # Capture state
            script_lines.append(f'            screenshot_bytes = await page.screenshot()')
            # Extract current state text or result for the output table
            script_lines.append(f'            captured_output = f"URL: {{page.url}}"')
            if name == 'extract_content':
                script_lines.append(f'            if "extracted_content" in locals(): captured_output = extracted_content')
            
            script_lines.append(f'            steps_data.append({{"step": {step_num}, "action": "{name}", "screenshot": screenshot_bytes, "output": captured_output}})')

    script_lines.extend([
        '        except Exception as e:',
        '            print(f"Error: {e}")',
        '            await page.screenshot(path="failure_debug.png")',
        '        finally: await browser.close()',
        '',
        '        # Generate Rich Report (Golden Standard)',
        '        print("\\nGenerating professional report...")',
        '        doc = Document()',
        '        doc.add_heading("Playwright Automation Report", 0)',
        '        ',
        '        # 1. Execution Summary',
        '        doc.add_heading("Execution Summary", level=1)',
        '        summary_para = doc.add_paragraph()',
        '        summary_para.add_run("Script ID: ").bold = True',
        f'        summary_para.add_run("{run_id}.py")',
        '        ',
        '        summary_para = doc.add_paragraph()',
        '        summary_para.add_run("Status: ").bold = True',
        '        summary_para.add_run("Execution Complete")',
        '        ',
        '        summary_para = doc.add_paragraph()',
        '        summary_para.add_run("Total Steps: ").bold = True',
        '        summary_para.add_run(str(len(steps_data)))',
        '        ',
        '        # 2. Captured Outputs Table',
        '        doc.add_heading("Captured Outputs", level=1)',
        '        outputs = [entry for entry in steps_data if entry.get("output")]',
        '        if outputs:',
        '            table = doc.add_table(rows=1, cols=3)',
        '            table.style = "Table Grid"',
        '            hdr = table.rows[0].cells',
        '            hdr[0].text = "Step"',
        '            hdr[1].text = "Action"',
        '            hdr[2].text = "Output Details"',
        '            for o in outputs:',
        '                row = table.add_row().cells',
        '                row[0].text = str(o.get("step"))',
        '                row[1].text = o.get("action")',
        '                # Clean up output for table cell (shorten if needed)',
        '                out_txt = str(o.get("output"))',
        '                row[2].text = out_txt[:1000] + ("..." if len(out_txt) > 1000 else "")',
        '        else:',
        '            doc.add_paragraph("No significant outputs captured.")',
        '            ',
        '        # 3. Automation Steps with Screenshots',
        '        doc.add_heading("Detailed Automation Steps", level=1)',
        '        for entry in steps_data:',
        '            step_num = entry["step"]',
        '            action = entry["action"]',
        '            scr = entry["screenshot"]',
        '            out = entry["output"]',
        '            ',
        '            doc.add_heading(f"Step {step_num}: {action}", level=2)',
        '            ',
        '            # Action Details Table',
        '            details_table = doc.add_table(rows=2 if out else 1, cols=2)',
        '            details_table.style = "Table Grid"',
        '            row = details_table.rows[0].cells',
        '            row[0].text = "Action"',
        '            row[1].text = action',
        '            ',
        '            if out:',
        '                row = details_table.rows[1].cells',
        '                row[0].text = "Captured Result"',
        '                row[1].text = str(out)',
        '            ',
        '            if scr:',
        '                try:',
        '                    doc.add_paragraph("Screenshot:")',
        '                    doc.add_picture(io.BytesIO(scr), width=Inches(6))',
        '                except: doc.add_paragraph("[Screenshot Error]")',
        '        doc.add_paragraph()',
        '        ',
        f'        report_path = os.path.join(current_dir, "{run_id}_verification_report.docx")',
        '        doc.save(report_path)',
        '        print(f"✅ Professional report saved to: {report_path}")',
        '',
        'if __name__ == "__main__":',
        '    asyncio.run(run_automation())'
    ])
    
    return '\n'.join(script_lines)

if __name__ == '__main__':
    # Default execution for testing
    pass