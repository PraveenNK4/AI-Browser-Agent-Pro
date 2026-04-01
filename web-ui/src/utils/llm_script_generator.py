"""
llm_script_generator.py  (v5)
==============================
Generated scripts are fully standalone and read credentials
directly from the vault — no env vars, no helper imports.

A tiny vault-reader snippet is injected into every prompt so
the LLM knows exactly how to access credentials. The LLM then
writes 100% of the script using raw Playwright + python-docx.
"""

from __future__ import annotations

import json
import logging
import os
import re
import argparse
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.llm_provider import get_llm_model
from src.utils.config import (
    SCRIPT_GEN_MODEL,
    SCRIPT_GEN_PROVIDER,
    SCRIPT_GEN_TEMPERATURE,
    SCRIPT_GEN_NUM_CTX,
    VAULT_CREDENTIAL_PREFIX,
)

logger = logging.getLogger(__name__)


# The canonical Chrome frame – injected as a verbatim Python helper into every generated script.
# This is the approved, reference-quality implementation.
_CAPTURE_STEP_CODE = '''\
    import base64 as _b64
    from urllib.parse import urlparse as _urlparse

    def _build_chrome_frame_html(page_png_b64, url, title, favicon_data_uri="", page_width=1280, page_height=900):
        is_secure = url.startswith("https://")
        hostname  = _urlparse(url).hostname or url
        path_part = _urlparse(url).path or "/"
        lock_col  = "#188038" if is_secure else "#c5221f"
        tab_title = title[:30] + "\u2026" if len(title) > 32 else title
        import time as _t
        _tzn = _t.tzname[_t.daylight] if _t.daylight else _t.tzname[0]
        _tz  = "".join(w[0] for w in _tzn.split()) if len(_tzn) > 5 else _tzn
        ts_label = datetime.now().strftime(f"%Y-%m-%d %H:%M:%S {_tz}")
        if is_secure:
            lock_svg = f\'<svg viewBox="0 0 16 16" width="14" height="14" fill="{lock_col}" style="flex-shrink:0"><path d="M8 1a3.5 3.5 0 0 0-3.5 3.5V6H4a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1h-.5V4.5A3.5 3.5 0 0 0 8 1zm2 5H6V4.5a2 2 0 1 1 4 0V6z"/></svg>\'
            secure_label = f\'<span style="font-size:11px;color:{lock_col};font-weight:600;white-space:nowrap">Secure</span>\'
        else:
            lock_svg = f\'<svg viewBox="0 0 20 20" width="14" height="14" fill="{lock_col}" style="flex-shrink:0"><path d="M3.7 3.7a.75.75 0 0 0-1.1 1.1l2.2 2.2H4a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h8a1 1 0 0 1 .6.2l1.7 1.7a.75.75 0 1 0 1.1-1.1L3.7 3.7z"/></svg>\'
            secure_label = f\'<span style="font-size:11px;color:{lock_col};font-weight:600;white-space:nowrap">Not secure</span>\'
        if favicon_data_uri:
            fav_html = f\'<img src="{favicon_data_uri}" width="16" height="16" style="flex-shrink:0;border-radius:2px" onerror="this.style.display=\\\'none\\\'">\' 
        else:
            fav_html = \'<svg viewBox="0 0 16 16" width="16" height="16" fill="#5f6368" style="flex-shrink:0"><circle cx="8" cy="8" r="7" fill="none" stroke="#5f6368" stroke-width="1.2"/><ellipse cx="8" cy="8" rx="3" ry="7" fill="none" stroke="#5f6368" stroke-width="1.2"/><line x1="1" y1="8" x2="15" y2="8" stroke="#5f6368" stroke-width="1.2"/></svg>\'
        css = f"""html,body{{overflow:hidden;}} *{{box-sizing:border-box;margin:0;padding:0;font-family:\'Segoe UI\',-apple-system,sans-serif;}} body{{background:#e8eaed;display:inline-flex;flex-direction:column;}} .window{{display:flex;flex-direction:column;width:{page_width+2}px;box-shadow:0 2px 16px rgba(0,0,0,.25);border-radius:8px 8px 0 0;overflow:hidden;border:1px solid #b0b0b0;}} .tabbar{{background:#dee1e6;display:flex;align-items:flex-end;height:40px;padding-left:10px;flex-shrink:0;}} .tab{{background:#fff;height:34px;padding:0 12px;min-width:180px;max-width:240px;display:flex;align-items:center;gap:8px;border-radius:8px 8px 0 0;font-size:12px;color:#202124;flex-shrink:0;position:relative;}} .tab::before{{content:\'\';position:absolute;bottom:0;left:-8px;width:8px;height:8px;background:transparent;border-bottom-right-radius:8px;box-shadow:2px 2px 0 2px #fff;}} .tab::after{{content:\'\';position:absolute;bottom:0;right:-8px;width:8px;height:8px;background:transparent;border-bottom-left-radius:8px;box-shadow:-2px 2px 0 2px #fff;}} .tab-title{{flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-size:12px;}} .tab-x{{color:#5f6368;font-size:16px;flex-shrink:0;width:18px;height:18px;display:flex;align-items:center;justify-content:center;border-radius:50%;line-height:1;}} .newtab{{background:none;border:none;width:28px;height:28px;margin:0 2px;align-self:center;display:flex;align-items:center;justify-content:center;}} .wc{{margin-left:auto;display:flex;height:40px;align-self:flex-start;flex-shrink:0;}} .wb{{width:46px;height:32px;border:none;background:none;display:flex;align-items:center;justify-content:center;}} .wb svg{{fill:#3c4043;width:10px;height:10px;}} .toolbar{{background:#fff;height:40px;display:flex;align-items:center;padding:0 8px;gap:4px;border-bottom:1px solid #dadce0;flex-shrink:0;}} .nb{{width:30px;height:30px;border-radius:50%;border:none;background:none;display:flex;align-items:center;justify-content:center;}} .nb svg{{fill:#c4c7c5;width:16px;height:16px;}} .reload svg{{fill:#5f6368;}} .omni{{flex:1;height:32px;background:#f1f3f4;border-radius:24px;display:flex;align-items:center;gap:6px;padding:0 12px;margin:0 4px;}} .addr{{font-size:14px;color:#202124;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}} .addr-host{{color:#202124;}} .addr-rest{{color:#5f6368;}} .tbb{{width:30px;height:30px;border-radius:50%;border:none;background:none;display:flex;align-items:center;justify-content:center;}} .tbb svg{{fill:#5f6368;width:18px;height:18px;}} .page img{{display:block;}} .status{{background:#f8f9fa;height:22px;display:flex;align-items:center;padding:0 10px;font-size:11px;color:#80868b;border-top:1px solid #e8eaed;flex-shrink:0;justify-content:space-between;}}"""
        return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{css}</style></head><body><div class="window">
  <div class="tabbar"><div class="tab">{fav_html}<span class="tab-title">{tab_title}</span><span class="tab-x">&times;</span></div>
    <button class="newtab"><svg viewBox="0 0 16 16" width="16" height="16"><path d="M8 1v14M1 8h14" stroke="#5f6368" stroke-width="1.5" fill="none" stroke-linecap="round"/></svg></button>
    <div class="wc"><button class="wb"><svg viewBox="0 0 10 1"><rect width="10" height="1" fill="#3c4043"/></svg></button><button class="wb"><svg viewBox="0 0 10 10"><rect x=".5" y=".5" width="9" height="9" fill="none" stroke="#3c4043" stroke-width="1"/></svg></button><button class="wb"><svg viewBox="0 0 10 10"><path d="M0 0L10 10M10 0L0 10" stroke="#3c4043" stroke-width="1.2"/></svg></button></div></div>
  <div class="toolbar">
    <button class="nb"><svg viewBox="0 0 24 24"><path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/></svg></button>
    <button class="nb"><svg viewBox="0 0 24 24"><path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8z"/></svg></button>
    <button class="nb reload"><svg viewBox="0 0 24 24"><path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg></button>
    <div class="omni"><span style="flex-shrink:0;display:flex;align-items:center">{lock_svg}</span>{secure_label}<div class="addr"><span class="addr-host">{hostname}</span><span class="addr-rest">{path_part}</span></div></div>
    <button class="tbb"><svg viewBox="0 0 24 24"><path d="M17 3H7c-1.1 0-2 .9-2 2v16l7-3 7 3V5c0-1.1-.9-2-2-2zm0 15l-5-2.18L7 18V5h10v13z"/></svg></button>
    <button class="tbb"><svg viewBox="0 0 24 24"><path d="M20.5 11H19V7c0-1.1-.9-2-2-2h-4V3.5a2.5 2.5 0 00-5 0V5H4c-1.1 0-2 .9-2 2v3.8h1.5c1.5 0 2.7 1.2 2.7 2.7s-1.2 2.7-2.7 2.7H2V20c0 1.1.9 2 2 2h3.8v-1.5c0-1.5 1.2-2.7 2.7-2.7s2.7 1.2 2.7 2.7V22H17c1.1 0 2-.9 2-2v-4h1.5a2.5 2.5 0 000-5z"/></svg></button>
    <button class="tbb"><svg viewBox="0 0 24 24"><circle cx="12" cy="5" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="12" cy="19" r="2"/></svg></button></div>
  <div class="page"><img src="data:image/png;base64,{page_png_b64}" width="{page_width}" height="{page_height}"></div>
  <div class="status"><span>{url}</span><span>Captured {ts_label}</span></div>
</div></body></html>"""
    steps = []
    async def capture_step(page, action_name, result_text):
        # Inline favicon fetcher — do NOT extract this to a separate function
        async def _get_fav():
            try:
                r = await page.evaluate("""async () => {
                    const el = document.querySelector("link[rel='icon'],link[rel='shortcut icon'],link[rel='apple-touch-icon']");
                    const url = el ? el.href : location.origin + '/favicon.ico';
                    try {
                        const res = await fetch(url);
                        if (!res.ok) return '';
                        const buf = await res.arrayBuffer();
                        const bytes = new Uint8Array(buf);
                        let b = ''; bytes.forEach(x => b += String.fromCharCode(x));
                        return 'data:image/png;base64,' + btoa(b);
                    } catch { return ''; }
                }""")
                return r or ""
            except Exception:
                return ""
        try:
            screenshot_bytes = await page.screenshot(timeout=5000)
            cur_url   = page.url
            cur_title = (await page.title() or _urlparse(cur_url).hostname or "Page")[:60]
            fav       = await _get_fav()
            page_b64  = _b64.b64encode(screenshot_bytes).decode()
            frame_html = _build_chrome_frame_html(page_b64, cur_url, cur_title, fav)
            tmp = await page.context.new_page()
            try:
                await tmp.goto("data:text/html;base64," + _b64.b64encode(frame_html.encode()).decode())
                await tmp.set_viewport_size({"width": 1325, "height": 2000})
                fh = await tmp.evaluate("() => { const el=document.querySelector('.window'); return el ? Math.ceil(el.getBoundingClientRect().height)+4 : document.body.scrollHeight; }")
                await tmp.set_viewport_size({"width": 1284, "height": int(fh)+4})
                framed = await tmp.locator(".window").screenshot(timeout=8000)
            finally:
                await tmp.close()
            steps.append({"name": action_name, "result": result_text, "image": framed})
            doc.add_heading(f"Step: {action_name}", level=2)
            doc.add_paragraph(f"Result: {result_text}")
            doc.add_picture(io.BytesIO(framed), width=Inches(6))
            doc.add_page_break()
'''

# Robust Standalone Helpers - injected into every prompt
_RESILIENT_HELPERS_CODE = '''\
    async def find_resilient(page, selector, text=None, timeout=10000):
        deadline = asyncio.get_event_loop().time() + (timeout / 1000.0)
        search_target = re.compile(re.escape(text.strip()), re.I) if text else None
        
        # Support for comma-separated multiple candidate selectors
        selectors = [s.strip() for s in selector.split(",")]
        
        while asyncio.get_event_loop().time() < deadline:
            for target in [page] + page.frames:
                for sel in selectors:
                    try:
                        locs = target.locator(sel)
                        if search_target: locs = locs.filter(has_text=search_target)
                        if await locs.count() > 0:
                            for i in range(await locs.count()):
                                l = locs.nth(i)
                                if await l.is_visible(): return l
                    except: continue
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Element '{selector}' (text='{text}') not found.")

    async def resilient_click(page, selector, text=None):
        el = await find_resilient(page, selector, text)
        await el.click()

    async def resilient_fill(page, selector, value):
        el = await find_resilient(page, selector)
        await el.fill(value)

    async def resilient_upload(page, target_name, local_path):
        """Standalone bulk upload with directory expansion."""
        paths = []
        if os.path.isdir(local_path):
            paths = [os.path.join(local_path, f) for f in os.listdir(local_path) if os.path.isfile(os.path.join(local_path, f))]
        elif os.path.isfile(local_path):
            paths = [local_path]
        if not paths: raise FileNotFoundError(f"No files found at {local_path}")
        async with page.expect_file_chooser() as fc_info:
            await resilient_click(page, "li[data-binf-action='add-document'], button:visible", text=target_name)
        await (await fc_info.value).set_files(paths)
        await page.wait_for_load_state("networkidle")
        return len(paths)
'''




_VAULT_SNIPPET = '''\
# ── Vault credential reader ───────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path

def vault_creds(prefix: str) -> dict:
    """Return {'username': ..., 'password': ...} from the encrypted vault."""
    # Walk up from this script to find the web-ui root
    base = _Path(__file__).resolve()
    for _ in range(8):
        base = base.parent
        candidate = base / "web-ui"
        if candidate.exists():
            _sys.path.insert(0, str(candidate))
            break
    try:
        from src.utils.vault import vault
        creds = vault.get_credentials(prefix) or vault.get_credentials(prefix.lower())
        if not creds:
            raise RuntimeError(f"No vault entry found for prefix '{prefix}'")
        lower = {k.lower(): v for k, v in creds.items()}
        def _pick(cs):
            for c in cs:
                if c in lower: return lower[c]
            return list(lower.values())[0] if lower else ""
        return {
            "username": _pick(("username","user","login","email")),
            "password": _pick(("password","passwd","pwd","pass")),
        }
    except Exception as e:
        raise RuntimeError(f"Vault error for '{prefix}': {e}")
# ─────────────────────────────────────────────────────────────────────────────
'''


# ─────────────────────────────────────────────────────────────────────────────
# Selector reference
# ─────────────────────────────────────────────────────────────────────────────

def _build_selector_map(history_data: dict) -> str:
    steps = history_data.get("history") or history_data.get("steps") or []
    if not steps:
        return ""

    PRIORITY = ["preferred", "id", "css", "recommended_selector", "xpath"]
    SKIP = {"go_to_url", "done", "scroll", "wait", "extract_content",
            "check_ssl_certificate", "retrieve_value_by_element"}

    def best_sel(el: dict) -> str:
        if not isinstance(el, dict):
            return ""
        # Handle both dom_context and interacted_element formats
        el_id = el.get("id") or el.get("attributes", {}).get("id")
        if el_id and str(el_id).strip():
            return f"#{el_id}"
            
        # Try selectors field (dom_context format)
        sels = el.get("selectors") or {}
        rec  = el.get("recommended_selector")
        if isinstance(rec, dict):
            rec = rec.get("selector", "")
            
        for key in PRIORITY:
            val = rec if key == "recommended_selector" else sels.get(key, "")
            if val and str(val).strip() not in ("", "table"):
                if "nth-child" not in str(val):
                    return str(val).strip()
        
        # Try attributes (interacted_element format)
        attrs = el.get("attributes") or {}
        if attrs.get("data-otname"):
            return f'[data-otname="{attrs["data-otname"]}"]'
        if attrs.get("title"):
             # For a.functionMenuHotspot, title is better than nothing
            return f'[title="{attrs["title"]}"]'
            
        return ""

    lines = ["## SELECTOR REFERENCE (real selectors from the live agent run)"]
    found_any = False

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        model_out   = step.get("model_output") or {}
        raw_actions = model_out.get("action") or []
        state       = step.get("state") or {}
        raw_dom     = state.get("interacted_element") or []
        url         = state.get("url") or step.get("url", "")
        
        actions = raw_actions
        dom_els = raw_dom
        if not isinstance(dom_els, list):
            dom_els = [dom_els]

        step_lines = []
        for ai, act in enumerate(actions):
            if not isinstance(act, dict):
                continue
            atype = next(iter(act.keys()), "")
            if atype in SKIP:
                continue
            
            # Match action to dom element
            el = dom_els[ai] if ai < len(dom_els) else (dom_els[-1] if dom_els else {})
            if not el and atype == "smart_login":
                el = act.get("dom_context")

            sel   = best_sel(el) if isinstance(el, dict) else ""
            
            # Extract text/aria/etc with dual format support
            if not isinstance(el, dict): el = {}
            attrs = el.get("attributes") or {}
            
            text = str(el.get("text") or attrs.get("text") or el.get("tag_name") or "")[:60].strip()
            aria = str(el.get("aria-label") or attrs.get("aria-label") or attrs.get("title") or "")[:40].strip()
            etype = str(el.get("type") or attrs.get("type") or "")
            
            parts = [f"  [{atype}]  selector: {sel or '(none)'}"]
            if text:   parts.append(f'label: "{text}"')
            if aria:   parts.append(f'description: "{aria}"')
            if etype:  parts.append(f'type: "{etype}"')
            
            # Context for smart_login or menu buttons
            comp = el.get("comprehensive", {})
            siblings = comp.get("siblings")
            row_ctx = comp.get("row_context")
            
            if (atype == "smart_login" or "menu" in str(sel).lower() or row_ctx) and "outerHTML" in el:
                html_context = el["outerHTML"].replace("\n", " ")[:1000]
                parts.append(f'context: "{html_context}..."')
                
                if row_ctx and row_ctx.get("allCellText"):
                    cells = ", ".join([f"'{t}'" for t in row_ctx["allCellText"]])
                    parts.append(f'row_context: [ {cells} ]')
                
                if siblings:
                    prev = siblings.get("prev")
                    next_sib = siblings.get("next")
                    if prev: parts.append(f"prev_sibling: <{prev['tag']}> \"{prev['text']}\"")
                    if next_sib: parts.append(f"next_sibling: <{next_sib['tag']}> \"{next_sib['text']}\"")

            step_lines.append("  " + " | ".join(parts))
            found_any = True

        if step_lines:
            hdr = f"\nSTEP {idx + 1}"
            if url:
                hdr += f" | {url}"
            lines.append(hdr)
            lines.extend(step_lines)

    if not found_any:
        return ""
    lines.append("")
    return "\n".join(lines)

def _compact_history(history_data: dict) -> str:
    steps = history_data.get("history") or history_data.get("steps") or []
    out = []
    seen_urls: set = set()

    for step in steps:
        if not isinstance(step, dict):
            continue
        state     = step.get("state") or {}
        model_out = step.get("model_output") or {}
        results   = step.get("result") or []
        url       = state.get("url") or step.get("url", "")
        raw_actions   = model_out.get("action") or []
        clean_actions = step.get("actions") or []
        step_actions  = raw_actions if raw_actions else clean_actions

        for ai, act in enumerate(step_actions):
            if not isinstance(act, dict):
                continue
            atype  = next(iter(act.keys()), "")
            params = act.get(atype) or {}
            res = results[ai] if ai < len(results) else {}
            if isinstance(res, dict) and res.get("error"):
                continue

            entry: dict = {"action": atype}
            if url and url not in seen_urls:
                entry["url"] = url
                seen_urls.add(url)
            if isinstance(params, dict):
                safe = {k: v for k, v in params.items()
                        if k not in ("dom_context", "element_info")
                        and not (isinstance(v, str) and len(v) > 500)}
                if safe:
                    entry["params"] = safe
            if isinstance(res, dict):
                content = res.get("extracted_content", "")
                if content and isinstance(content, str):
                    entry["result"] = content[:300]

            out.append(entry)

    return json.dumps(out, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# DOM Context Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _get_dom_context(history_path: str, step_index: int, target_index: int = None) -> str:
    """Find and summarize the DOM snapshot for a specific step."""
    try:
        hist_p = Path(history_path)
        run_id = hist_p.stem
        snapshot_dir = hist_p.parent.parent.parent / "dom_snapshots" / run_id
        snapshot_file = snapshot_dir / f"{step_index}.json"
        
        if not snapshot_file.exists():
            return ""

        with open(snapshot_file, "r", encoding="utf-8") as f:
            dom_data = json.load(f)
        
        context_parts = []
        elements = dom_data.get("elements", [])

        # 1. Target Element
        if target_index is not None:
            target = next((el for el in elements if el.get("identity", {}).get("order") == target_index), None)
            if target:
                context_parts.append(f"TARGET ELEMENT (Idx {target_index}):\n{_summarize_node(target)}")

        # 2. Extract Tables (High-Value Landmarks)
        table_count = 0
        for el in elements:
            if el.get("identity", {}).get("tagName") == "table":
                # For flat JSON, the "text" property in provenance contains the flattened table content
                summary = el.get("selector_provenance", {}).get("text", "")[:1200]
                if summary.strip():
                    t_id = el.get("attribute_fingerprint", {}).get("id", "none")
                    frame = el.get("identity", {}).get("frame_path", [])
                    context_parts.append(f"Table(id='{t_id}', frame={frame}):\n  {summary}")
                    table_count += 1
                if table_count >= 5: break

        # 3. Structure Summary (Fallback)
        if not context_parts:
            struct = "\n".join([f"<{e.get('identity',{}).get('tagName')}> {e.get('selector_provenance',{}).get('text','')[:50]}" 
                              for e in elements[:20]])
            context_parts.append(f"PAGE STRUCTURE:\n{struct}")

        return "\n\n".join(context_parts)
    except Exception as e:
        logger.warning(f"[_get_dom_context] Failed: {e}")
        return ""

def _summarize_node(node: dict) -> str:
    """Compact summary of an element node from flat structure."""
    meta = node.get("identity", {})
    attrs = node.get("attribute_fingerprint", {})
    prov = node.get("selector_provenance", {})
    
    parts = [
        f"Tag: <{meta.get('tagName')}>",
        f"ID: {attrs.get('id') or 'none'}",
        f"Frame: {meta.get('frame_path') or 'Main'}",
        f"Text Sample: {prov.get('text', '')[:150].strip()}",
        f"OuterHTML: {node.get('integrity', {}).get('outerHTML', '')[:400]}..."
    ]
    return "\n  ".join(parts)
    return parts

def _get_text_recursive(node: dict) -> str:
    """Legacy helper for tree structures."""
    return ""
    if not isinstance(node, dict): return icons
    if node.get("tagName") in ("img", "svg"):
        attrs = node.get("attributes", {})
        label = attrs.get("title") or attrs.get("alt") or attrs.get("aria-label") or ""
        if label: icons.append(label)
        elif node.get("tagName") == "svg":
            # Check class list for status-like names
            cls = " ".join(node.get("classList", []))
            if any(s in cls.lower() for s in ("success", "running", "error", "failed", "check", "warn")):
                icons.append(f"svg-icon({cls})")
    
    for child in node.get("children", []):
        icons.extend(_find_icons_recursive(child))
    return list(set(icons))


# ─────────────────────────────────────────────────────────────────────────────
# DOM Digest — scans ALL snapshots for a run, builds a compact DOM Intelligence
# ─────────────────────────────────────────────────────────────────────────────

def _build_dom_digest(history_path: str) -> str:
    """Scan ALL DOM snapshots for this run and return a compact DOM Intelligence block.
    
    Extracts:
    - Every table#id with its header row (column names)
    - Duplicate ID detection: flags when the SAME id appears >1 time in a single snapshot
    - key input/form element IDs
    - Elements with data-otname or significant aria-label attributes
    """
    try:
        hist_p = Path(history_path)
        run_id = hist_p.stem
        snapshot_dir = hist_p.parent.parent.parent / "dom_snapshots" / run_id
        if not snapshot_dir.exists():
            return ""

        # tables[el_id] -> list of occurrences (one per unique table content seen)
        # Each occurrence: {frame, selector, headers, sample_rows, snap_file, count_in_snap}
        tables: dict = {}
        inputs: dict = {}
        buttons: list = []

        for snap_file in sorted(snapshot_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
            try:
                with open(snap_file, "r", encoding="utf-8") as f:
                    dom_data = json.load(f)
                elements = dom_data.get("elements", [])
                if not isinstance(elements, list):
                    continue

                # Count how many times each table ID appears IN THIS SINGLE SNAPSHOT
                id_counts_in_snap: dict = {}
                for el in elements:
                    if not isinstance(el, dict): continue
                    meta  = el.get("identity", {})
                    attrs = el.get("attribute_fingerprint", {})
                    if meta.get("tagName", "").lower() == "table" and attrs.get("id"):
                        tid = attrs["id"]
                        id_counts_in_snap[tid] = id_counts_in_snap.get(tid, 0) + 1

                for el in elements:
                    if not isinstance(el, dict):
                        continue
                    meta  = el.get("identity", {})
                    attrs = el.get("attribute_fingerprint", {})
                    prov  = el.get("selector_provenance", {})
                    tag   = meta.get("tagName", "").lower()
                    el_id = attrs.get("id", "")
                    frame = meta.get("frame_path", "main")

                    # --- Tables ---
                    if tag == "table" and el_id:
                        text = prov.get("text", "")[:2000]
                        lines = [l.strip() for l in text.replace("\t", "|").split("\n") if l.strip()]
                        header_line = lines[0] if lines else ""
                        sample_rows = lines[1:4]
                        count_in_snap = id_counts_in_snap.get(el_id, 1)

                        if el_id not in tables:
                            tables[el_id] = {
                                "frame": frame,
                                "selector": f"table#{el_id}",
                                "headers": header_line,
                                "sample_rows": sample_rows,
                                "max_count_in_snap": count_in_snap,
                                "all_headers": [header_line] if header_line else [],
                            }
                        else:
                            # Update max duplicate count
                            tables[el_id]["max_count_in_snap"] = max(tables[el_id]["max_count_in_snap"], count_in_snap)
                            # Collect all distinct header rows to show both tables
                            if header_line and header_line not in tables[el_id]["all_headers"]:
                                tables[el_id]["all_headers"].append(header_line)

                    # --- Inputs with IDs ---
                    if tag == "input" and el_id:
                        itype = attrs.get("type", "text")
                        name  = attrs.get("name", "")
                        label = attrs.get("aria-label", "") or attrs.get("placeholder", "") or name
                        sel   = f"#{el_id}"
                        if sel not in inputs:
                            inputs[sel] = f"input[type={itype}] label=\"{label}\""

                    # --- Buttons/links with data-otname ---
                    if attrs.get("data-otname") and tag in ("a", "button", "span"):
                        desc = f'[data-otname="{attrs["data-otname"]}"] text="{prov.get("text","")[:50]}"'
                        if desc not in buttons:
                            buttons.append(desc)

            except Exception as e:
                logger.debug(f"[_build_dom_digest] skip {snap_file.name}: {e}")
                continue

        if not tables and not inputs:
            return ""

        out = ["## DOM INTELLIGENCE (extracted from all page snapshots)"]
        out.append("USE THESE EXACT SELECTORS — they are confirmed from the live DOM.")
        out.append("")

        if tables:
            out.append("### Tables (verified IDs)")
            out.append("| Selector | Count on Page | Columns (each distinct table) | Action Required |")
            out.append("|---|---|---|---|")
            for tid, info in tables.items():
                count = info["max_count_in_snap"]
                all_hdrs = info["all_headers"] or ["(no header)"]
                hdr_display = " | ALSO: ".join(h[:80] for h in all_hdrs)[:200]
                if count > 1:
                    action = f"⚠️ DUPLICATE ID ({count}x) — MUST use `.first` or `.nth(N)` to select specific table"
                else:
                    action = "✅ unique — use directly"
                out.append(f"| {info['selector']} | {count} | {hdr_display} | {action} |")
            out.append("")
            out.append("**CRITICAL RULE:** If 'Count on Page' > 1, you MUST use `.first` to avoid Playwright strict mode violation:")
            out.append("  ✅ CORRECT: `page.locator('table#statusTable').first`")
            out.append("  ❌ WRONG:   `page.locator('table#statusTable')` (will crash with strict mode error)")
            out.append("Determine if horizontal (column-index extraction) or vertical (row-filter sibling) — see TABLE EXTRACTION RULES below.")


        if inputs:
            out.append("")
            out.append("### Form Inputs (verified IDs)")
            for sel, desc in list(inputs.items())[:20]:
                out.append(f"  {sel}  → {desc}")

        if buttons:
            out.append("")
            out.append("### Action Buttons (data-otname)")
            for b in buttons[:10]:
                out.append(f"  {b}")

        return "\n".join(out)
    except Exception as e:
        logger.warning(f"[_build_dom_digest] Failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = r"""\
You are an expert Playwright automation engineer.

Write a complete, self-contained Python script based on the agent execution history.
Use raw Playwright APIs. Define any helper functions you need yourself.
The only allowed external imports are:
  playwright, docx, standard library modules (os, sys, io, logging, re, asyncio, etc.).
Mandatory: Use `import os, sys, logging, asyncio, io, re, base64` at the top of every script.

CREDENTIALS
-----------
A vault_creds() function is already defined in the VAULT SNIPPET at the top of every
script. Call it to get credentials:
  creds = vault_creds("PREFIX")        # returns {"username": "...", "password": "..."}
  username = creds["username"]
  password = creds["password"]
Never hardcode credentials. Never use os.getenv() for credentials.
DYNAMIC PARAMETERIZATION (NO HARDCODING)
---------------------------------------
To ensure the script is dynamic and not tied to hardcoded strings:
1. Define a `SCRIPT_CONFIG` dictionary at the top of `run()`.
2. Move ALL site-specific strings (URLs, target headers, column names, menu text) into this config.
3. Access them throughout the script via `SCRIPT_CONFIG["key"]`.
4. This allows the user to easily repurpose the script for other websites.

Example:
```python
async def run():
    SCRIPT_CONFIG = {
        "base_url": "http://...",
        "dashboard_title": "Distributed Agent Status",
        "search_label": "User name",
    }
    await page.goto(SCRIPT_CONFIG["base_url"])
    # AVOID ^ and $ anchors in locators—they are too brittle for dynamic sites
    await page.locator("h2").filter(has_text=re.compile(SCRIPT_CONFIG["dashboard_title"], re.I)).first.wait_for()
```

ROBUST LOCATORS (MANDATORY)
--------------------------
1. **BAN** `^` (start) or `$` (end) anchors in regex for labels/text.
   - These anchors are fragile and fail if there is any invisible whitespace, non-breaking spaces, or nested `<span>` elements (which are common in OTCS).
   - **CORRECT**: `re.compile(re.escape(text), re.I)` or `re.compile(f"substring", re.I)`
   - **WRONG**: `re.compile(f"^{text}$")`
2. Always use `.first` to get the actual data cell and skip "outer" layout cells.

LOGIN (DYNAMIC & RESILIENT)
---------------------------
After page.goto() to the login URL:
1. Wait for ANY password field: await page.wait_for_selector('input[type="password"]', timeout=15000)
2. Fill username. Use the `context` provided in the SELECTOR REFERENCE to find IDs/names.
   - Strategy: `page.locator("input[name*='user' i], input[id*='user' i], input[name*='login' i]").first`
   - Common patterns: `otds_username`, `#username`, `input[name='login']`.
3. Fill password similarly: `page.locator("input[type='password']").first`
4. Click the standard submit button — NEVER click SSO/federation/test buttons.
   - Strategy: `page.locator("button[type='submit'], input[type='submit'], #loginbutton, #login_button, .submit-btn").filter(visible=True).first`
   - Also try: `page.get_by_role("button", name=re.compile(r"Sign in|Login", re.I))`
5. await page.wait_for_load_state("networkidle")

AFTER LOGIN — DO NOT use wait_for_url() to verify login.
Complex apps redirect WITHIN the same URL path (e.g. /app/ → /app/#/home).
A lambda like `wait_for_url(lambda url: "app-path" not in url)` will ALWAYS time out,
throw an exception, and skip every step after it.
Just use: await page.wait_for_load_state("networkidle")
If you need to confirm login succeeded, check that the password field is gone:
  await page.wait_for_selector('input[type="password"]', state="hidden", timeout=15000)

EXCEPTION HANDLING & GUARANTEED REPORTING (CRITICAL)
-----------------------------------------------
1. Wrap the entire automation logic in a `try/except` block.
2. If an exception occurs, log it and print the traceback, but DO NOT stop the script before reporting.
3. The report generation (Word document) MUST be in a `finally:` block or after the main `try/except`
   to ensure that whatever steps were captured BEFORE the failure are still included in the report.
4. Correct structure:
   ```python
   async def run():
       steps = []
       p = None; browser = None # Define outside try for finally block
       try:
           p = await async_playwright().start()
           browser = await p.chromium.launch(...)
           # ... automation steps with capture_step() ...
       except Exception as e:
           logging.error(f"Execution failed: {e}")
           import traceback; traceback.print_exc()
       finally:
           # CLOSING BROWSER AND GENERATING REPORT MUST BE IN FINALLY
           if browser: await browser.close()
           if p: await p.stop()
           await asyncio.sleep(0.5)
           
           # --- GENERATE REPORT HERE ---
           doc = Document()
           # ... (use 'steps' list) ...
           doc.save(report_path)
   ```
A script that crashes without producing a report is a FAILURE. Always produce a report.

SELECTORS
---------
Use ONLY selectors from the SELECTOR REFERENCE in the prompt.
Never invent selectors.

CLICKING NAVIGATION ITEMS (menus, links, buttons with text)
------------------------------------------------------------
Responsive pages render nav items TWICE — once for desktop (visible) and once for
mobile (hidden). page.get_by_text("x").first picks the HIDDEN one and times out.

ALWAYS use a visible-only locator for navigation clicks:
  await page.locator("a:visible, button:visible, [role='menuitem']:visible, li:visible > a:visible")\
            .filter(has_text=re.compile(r"Dashboard", re.I)).first.click()

Or with a short explicit wait pattern:
  dashboard = page.get_by_text("Dashboard", exact=True).filter(visible=True).first
  # if .filter(visible=True) is not available in your Playwright version, use:
  dashboard = page.locator("a:visible:has-text('Dashboard'), li:visible a:has-text('Dashboard'), .nav-link:visible:has-text('Dashboard')").first
  await dashboard.wait_for(state="visible", timeout=10000)
  await dashboard.click()

NEVER use .first alone on get_by_text() for nav items — it is not visibility-filtered.

WAITS & NAVIGATION
------------------
After every page.goto():   await page.wait_for_load_state("networkidle")
SPA hash-routes (#/path):  await page.wait_for_url("**/path**", timeout=15000)
                           await page.wait_for_selector("REAL_SELECTOR", timeout=15000)
                           await page.wait_for_timeout(2000)

RELATIVE URLS:
If you extract a URL from an attribute (e.g., `href = await el.get_attribute("href")`), it 
may be relative. NEVER pass raw relative URLs to `page.goto()`.
  1. Prefer clicking the element: `await el.click()`
  2. Or normalize it: `from urllib.parse import urljoin; await page.goto(urljoin(page.url, href))`

OTCS SPECIFIC LOGIC (ROBUSTNESS):
--------------------------------
1. **Saving Folders/Items**: After typing a name in an inline form, ALWAYS use `await page.keyboard.press("Enter")` to save.
2. **Finding New Items (SORTING)**: After creation, ALWAYS click the "Modified" column header to sort.
   - **CRITICAL**: OTCS may sort Ascending first. You MUST verify or click twice to ensure **Descending** (newest first).
3. **Navigation (STABILITY)**: Entering folders via `.click()` can be unstable in OTCS Smart View (SPA).
   - **CORRECT**: Extract the `href` attribute and use `await page.goto(folder_url)`. This is much more stable and avoids `TargetClosedError`.
   - **Example**:
     ```python
     link = page.locator("a").filter(has_text=re.compile(re.escape(name), re.I)).first
     href = await link.get_attribute("href")
     if href: await page.goto(href)
     else: await link.click() # Fallback
     ```
4. **Bulk Upload (STABILITY)**: Clicking "Document" may NOT trigger a standard Playwright `file_chooser` event in some OTCS versions.
   - **CORRECT**: Use the "Injected Input" strategy. Click "Document", then wait for the hidden input to appear in the DOM.
   - **Example**:
     ```python
     await page.locator("a:has-text('Document')").click()
     file_input = await page.wait_for_selector('div.csui-file-open input[type="file"], input[type="file"][style*="display: none"]', state="attached")
     await file_input.set_input_files(file_paths)
     ```

DATA EXTRACTION
---------------
ALWAYS use r\"\"\"...\"\"\" (raw Python string) for ANY JavaScript passed to page.evaluate()
that contains regex like /\\d/ or /\\s/. Plain strings cause SyntaxWarning errors.
Correct:  value = await page.evaluate(r\"\"\"() => { ... /^[\\d.]+\\s*GB$/i ... }\"\"\")
Wrong:    value = await page.evaluate(\"\"\"() => { ... /^[\\d.]+\\s*GB$/i ... }\"\"\")

To find the CENTER value of a pie/donut chart:
  - Query all SVG text/tspan elements
  - Filter for those matching a size pattern like /^[\d.]+\s*(GB|TB|MB|%)$/i
  - Sort descending by numeric value — the center total is always the largest number
  - Return the primary result text, or "Not Found" if no matches exist.
  - Return "Not Found" as is. NEVER append units like " GB" manually to a "Not Found" result.
  - Correct: `print(f"Value: {val}")`. Wrong: `print(f"Value: {val} GB")`.

OTCS SPECIFIC LOGIC (ROBUSTNESS):
--------------------------------
1. **Saving Folders/Items**: After typing a name in an inline form, ALWAYS use `await page.keyboard.press("Enter")` to save.
2. **Finding New Items (SORTING)**: After creation, ALWAYS click the "Modified" column header to sort.
   - **CRITICAL**: OTCS may sort Ascending first. You MUST verify or click twice to ensure **Descending** (newest first).
3. **Navigation (STABILITY)**: Entering folders via `.click()` can be unstable in OTCS Smart View (SPA).
   - **CORRECT**: Extract the `href` attribute and use `await page.goto(folder_url)`. This is much more stable and avoids `TargetClosedError`.
   - **Example**:
     ```python
     link = page.locator("a").filter(has_text=re.compile(re.escape(name), re.I)).first
     href = await link.get_attribute("href")
     if href: await page.goto(href)
     else: await link.click() # Fallback
     ```

BROWSER LAUNCH — ALWAYS use these exact settings, no exceptions:
--------------
  browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
  context = await browser.new_context(no_viewport=True)
  page = await context.new_page()
headless=True causes pages to render at a tiny viewport — charts and nav items
may not appear. no_viewport=True is required for --start-maximized to take effect.

  await page.wait_for_selector(".app-main-content, #global-search", timeout=15000)

RESILIENT ACTIONS (MANDATORY)
-----------------------------
OTCS and dynamic apps often hide content in iframes.
MANDATORY: Copy the provided RESILIENT HELPERS verbatim at the start of `run()`.
Always use `await resilient_click(page, ...)` and `await resilient_fill(page, ...)`—never chain discovery and action yourself.

BULK UPLOAD & DIRECTORY SUPPORT:
--------------------------------
If the task involves a "folder" or "bulk" upload, use `await resilient_upload(page, "Document", local_path)`.
This function automatically expands a directory into a list of files.

OTCS SPECIFIC LOGIC (ROBUSTNESS):
--------------------------------
1. **Saving Folders/Items**: After typing a name in an inline form, ALWAYS use `await page.keyboard.press("Enter")` to save. It is more reliable than clicking a checkmark button.
2. **Finding New Items**: After creation, ALWAYS click the "Modified" column header to sort. The new item will be in the first row. Use `page.locator("a").filter(has_text=re.compile(f"^{re.escape(name)}$")).first` to click it.
CRITICAL TABLE EXTRACTION RULES (READ CAREFULLY):
-------------------------------------------------
**IMPORTANT: Before extracting table data, ALWAYS ask: "Is this a horizontal or vertical table?"**

1. **HORIZONTAL TABLE** (Most common in OTCS/enterprise apps):
   - Structure: Table has HEADER ROW + DATA ROWS. Each DATA ROW is one record.
   - Example DOM text: `Partition\tRAM (%)\tDisk (%)\tState\n...\nPartition 1\t5\t0\tNormal\t...`
   - **CORRECT STRATEGY**: Find the data row by its name, then get the Nth `<td>` by column index.
   - Use `resilient_table_extract(page, "TableID", "Row Name", col_index=N)`
   - `col_index` is determined by counting columns from the header row (0-indexed).
   - **NEVER** look for a label element "Partition State" — it DOES NOT EXIST in horizontal tables.
   - **NEVER** use `./following-sibling::td` XPath in horizontal tables.

2. **VERTICAL TABLE** (Key-Value layout):
   - Structure: Each ROW has a `<td>` label (e.g. "Status:") and a `<td>` value.
   - Example DOM: `<tr><td>Status</td><td>Running</td></tr>`
   - **CORRECT STRATEGY**: Find the label cell, then get the sibling `<td>`.
   - Use: `page.locator("tr").filter(has=page.locator("td").filter(has_text="Status")).locator("td").nth(1)`

3. **CRITICAL: DO NOT USE `find_resilient` to look for a "Partition State" label.**
   - In OTCS `PartitionMapTable`, the DOM text shows: `Partition | RAM (%) | Disk (%) | State`
   - `State` is a **column header**, not a row label. The VALUE is e.g. `Normal`.
   - To get it: `await table.locator("tr").filter(has_text="Partition 1").locator("td").nth(5).text_content()`
   - (Column 5 = State column, confirmed from DOM snapshot analysis)

CRITICAL: Initialise ALL data variables (lists, dicts, strings) at the VERY START of `run()` to avoid `UnboundLocalError` in the `finally` block.
Always use `await resilient_click(page, ...)` and `await resilient_fill(page, ...)`—never chain discovery and action yourself.

REPORTING & SCREENSHOTS (MANDATORY — COPY VERBATIM)
----------------------------------------------------
The following helpers MUST be copied VERBATIM at the START of `run()`, exactly as-is.
DO NOT modify, simplify, or rewrite them. They contain the approved Chrome frame implementation.

```python
## ── PASTE THE CAPTURE_STEP_CODE BLOCK HERE ────────────────────────────────
{{CAPTURE_STEP_CODE}}
## ─────────────────────────────────────────────────────────────────────────
```

After every `page.goto()` or click, call `await capture_step(page, "Step name", "Result text")`.
Call it immediately after every major action:
  `await capture_step(page, "Navigate to Dashboard", "Navigated to <url>")`
  `await capture_step(page, "Open Action Menu for AdminServer-01", "Menu opened")`

FINAL REPORT STRUCTURE:
1. Create a `Document()`.
2. **Execution Summary Table**: Two columns (Step, Action, Result).
3. **Data Tables**: If data was extracted before failure, include it.
4. **Step Screenshots**: Heading for each step followed by the framed screenshot.
5. Save the report using an absolute path co-located with the script.
   - Use: `report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCRIPT_CONFIG["report_filename"])`
6. MANDATORY: The report generation MUST execute even if the automation fails midway.

DATA CLEANING & SANITIZATION:
Mandatory: Use a `clean_id` helper at the start of `run()`:
```python
    def clean_id(raw):
        # Normalize whitespace, take first line, truncate to 100 chars
        return re.sub(r'\s+', ' ', (raw or "").split('\n')[0].strip())[:100]
```

XPATH & SIBLING LOOKUPS (CRITICAL)
---------------------------------
1. **XPath Unions**: Use `|` for unions, NEVER use a comma `,`.
   - **WRONG**: `locator("xpath=//div, //span")`
   - **CORRECT**: `locator("xpath=//div | //span")`
2. **Finding Values next to Labels** (The "Golden Pattern"):
   - Apps like OTCS often put labels in one `<td>` and values in the next.
   - **STRATEGY A (Row Filter - Best)**: Find the row that contains the label, then get the second cell.
     ```python
     row = page.locator("tr").filter(has=page.locator("td, th, span").filter(has_text=re.compile(label, re.I))).first
     value = await row.locator("td").nth(1).text_content()
     ```
   - **STRATEGY B (XPath Sibling)**: 
     `locator("xpath=./following-sibling::td[1] | ./parent::*/following-sibling::td[1]")`

CLEAN EXIT (WINDOWS)
--------------------
To prevent noisy "I/O operation on closed pipe" warnings on Windows, add this BEFORE `asyncio.run()` or at the top of the file:
```python
import sys
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
```
And suppress the logger in `run()`:
  `logging.getLogger("asyncio").setLevel(logging.ERROR)`
Ensure you use `await browser.close()` inside the `try/finally` block.
Mandatory: Add `await asyncio.sleep(0.5)` after `await browser.close()` to avoid "RuntimeError: Event loop is closed" on Windows.
CRITICAL: If the script fails on a `wait_for` step for a header, it means the tag or text is incorrect. Broaden search to include `div, td`.

STRICT LOGIC ADHERENCE
----------------------
1. Follow the AGENT HISTORY exactly once.
2. Follow the logic specified in the ## TASK objective for reconfiguration or triggers.
   - If the user says "config only if Running", do NOT use "if not Running".
   - Follow the user's direction over common-sense defaults.
3. Do NOT generate loops, retries, or "if/else" logic unless explicitly present in history or required by the objective.
4. STOP only after the ABSOLUTE FINAL action in the provided AGENT HISTORY.
5. NEVER append explanation text or alternative versions.

TABLE EXTRACTION (DYNAMIC & ROBUST)
----------------------------------
Nested tables and non-standard header names (e.g. "Worker ID" vs "Agent Name") require a high-resiliency approach.
1. **Find the Header Row (ROBUST DUAL-VERIFICATION)**:
   - DO NOT look for headers by page title. You MUST find a `tr` that contains cells for BOTH an identifier and a status within the SAME row.
   - **Inclusive Synonyms**: Use a broad regex for identifiers in `SCRIPT_CONFIG`. Example: `(Worker ID|Agent Name|ID|Agent|Worker|Name|User)`. For status: `(Status|State|Result)`.
   - **Unique Index Enforcement**: Ensure the identifier and status columns are in different cells (`idx_1 != idx_2`). This prevents matching page titles where both keywords appear in one cell.
   - **Diagnostic Fallback (MANDATORY)**: If the header row discovery loop finishes without a match, the script MUST iterate through all `tr` elements one more time and print the `clean_id` text of every cell. This is critical for debugging why a header wasn't found.
   - **Direct-Child Cell Matching**: ALWAYS use `> th, > td` to avoid nested layout table pollution.
   - Example Pattern (Robust):
     ```python
     all_trs = await page.locator("tr").all()
     header_found = False
     for row in all_trs:
         cells = await row.locator("> th, > td").all()
         if len(cells) < 2: continue
         texts = [clean_id(await c.text_content()) for c in cells]
         # ... regex check across texts ...
         if match:
             header_found = True; break
     if not header_found:
         for r in all_trs: print(f"DEBUG Table Row: {[clean_id(await c.text_content()) for c in await r.locator('> th, > td').all()]}")
         raise RuntimeError("Could not find table headers")
     ```
2. **Identify the Data Table**:
   - Once the header row is found, the data table is ALWAYS `header_row.locator("xpath=./ancestor::table[1]")`.
3. **BAN Title-Based Locators**: NEVER locate a table by searching for its title text (e.g. `h1-h6`, `span`, `div`). ALWAYS use the **Header-First** pattern above.
4. **Header Row Exclusion**: DO NOT use regex substring search (`re.search`) to skip the header row, as it might accidentally match data rows (like "Agent" matching "DAAgent"). Instead, use EXACT STRING EQUALITY to compare the extracted `agent_name` to the known header text.
   - Example: First save the header text (`header_text = clean_id(await cells[id_col].text_content())`), then in the data loop: `if agent_name.lower() == header_text.lower(): continue`
5. **Scoped Action Menu (PRECISION)**:
   - To click an action menu for a specific row, first identify the specific link or cell containing the unique text (e.g. the server name).
   - **Ancestor Row Scoping (MANDATORY)**: Use `locator("xpath=./ancestor::tr[1]")` to find the *innermost* row that contains that specific cell. This prevents matching broad layout containers.
   - **Caret Identification**: Inside that innermost row, locate the caret/menu button using a class or ID from the SELECTOR REFERENCE.
   - Example:
     ```python
     item_link = page.get_by_text(name).filter(visible=True).first
     target_row = item_link.locator("xpath=./ancestor::tr[1]")
     menu_btn = target_row.locator(".caret_or_menu_selector_from_history").filter(visible=True).first
     ```
   - **Vigorous Menu Interaction (VMI)**: For function menus/dropdowns, standard `.click()` is often insufficient. MANDATORY to use:
     `await menu_el.hover()`
     `await menu_el.dispatch_event('mousedown')`
     `await menu_el.dispatch_event('mouseup')`
     `await menu_el.click()`
     `await page.wait_for_timeout(1000) # MANDATORY: Wait for menu to animate`
6. **Shallow Columns (CRITICAL)**:
   - When extracting cells from a row, ALWAYS use `locator("> th, > td")`.
7. **Async Iteration (CRITICAL)**:
   - When using `.all()`, await it once: `rows = await locator.all()`.
   - Then iterate normally: `for row in rows:`.
8. **Resilient Data Row Extraction**:
   - Filter out spacing rows: `if len(cells) > max_index and "".join([await c.text_content() for c in cells]).strip():`.
9. **Timestamped Reports (MANDATORY)**:
   - To avoid file-lock errors, generate a unique filename with a timestamp AT RUNTIME.
   - In `run()`, use: `filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx")`.
10. **ZERO HARDCODING**:
   - Never use specific strings from prompt examples in your code. Derive ALL names, statuses, and labels from the provided context.
11. **Icon-Based Status Extraction**:
    - If a status is represented by an icon (e.g., a green checkmark), the text will NOT be in `text_content()`. It will be in the `title` or `alt` attribute of an `img` tag.
    - When extracting statuses, implement a fallback: if text extraction yields empty/Not Found, search for an `img` tag within the element or its parent row, and use `await img.get_attribute("title")` or `alt`. Example:
      ```python
      img_locator = target_cell.locator("img").first
      if await img_locator.count() > 0:
          status_value = await img_locator.get_attribute("title") or await img_locator.get_attribute("alt")
      ```

12. **Precision Row-Specific Selection (ROBUST)**:
    - When interacting with an element inside a table (like a menu button for a specific row), do NOT rely on a global index or broad selector.
    - **Use `row_context`**: Identify a unique string from the `row_context` (e.g., the server or agent name) and use it to locate the row.
    - **Identify the Target**: Once the row is located via `.filter(has_text=...)`, find the target element (button, link, caret) within that row.
    - Example:
      ```python
      # The selector reference shows row_context: ['MyServer', 'Running', ...] for a click
      target_name = "MyServer" 
      target_row = page.locator("tr").filter(has_text=re.compile(rf"\b{target_name}\b", re.I)).first
      menu_btn = target_row.locator(".menu-selector-from-history").first
      await menu_btn.click()
      ```

13. **SYNTAX: DO NOT AWAIT LOCATORS (CRITICAL)**:
    - In Playwright for Python, `locator()`, `.first`, `.last`, `.nth()`, and `.filter()` are for defining WHERE to look and are **NOT** awaitable.
    - **WRONG**: `await page.locator("...").first`
    - **CORRECT**: `page.locator("...").first` (standard assignment)
    - **CORRECT**: `await page.locator("...").first.click()` (awaiting the action)
    - **CORRECT**: `await page.locator("...").first.wait_for()` (awaiting the wait)
    - **CORRECT**: `await page.locator("...").count()` (awaiting the count)
    - **CORRECT**: `await page.locator("...").all()` (awaiting the list retrieval)
    - **CORRECT**: `await page.locator("...").text_content()` (awaiting the extraction)
    - If you are assigning a locator to a variable for later use, DO NOT use `await`.
529: 
530: SELECTORS (ROBUSTNESS)
----------------------
CRITICAL: TOTAL BAN on `page.wait_for_selector()`. It is prone to hallucinations. Use `locator().wait_for()` instead (especially for login).
CRITICAL: TOTAL BAN on all `:has-text()` pseudo-selectors. They cause syntax errors. Use `.filter(has_text=...)` instead.
CRITICAL: NEVER assign the result of `locator.wait_for()` to a variable. It returns `None`.

To wait for an element with specific text:
- Target: `page.locator("table").filter(has=page.locator("th, td").filter(has_text=re.compile(r"ColName", re.I)))).filter(visible=True).first`
- Use `.first` or `.last` to avoid outer "Shell" tables.
- CRITICAL: Use `re.compile(r"...", re.I)` for ALL `has_text` filters (column names, table titles, nav items) to avoid whitespace and case issues.
BROWSER LAUNCH & WINDOWS STABILITY (CRITICAL):
----------------------------------------------
To avoid "RuntimeError: Event loop is closed" on Windows, use this exact manual startup/cleanup pattern (do NOT use `async with`):
```python
    p = None
    browser = None
    # INITIALIZE ALL DATA VARIABLES TO EMPTY LISTS/DICTS/STRINGS HERE
    distributed_agents_data = [] 
    partition_map_summary = {} 
    # ... etc
    
    try:
        p = await async_playwright().start()
        browser = await p.chromium.launch(...)
        ...
    finally:
        # CLEANUP MUST BE IN A PROTECTED TRY-EXCEPT
        try:
            if browser: await browser.close()
            if p: await p.stop()
        except: pass # Ignore playwright connection closed errors
```

Headers and Titles:
- Use broad locators for robustness: `page.locator("h1, h2, h3, h4, h5, h6, b, span, div, td").filter(has_text=text).first`

Rows and Inputs:
- Correct: `page.locator("tr").filter(has_text=row_id).locator("input").first`
- Wrong:   `page.locator("tr:has-text('...') input")`

CLICKING NAVIGATION & MENUS (DYNAMIC)
-------------------------------------
Always filter for visibility to avoid clicking hidden/mobile clones:
  await page.get_by_text(nav_text, exact=True).filter(visible=True).first.click()

Action Menus & Buttons:
- Prioritize selectors provided in the `## SELECTOR REFERENCE` section for specific actions like `click_element` or `open_action_menu`.
- Common Action Menu Patterns (Classic & Smart View):
  - `[title*='Functions']`, `[title*='Actions']`, `.functionMenuHotspot`, `[data-otname='objFuncMenu']`
  - `button[aria-label*='actions']`, `.icon-dropdown`, `.csui-table-row-action-menu`
  - `a[role='button'][data-csui-extension='dropdown']`
- ALWAYS use visibility filtering: `.filter(visible=True).first.click()`

OUTPUT
------
Return ONLY valid Python code. No markdown fences. No explanation.
Exactly one `async def run():` block.
The script MUST end with this exact entry point:
```python
if __name__ == "__main__":
    asyncio.run(run())
```
"""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_script(
    history_path: str,
    output_path: str = None,
    model_name: str = None,
    provider: str = None,
    objective: str = None,
    mandatory_history_path: str = None,
    vault_prefix: str = None,
) -> tuple:
    """Generate a pure Playwright script. Returns (output_path, code)."""
    model_name = model_name or SCRIPT_GEN_MODEL
    provider   = provider   or SCRIPT_GEN_PROVIDER

    with open(history_path, "r", encoding="utf-8") as f:
        history_data = json.load(f)
    run_id = Path(history_path).stem
    logger.info(f"[generate_script] Loaded: {run_id}")

    if mandatory_history_path and os.path.exists(mandatory_history_path):
        try:
            with open(mandatory_history_path, "r", encoding="utf-8") as f:
                mandatory_data = json.load(f)
            history_data["history"] = (
                mandatory_data.get("history", []) + history_data.get("history", [])
            )
        except Exception as e:
            logger.warning(f"[generate_script] Could not merge mandatory history: {e}")

    if not vault_prefix:
        search_text = objective or history_data.get("task", "") or ""
        matches = re.findall(r"@vault\.([a-zA-Z0-9_]+)", search_text)
        if matches:
            vault_prefix = matches[0].upper()
            logger.info(f"[generate_script] Auto-detected vault prefix: '{vault_prefix}'")

    effective_prefix = (vault_prefix or VAULT_CREDENTIAL_PREFIX).upper()
    logger.info(f"[generate_script] Vault prefix: '{effective_prefix}'")

    selector_map = _build_selector_map(history_data)
    compact_hist = _compact_history(history_data)
    task_text    = objective or history_data.get("task", "Complete the automation task")

    # Add DOM Context for relevant steps (e.g. table extraction steps)
    dom_context_blocks = []
    steps = history_data.get("history") or history_data.get("steps") or []
    for idx, step in enumerate(steps):
        model_out = step.get("model_output") or {}
        actions = model_out.get("action") or []
        
        # Extract target index if this is a targeted element action
        target_idx = None
        for act in actions:
            if not isinstance(act, dict): continue
            for key in ("retrieve_value_by_element", "click_element", "hover_capture"):
                if key in act:
                    target_idx = act[key].get("index") or act[key].get("element_index")
                    break
            if target_idx: break

        # Always provide context for the last 3 steps, or any step with a target index
        if idx >= len(steps) - 3 or target_idx:
            context = _get_dom_context(history_path, idx + 1, target_idx)
            if context:
                dom_context_blocks.append(f"### STEP {idx + 1} PAGE STRUCTURE\n{context}")

    dom_ctx_str = "\n\n".join(dom_context_blocks) if dom_context_blocks else ""

    dom_digest_str = _build_dom_digest(history_path)

    user_prompt = f"""## TASK
{task_text}

## VAULT PREFIX
Call `vault_creds("{effective_prefix}")` to get credentials.
The vault_creds() function is already defined — copy the VAULT SNIPPET below verbatim
at the top of the script before your imports, then call it in run().

## VAULT SNIPPET (copy this verbatim at the very top of the script)
```
{_VAULT_SNIPPET}
```

```python
{_CAPTURE_STEP_CODE}
```

## MANDATORY CODE BLOCK — RESILIENT HELPERS (copy verbatim inside run(), before automation steps)
The code below provides robust navigation and bulk upload. You MUST use these helpers for ALL interactions.
```python
{_RESILIENT_HELPERS_CODE}
```

{selector_map}

{dom_digest_str}

## PAGE STRUCTURE REFERENCE (Full structural context for relevant steps)
{dom_ctx_str or "No additional DOM snapshots found."}

## AGENT HISTORY (successful path — failed/retry actions removed)
{compact_hist}

Write the complete Python script now. Start with the VAULT SNIPPET verbatim."""

    current_system_prompt = _SYSTEM_PROMPT.replace("{{CAPTURE_STEP_CODE}}", _CAPTURE_STEP_CODE)

    prompt_tokens = (len(current_system_prompt) + len(user_prompt)) // 4
    # Increase headroom to 16k to ensure long scripts aren't truncated
    num_ctx = max(16384, min(prompt_tokens + 16384, SCRIPT_GEN_NUM_CTX))
    logger.info(f"[generate_script] {provider}/{model_name} | ~{prompt_tokens} tok | ctx={num_ctx}")

    llm = get_llm_model(
        provider=provider,
        model_name=model_name,
        temperature=SCRIPT_GEN_TEMPERATURE,
        num_ctx=num_ctx,
    )

    response = llm.invoke([
        SystemMessage(content=current_system_prompt),
        HumanMessage(content=user_prompt),
    ])
    code = response.content

    # Strip markdown fences (including partial ones if truncated)
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:].strip()
    elif code.startswith("```"):
        code = code[3:].strip()
    
    if code.endswith("```"):
        code = code[:-3].strip()

    # Ensure vault snippet is present — prepend if LLM forgot it
    if "vault_creds" not in code:
        logger.warning("[generate_script] vault_creds missing — prepending snippet")
        code = _VAULT_SNIPPET + "\n" + code

    if 'if __name__ == "__main__":' not in code:
        logger.warning("[generate_script] Entry point missing — appending fallback")
        code += '\n\nif __name__ == "__main__":\n    asyncio.run(run())\n'

    if not output_path:
        output_path = str(Path(history_path).parent / f"{run_id}_LLM.py")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)

    logger.info(f"[generate_script] Saved: {output_path}")
    return output_path, code


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Generate pure Playwright script")
    parser.add_argument("history_file")
    parser.add_argument("--output",       default=None)
    parser.add_argument("--provider",     default=SCRIPT_GEN_PROVIDER)
    parser.add_argument("--model",        default=SCRIPT_GEN_MODEL)
    parser.add_argument("--vault-prefix", default=None)
    parser.add_argument("--objective",    default=None)
    args = parser.parse_args()

    path, _ = generate_script(
        history_path=args.history_file,
        output_path=args.output,
        model_name=args.model,
        provider=args.provider,
        objective=args.objective,
        vault_prefix=args.vault_prefix,
    )
    print(f"Script written to: {path}")