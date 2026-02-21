import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from playwright.async_api import Page

logger = logging.getLogger(__name__)

_RUN_ID = os.environ.get("RUN_ID") or None
_STEP_LOCK = asyncio.Lock()
_step_index = 0

# Import Ollama verifier for live verification
try:
    import sys
    from pathlib import Path
    utils_path = Path(__file__).parent
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    from ollama_max_accuracy_verifier import verify_step_element
    # Runtime toggle: set ENABLE_OLLAMA_VERIFICATION=1 to enable verification (DEBUG ONLY)
    OLLAMA_VERIFICATION_ENABLED = os.getenv("ENABLE_OLLAMA_VERIFICATION", "0") == "1"
    if OLLAMA_VERIFICATION_ENABLED:
        logger.info("✓ Live Ollama verification ENABLED for DOM capture")
    else:
        logger.info("⏸️ Live Ollama verification DISABLED (Standard Mode)")
except (ImportError, ModuleNotFoundError) as e:
    OLLAMA_VERIFICATION_ENABLED = False
    logger.warning(f"⚠️ Live Ollama verification DISABLED: {e}")


def set_run_id(run_id: str) -> None:
    """Override the run identifier used for snapshot storage."""
    global _RUN_ID, _step_index
    _RUN_ID = run_id
    _step_index = 0
    os.environ["RUN_ID"] = run_id


def _get_run_id() -> str:
    global _RUN_ID
    if not _RUN_ID:
        _RUN_ID = os.environ.get("RUN_ID") or f"run-{int(time.time())}-{uuid.uuid4().hex}"
        os.environ["RUN_ID"] = _RUN_ID
    return _RUN_ID


def _safe_hash(value: str) -> str:
    try:
        return hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest()
    except Exception:
        return ""


async def _next_step_index() -> int:
    global _step_index
    async with _STEP_LOCK:
        _step_index += 1
        return _step_index


async def capture_dom_snapshot(page: Page, reason: str, action_params: Dict[str, Any] = None):
    """
    Captures canonical DOM state for deterministic replay.
    
    Args:
        page: Playwright page object
        reason: Description of why snapshot is being taken (e.g., "pre_action:click_element_by_index")
        action_params: Optional dictionary of action parameters (e.g., {"index": 5, "text": "hello"})
    """
    try:
        if page is None:
            return

        step_index = await _next_step_index()
        timestamp = datetime.now(timezone.utc).isoformat()
        url = page.url

        navigation_id = await page.evaluate(
            "() => (performance.getEntriesByType('navigation') || []).slice(-1)[0]?.name || document.referrer || document.location.href"
        )

        # PERF: Faster way to get page HTML than page.content() which can be heavy
        page_html = await page.evaluate("() => document.documentElement.outerHTML")
        page_dom_hash = _safe_hash(page_html)

        dom_data: List[Dict[str, Any]] = await page.evaluate(
            """
            () => {
                const selectors = [
                    'a', 'button', 'input', 'select', 'textarea', 'option', 'summary', 'details', '[role]', '[tabindex]',
                    'table', 'thead', 'tbody', 'tr', 'th', 'td'
                ];
                const elements = Array.from(document.querySelectorAll(selectors.join(',')));

                const escapeCss = (value) => {
                    if (typeof CSS !== 'undefined' && CSS.escape) return CSS.escape(value);
                    return String(value).replace(/([ !"#$%&'()*+,./:;<=>?@[\\]^`{|}~])/g, '\\$1');
                };

                // OPTIMIZED Path Builders
                const buildCssPath = (el) => {
                    const segments = [];
                    let current = el;
                    while (current && current.tagName) {
                        let segment = current.tagName.toLowerCase();
                        if (current.id) {
                            segment += `#${escapeCss(current.id)}`;
                            segments.unshift(segment);
                            break;
                        }
                        const className = current.classList ? current.classList[0] : null;
                        if (className) segment += `.${escapeCss(className)}`;
                        
                        const parent = current.parentElement;
                        if (parent) {
                            let index = 1;
                            for (let i = 0; i < parent.children.length; i++) {
                                if (parent.children[i] === current) {
                                    index = i + 1;
                                    break;
                                }
                            }
                            segment += `:nth-child(${index})`;
                        }
                        segments.unshift(segment);
                        current = parent;
                    }
                    return segments.join(' > ');
                };

                const buildXPath = (el) => {
                    const parts = [];
                    let current = el;
                    while (current && current.nodeType === Node.ELEMENT_NODE) {
                        let index = 1;
                        let sibling = current.previousElementSibling;
                        while (sibling) {
                            if (sibling.tagName === current.tagName) index += 1;
                            sibling = sibling.previousElementSibling;
                        }
                        parts.unshift(`${current.tagName.toLowerCase()}[${index}]`);
                        current = current.parentElement;
                    }
                    return '/' + parts.join('/');
                };

                // Frame path detection
                let framePath = [];
                try {
                    let win = window;
                    while (win.parent && win !== win.parent) {
                        framePath.unshift(win.frameElement ? win.frameElement.tagName.toLowerCase() : 'iframe');
                        win = win.parent;
                    }
                } catch(e) {}

                return elements.map((el, idx) => {
                    const rect = el.getBoundingClientRect();
                    const tagName = el.tagName.toLowerCase();
                    const isTablePart = ["tr", "td", "th", "thead", "tbody"].includes(tagName);
                    
                    // PERF: Skip computed style for table internals unless it's a direct action target
                    let style = { visibility: 'visible', display: 'block' };
                    if (!isTablePart) {
                        style = window.getComputedStyle(el);
                    }

                    const dataAttrs = {};
                    for (const attr of el.attributes) {
                        if (attr.name.startsWith('data-') || attr.name === 'atp') {
                            dataAttrs[attr.name] = attr.value;
                        }
                    }

                    const textSelector = (el.innerText || el.textContent || '').trim().substring(0, 100);
                    const ariaLabel = el.getAttribute('aria-label') || el.getAttribute('aria-labelledby') || '';
                    
                    // Preferred selector builder (optimized)
                    let preferred = el.id ? `#${escapeCss(el.id)}` : null;
                    if (!preferred && ariaLabel) preferred = `[aria-label="${ariaLabel.replace(/"/g, '\\"')}"]`;
                    if (!preferred) {
                        const dataKeys = Object.keys(dataAttrs);
                        if (dataKeys.length) preferred = `[${dataKeys[0]}="${String(dataAttrs[dataKeys[0]]).replace(/"/g, '\\"')}"]`;
                    }
                    if (!preferred) preferred = buildCssPath(el);

                    return {
                        tag_name: tagName,
                        child_index: idx, // Simplified sequence index
                        frame_path: framePath,
                        selectors: {
                            css: "", // Lazy built in script if needed
                            xpath: "", // Lazy built in script if needed
                            text: textSelector,
                            aria: ariaLabel,
                            preferred,
                        },
                        attributes: {
                            id: el.id || '',
                            name: el.getAttribute('name') || '',
                            role: el.getAttribute('role') || '',
                            type: el.getAttribute('type') || '',
                            data: dataAttrs,
                        },
                        state: {
                            visible: rect.width > 0 && rect.height > 0,
                            enabled: !el.disabled,
                            bounding_box: {
                                x: rect.x + window.scrollX,
                                y: rect.y + window.scrollY,
                                width: rect.width,
                                height: rect.height,
                            },
                        },
                        integrity: {
                            outer_html: el.outerHTML.substring(0, 500), // Limit size for snapshot
                        },
                        order: idx,
                    };
                });
            }
            """
        )

        elements = []
        for el in dom_data:
            outer_html = el.get("integrity", {}).get("outer_html", "")
            elements.append(
                {
                    "identity": {
                        "tagName": el.get("tag_name", ""),
                        "parent_path": el.get("parent_path", []),
                        "child_index": el.get("child_index", 0),
                        "frame_path": el.get("frame_path", []),
                        "order": el.get("order"),
                    },
                    "selector_provenance": {
                        "css": el.get("selectors", {}).get("css", ""),
                        "xpath": el.get("selectors", {}).get("xpath", ""),
                        "text": el.get("selectors", {}).get("text", ""),
                        "aria": el.get("selectors", {}).get("aria", ""),
                        "preferred": el.get("selectors", {}).get("preferred", ""),
                    },
                    "attribute_fingerprint": {
                        "id": el.get("attributes", {}).get("id", ""),
                        "name": el.get("attributes", {}).get("name", ""),
                        "class_list": el.get("attributes", {}).get("class_list", []),
                        "role": el.get("attributes", {}).get("role", ""),
                        "type": el.get("attributes", {}).get("type", ""),
                        "data": el.get("attributes", {}).get("data", {}),
                    },
                    "state": el.get("state", {}),
                    "integrity": {
                        "outerHTML": outer_html,
                        "subtree_hash": _safe_hash(outer_html),
                        "page_dom_hash": page_dom_hash,
                    },
                }
            )
        
        # ============================================================
        # LIVE OLLAMA VERIFICATION - Verify each element at capture time
        # ============================================================
        if OLLAMA_VERIFICATION_ENABLED and action_params:
            action_name = action_params.get("action_name", "unknown")
            
            # Only verify for extract actions (extract_content, extract_table, etc.)
            is_extract_action = "extract" in action_name.lower()
            
            if is_extract_action:
                # Smart filtering: prioritize relevant elements for data extraction
                # Generic approach works across different websites
                MAX_ELEMENTS_TO_VERIFY = 30  # Configurable limit

                # First, try to directly target elements related to the action parameters
                def match_action_target(el, ap):
                    try:
                        fp = el.get("attribute_fingerprint", {})
                        prov = el.get("selector_provenance", {})
                        data_attrs = fp.get("data", {}) or {}
                        outer_html = el.get("integrity", {}).get("outerHTML", "")
                        text_val = prov.get("text", "")

                        # Match by index set via browser-use instrumentation
                        idx = ap.get("index") or ap.get("element_index")
                        if idx is not None:
                            for key in ("data-browser-use-index", "data-browser-index", "data-bu-index"):
                                if str(data_attrs.get(key, "")) == str(idx):
                                    return True

                        # Match by CSS/XPath/Preferred selector
                        css_param = ap.get("css") or ap.get("selector") or ap.get("preferred")
                        if css_param:
                            if css_param == prov.get("preferred") or css_param == prov.get("css"):
                                return True

                        xpath_param = ap.get("xpath")
                        if xpath_param and xpath_param == prov.get("xpath"):
                            return True

                        # Match by id/name
                        id_param = ap.get("id")
                        if id_param and id_param == fp.get("id"):
                            return True
                        name_param = ap.get("name")
                        if name_param and name_param == fp.get("name"):
                            return True

                        # Match by text snippet
                        txt = (ap.get("text") or "").strip()
                        if txt:
                            if txt in text_val or txt in outer_html:
                                return True
                    except Exception:
                        pass
                    return False

                targeted_elements = []
                if action_params:
                    targeted_elements = [el for el in elements if match_action_target(el, action_params)]

                # Filter elements: prioritize data-rich elements using generic patterns
                def is_relevant_for_extraction(el):
                    """Check if element is relevant for data extraction (generic algorithm)"""
                    tag_name = el.get("attributes", {}).get("tag_name", "").lower()
                    text = el.get("selectors", {}).get("text", "").strip()
                    class_list = el.get("attributes", {}).get("class_list", [])
                    classes = " ".join(class_list).lower()

                    # Priority 1: Table structure elements (universal)
                    if tag_name in ["td", "th"]:
                        if len(text) > 0:
                            return True

                    # Priority 2: Table rows with data (not just headers)
                    if tag_name == "tr" and len(text) > 5:
                        return True

                    # Priority 3: Elements with status/error/value/data indicators
                    generic_data_indicators = ["status", "error", "value", "data", "result", "content", "item"]
                    if any(indicator in classes for indicator in generic_data_indicators):
                        return True

                    # Priority 4: Semantic tags with meaningful content
                    if tag_name in ["span", "div", "p"] and len(text) > 3:
                        has_data_words = any(word in text.lower() for word in [
                            "active", "inactive", "error", "success", "failed", "running", "stopped", "pending", "complete"
                        ])
                        if has_data_words:
                            return True

                    # Priority 5: Numeric data (counts, values, IDs)
                    if text.strip().isdigit() and len(text) <= 10:
                        return True

                    # Priority 6: Links within tables (often data identifiers)
                    if tag_name == "a" and len(text) > 2:
                        return True

                    return False

                # Build final list: targeted elements first, then heuristic ones
                relevant_elements = [el for el in elements if is_relevant_for_extraction(el)]

                # Deduplicate by subtree hash
                seen_hashes = set()
                def dedup(seq):
                    out = []
                    for el in seq:
                        h = el.get("integrity", {}).get("subtree_hash")
                        if h and h not in seen_hashes:
                            seen_hashes.add(h)
                            out.append(el)
                    return out

                elements_to_verify = dedup(targeted_elements) + dedup(relevant_elements)
                elements_to_verify = elements_to_verify[:MAX_ELEMENTS_TO_VERIFY]

                logger.info(
                    f"🔍 Live Ollama verification (EXTRACT): {len(elements_to_verify)}/{len(elements)} elements "
                    f"(targeted {len(targeted_elements)}, heuristic {len(relevant_elements)})"
                )
                
                for idx, element in enumerate(elements_to_verify):
                    try:
                        # Extract element details for verification
                        element_text = element.get("selectors", {}).get("text", "")[:500]
                        element_html = element.get("integrity", {}).get("outerHTML", "")[:1000]
                        element_selector = element.get("selectors", {}).get("preferred", "")
                        
                        # Build extraction requirement based on action
                        extraction_req = f"{action_name} action on element"
                        if "text" in action_params:
                            extraction_req += f" with text: {action_params['text']}"
                        if "index" in action_params:
                            extraction_req += f" at index: {action_params['index']}"
                        
                        # Verify element with Ollama
                        is_valid, verification_result = verify_step_element(
                            step_name=reason,
                            element_text=element_text,
                            element_html=element_html,
                            element_selector=element_selector,
                            extraction_requirement=extraction_req,
                            expected_keywords=[action_name],
                            step_action=action_name
                        )
                        
                        # Add verification to element
                        element["ollama_verification"] = {
                            "verified_at_capture": True,
                            "is_valid": is_valid,
                            "confidence": verification_result.get("confidence", 0),
                            "method": verification_result.get("method", "Ollama (qwen2.5:14b)"),
                            "timestamp": verification_result.get("timestamp"),
                            "verification_stages": verification_result.get("verification_stages", {}),
                            "reason": verification_result.get("reason", "")
                        }
                        
                        if is_valid:
                            logger.debug(f"  ✓ Element {idx}: {verification_result.get('confidence')}% confidence")
                        else:
                            logger.debug(f"  ✗ Element {idx}: INVALID - {verification_result.get('reason')}")
                            
                    except Exception as verify_err:
                        logger.warning(f"Verification failed for element {idx}: {verify_err}")
                        element["ollama_verification"] = {
                            "verified_at_capture": True,
                            "is_valid": False,
                            "confidence": 0,
                        "error": str(verify_err)
                    }
                
                verified_count = sum(1 for el in elements_to_verify if el.get("ollama_verification", {}).get("is_valid", False))
                logger.info(f"✓ Verified {verified_count}/{len(elements_to_verify)} elements with Ollama (EXTRACT) - {len(elements) - len(elements_to_verify)} skipped")
            else:
                logger.debug(f"⏭️  Skipping verification for non-extract action: {action_name}")
        # ============================================================

        run_id = _get_run_id()

        snapshot = {
            "metadata": {
                "run_id": run_id,
                "step_index": step_index,
                "timestamp": timestamp,
                "reason": reason,
                "url": url,
                "navigation_id": navigation_id,
                "page_dom_hash": page_dom_hash,
                "action_params": action_params or {},
            },
            "elements": elements,
        }

        snapshot_dir = os.path.join("tmp", "dom_snapshots", run_id)
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshot_dir, f"{step_index}.json")

        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=True, indent=2)

        logger.info("DOM snapshot stored", extra={"snapshot_path": snapshot_path, "reason": reason, "step_index": step_index})
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Failed to capture DOM snapshot: {exc}")
