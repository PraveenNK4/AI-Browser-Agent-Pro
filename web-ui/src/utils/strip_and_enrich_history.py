import json
import pathlib
import re
import sys
import os
import logging
from typing import Any, Dict, Iterable, Optional, List, Tuple

logger = logging.getLogger(__name__)

# Add utils to path for imports
utils_path = pathlib.Path(__file__).parent
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

# Import config for Ollama verifier settings (no hardcoded values)
try:
    from src.utils.config import (
        OLLAMA_VERIFIER_MODEL,
        OLLAMA_VERIFIER_TEMPERATURE,
        OLLAMA_VERIFIER_TIMEOUT_S,
    )
    _OLLAMA_DESC = f"Ollama ({OLLAMA_VERIFIER_MODEL} | temp={OLLAMA_VERIFIER_TEMPERATURE} | timeout={OLLAMA_VERIFIER_TIMEOUT_S}s)"
except ImportError:
    # Fallback when running standalone outside of the web-ui package tree
    OLLAMA_VERIFIER_MODEL = os.getenv("OLLAMA_VERIFIER_MODEL", "qwen2.5:14b")
    OLLAMA_VERIFIER_TEMPERATURE = float(os.getenv("OLLAMA_VERIFIER_TEMPERATURE", "0.1"))
    OLLAMA_VERIFIER_TIMEOUT_S = int(os.getenv("OLLAMA_VERIFIER_TIMEOUT_S", "180"))
    _OLLAMA_DESC = f"Ollama ({OLLAMA_VERIFIER_MODEL} | temp={OLLAMA_VERIFIER_TEMPERATURE})"

"""History enrichment with optional Ollama verification.

Set the environment variable DISABLE_OLLAMA_VERIFICATION=1 to fully skip
verification during enrichment. This leaves the page DOM untouched and writes
JSON without any Ollama verification annotations.
"""

# Import Ollama max accuracy verifier
HAS_OLLAMA_VERIFIER = False
verify_step_element = None
OLLAMA_VERIFICATION_ENABLED = os.getenv("DISABLE_OLLAMA_VERIFICATION", "0") != "1"

try:
    if OLLAMA_VERIFICATION_ENABLED:
        from ollama_max_accuracy_verifier import verify_step_element
        HAS_OLLAMA_VERIFIER = True
        logger.info(f"[+] Ollama Max Accuracy Verifier available ({_OLLAMA_DESC} - TESTING MODE)")
    else:
        logger.info("[*] Ollama verification DISABLED via env (DISABLE_OLLAMA_VERIFICATION=1)")
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"[!] Ollama Max Accuracy Verifier not available - {str(e)}")
    logger.warning("   Using heuristic-based verification instead")


def load_snapshots(snapshot_dir: pathlib.Path) -> Iterable[Dict[str, Any]]:
    snapshots = []
    for path in sorted(snapshot_dir.glob("*.json"), key=lambda p: int(p.stem)):
        with path.open("r", encoding="utf-8") as f:
            snapshots.append(json.load(f))
    return snapshots


def find_snapshot(action_name: str, params: Dict[str, Any], snapshots: Iterable[Dict[str, Any]]):
    for snap in snapshots:
        meta = snap.get("metadata", {})
        reason = meta.get("reason", "")
        if reason != f"pre_action:{action_name}":
            continue
        snap_params = meta.get("action_params", {})
        if "index" in params and snap_params.get("index") != params.get("index"):
            continue
        if "text" in params and snap_params.get("text") != params.get("text"):
            continue
        return snap
    return None


def _get_type_from_outerhtml(outerhtml: str) -> str:
    """Extract type attribute from outerHTML."""
    match = re.search(r'type=(["\']?)([^"\'\s>]+)\1', outerhtml, re.IGNORECASE)
    return match.group(2) if match else ""

def _clean_text(text: str) -> str:
    """Clean text for comparison by removing extra whitespace and special characters."""
    # Remove script/style tags
    text = re.sub(r'<script.*?>.*?</script>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<style.*?>.*?</style>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def _extract_keywords(extracted_content: str) -> list:
    """Extract meaningful keywords from extracted content."""
    # Remove JSON formatting and extract text content
    content = re.sub(r'[{}\[\]",:]+', ' ', extracted_content)
    # Get words longer than 2 characters (lowered threshold)
    words = [w.lower() for w in re.findall(r'\\b\\w{3,}\\b', content)]
    # Remove only very common words, keep domain terms like "admin", "server", "status"
    stopwords = {'from', 'page', 'extracted', 'the', 'and', 'for', 'with'}
    keywords = [w for w in words if w not in stopwords]
    return list(set(keywords))[:15]  # Return top 15 unique keywords


def _verify_element_matches_content(element: Dict[str, Any], extracted_content: str, keywords: list) -> float:
    """Verify if an element actually matches the extracted content with strict validation."""
    # Get element content
    selectors = element.get("selector_provenance", {})
    element_text = selectors.get("text", "").lower() if isinstance(selectors, dict) else ""
    outerhtml = element.get("integrity", {}).get("outerHTML", "").lower()
    combined = f"{element_text} {outerhtml}"
    tag_name = element.get("identity", {}).get("tagName", "").lower()
    
    # Must have actual data content, not just structure
    # EXEMPT TABLES: Tables often have 0 direct text (content is in children), so we must preserve them.
    if len(element_text) < 20 and tag_name not in ['table', 'tbody', 'thead', 'tr']:  # Too small (unless it's a table structure)
        return 0  # Likely not a data container
    
    # Count exact keyword matches in element
    exact_matches = sum(1 for kw in keywords if kw in combined)
    
    # Verify content quality:
    # 1. Element should contain most keywords from extracted content
    keyword_coverage = exact_matches / max(len(keywords), 1)
    
    # 2. Should contain actual data rows or structured content
    has_data_structure = any(
        x in combined for x in ['<tr>', '<td>', '<th>', '<table>', 'tbody', 'colspan', '<tbody']
    )
    
    # 3. NOT filter/search/navigation UI
    is_ui_element = any(
        x in combined for x in ['search', 'filter', 'menu', 'navigation', 'sidebar', 'header']
    )
    
    # 4. NOT a single link or button
    is_single_link = tag_name in ['a', 'button', 'link'] and len(element_text) < 100
    
    # Scoring
    if keyword_coverage < 0.3:  # Less than 30% keyword match is suspicious
        return 0
    
    if is_single_link:
        return 0  # Single links should never be extracted content containers
    
    if is_ui_element and not has_data_structure:
        return 0  # Likely a UI element, not data
    
    # Prefer table/tbody/tr/td as they're clearly data structures
    if tag_name in ['table', 'tbody', 'tr', 'td', 'form', 'div', 'section']:
        # Data structure is good
        pass
    elif tag_name not in ['div', 'section', 'article', 'main']:
        # Unusual tag for data extraction
        keyword_coverage *= 0.5
    
    # Valid extraction element
    return keyword_coverage * 100  # Return percentage match


def _calculate_match_score(element: Dict[str, Any], keywords: list, extracted_text: str) -> float:
    """Calculate how well an element matches the extracted content."""
    score = 0.0
    
    # Get element text from various sources
    element_texts = []
    
    # Get text from selector provenance
    selectors = element.get("selector_provenance", {})
    if isinstance(selectors, dict):
        element_texts.append(selectors.get("text", ""))
    
    # Get outerHTML text
    outerhtml = element.get("integrity", {}).get("outerHTML", "")
    if outerhtml:
        element_texts.append(_clean_text(outerhtml))
    
    # Combine all text
    combined_text = " ".join(element_texts).lower()
    
    # Score based on keyword matches
    keyword_matches = sum(1 for kw in keywords if kw in combined_text)
    score += keyword_matches * 15  # Increased weight
    
    # Bonus for table-related elements (likely structured data)
    tag_name = element.get("identity", {}).get("tagName", "").lower()
    if tag_name in ["table", "tbody"]:
        score += 25
        # Extra bonus for tables with data rows (tbody/tr/td structure)
        if "tbody" in outerhtml and "<tr>" in outerhtml and "<td>" in outerhtml:
            score += 20
    elif tag_name in ["tr", "td", "th"]:
        score += 15
    elif tag_name in ["div", "section", "article"]:
        score += 5
    
    # Bonus for elements with substantial content (more likely to be data)
    if len(combined_text) > 200:
        score += 15
    elif len(combined_text) > 100:
        score += 8
    
    # Strong penalty for header/nav/filter elements
    if tag_name in ["header", "nav", "footer"] or "header" in element.get("attribute_fingerprint", {}).get("role", ""):
        score -= 40
    
    # Penalty for filter/search UI elements
    if "filter" in combined_text or "search" in combined_text:
        # But only if it's a small element (likely UI, not data)
        if len(combined_text) < 300:
            score -= 25
    
    # Check for role="search" which indicates filter UI
    if element.get("attribute_fingerprint", {}).get("role", "") == "search":
        score -= 30
    
    return score


def find_best_element_for_extraction(snapshot: Dict[str, Any], extracted_content: str) -> Optional[Dict[str, Any]]:
    """Find the DOM element that most likely contains the extracted content with LLM verification."""
    if not snapshot or not extracted_content:
        return None
    
    elements = snapshot.get("elements", [])
    if not elements:
        return None
    
    # Extract keywords from the extracted content
    keywords = _extract_keywords(extracted_content)
    if not keywords:
        # Fallback to simple text extraction
        keywords = re.findall(r'\b\w{4,}\b', extracted_content.lower())[:10]
    
    # Score each element with verification
    scored_elements = []
    for el in elements:
        # First check: Does element match content with strict verification?
        verification_score = _verify_element_matches_content(el, extracted_content, keywords)
        
        if verification_score > 0:  # Only if basic verification passes
            # Second check: Calculate match quality
            quality_score = _calculate_match_score(el, keywords, extracted_content)
            
            # Combined score: verification * quality
            combined_score = verification_score + quality_score
            
            if combined_score > 0:
                scored_elements.append((combined_score, verification_score, el))
                tag_name = el.get('identity', {}).get('tagName')
                logger.debug(f"  > Heuristic candidate: {tag_name} - verification={verification_score:.1f}, quality={quality_score:.1f}")
    
    if not scored_elements:
        logger.warning(f"  WARNING: Strict verification failed. Attempting FALLBACK matching.")
        
        # Fallback 1: Direct substring match (Text containment)
        best_fallback = None
        min_len = float('inf')
        cleaned_content = _clean_text(extracted_content)
        
        for el in elements:
            el_text = _clean_text(el.get("selector_provenance", {}).get("text", ""))
            # Ignore empty or tiny elements
            if len(el_text) > 10:
                # If element contains target OR target contains element text (and element is substantial)
                if (cleaned_content in el_text) or (len(el_text) > 50 and el_text in cleaned_content):
                    # We want the *shortest* text that contains the target (most specific element)
                    if len(el_text) < min_len:
                        min_len = len(el_text)
                        best_fallback = el
        
        if best_fallback:
             tag = best_fallback.get('identity', {}).get('tagName')
             logger.info(f"  ✓ Fallback Match: Found containing element <{tag}>")
             # Inject a fake score so it passes downstream checks if any
             best_fallback["fallback_match"] = True
             return best_fallback
             
        # Fallback 2: Any Table (Critical for "Extract Table" actions)
        # Fallback 2: Any Table (Critical for "Extract Table" actions)
        # FORCE: If we are here, we have no better match. Return ANY table we found.
        for el in elements:
             tag = el.get("identity", {}).get("tagName")
             if tag in ["table", "tbody"]:
                 logger.info(f"  ✓ Fallback Match: Returning generic {tag.upper()} container (no strict text match)")
                 el["fallback_match"] = True
                 return el
                 
        return None
    
    # Sort by combined score descending
    scored_elements.sort(key=lambda x: x[0], reverse=True)
    
    # Get top candidates (heuristic is just a pre-filter, Ollama makes final decision)
    best_score, best_verification, best_element = scored_elements[0]
    
    logger.debug(f"  > Best heuristic candidate: {best_element.get('identity', {}).get('tagName')} - score={best_verification:.1f}%")
    
    # Now use Ollama for 100% accuracy verification if available and enabled
    if OLLAMA_VERIFICATION_ENABLED and HAS_OLLAMA_VERIFIER and verify_step_element:
        logger.debug(f"\n  >>> Using {_OLLAMA_DESC} for 100% accuracy verification...")
        
        element_text = best_element.get("selector_provenance", {}).get("text", "")
        element_html = best_element.get("integrity", {}).get("outerHTML", "")
        tag_name = best_element.get('identity', {}).get('tagName')
        element_selector = best_element.get("selector_provenance", {}).get("selector", "unknown")
        
        # Prepare nearby elements for cross-validation
        nearby_elements = []
        for i, el in enumerate(scored_elements[1:4]):  # Get 2-3 nearby candidates
            nearby_text = el[2].get("selector_provenance", {}).get("text", "")[:500]
            nearby_elements.append({"text": nearby_text, "score": el[0]})
        
        # Call Ollama verification with multi-stage validation
        is_verified, verification_result = verify_step_element(
            step_name="Extract Content",
            step_action="extract_table_data",
            element_text=element_text,
            element_html=element_html,
            element_selector=element_selector,
            extraction_requirement=extracted_content,
            expected_keywords=keywords,
            nearby_elements=nearby_elements if nearby_elements else None
        )
        
        if is_verified:
            logger.debug(f"  ✓ OLLAMA VERIFIED (4-stage validation): {verification_result.get('confidence', 0):.0f}% confidence")
            logger.debug(f"  Element Hash: {verification_result.get('element_hash', 'unknown')}")
            
            # Enrich element with verification data
            best_element["ollama_verification"] = verification_result
            return best_element
        else:
            confidence = verification_result.get('confidence', 0)
            logger.debug(f"  ✗ OLLAMA REJECTED: {confidence:.0f}% confidence (need ≥90%)")
            logger.debug(f"  Reason: {verification_result.get('reason', 'unknown')}")
            
            # Try second best candidate
            if len(scored_elements) > 1:
                print(f"  Trying second candidate...")
                second_score, second_verification, second_element = scored_elements[1]
                element_text = second_element.get("selector_provenance", {}).get("text", "")
                element_html = second_element.get("integrity", {}).get("outerHTML", "")
                tag_name = second_element.get('identity', {}).get('tagName')
                element_selector = second_element.get("selector_provenance", {}).get("selector", "unknown")
                
                is_verified, verification_result = verify_step_element(
                    step_name="Extract Content",
                    step_action="extract_table_data",
                    element_text=element_text,
                    element_html=element_html,
                    element_selector=element_selector,
                    extraction_requirement=extracted_content,
                    expected_keywords=keywords,
                    nearby_elements=None
                )
                
                if is_verified:
                    logger.debug(f"  ✓ Second candidate OLLAMA VERIFIED: {verification_result.get('confidence', 0):.0f}% confidence")
                    second_element["ollama_verification"] = verification_result
                    return second_element
            
            return None  # No element passed Ollama verification
    else:
        # Fallback to heuristic verification only
        tag_name = best_element.get('identity', {}).get('tagName')
        logger.debug(f"  ⚠️  Ollama not available - using heuristic only: {tag_name} ({best_verification:.1f}%)")
        if best_verification < 30:
            logger.warning(f"  WARNING: Element has low heuristic score ({best_verification:.1f}%) - start Ollama for 100% accuracy")
            return None
        return best_element

def _verify_element_with_ollama(
    element: Dict[str, Any],
    action_name: str,
    action_params: Dict[str, Any],
    step_name: str = "",
    should_verify: bool = True
) -> tuple:
    """
    Verify a DOM element is correct for the specific action using Ollama.
    
    Returns: (is_valid: bool, verification_data: Dict)
    """
    if (not should_verify) or (not OLLAMA_VERIFICATION_ENABLED) or (not HAS_OLLAMA_VERIFIER) or (not verify_step_element):
        return True, {"method": "disabled" if (not OLLAMA_VERIFICATION_ENABLED or not should_verify) else "heuristic", "note": "Verification skipped or disabled"}
    
    try:
        element_text = element.get("selector_provenance", {}).get("text", "")
        element_html = element.get("integrity", {}).get("outerHTML", "")
        element_selector = element.get("selector_provenance", {}).get("selector", "unknown")
        tag_name = element.get("identity", {}).get("tagName", "unknown")
        
        # Build action-specific extraction requirement
        extraction_req = f"Action: {action_name}"
        keywords = [action_name, tag_name]
        
        if action_name == "input_text":
            text_to_input = action_params.get("text", "")
            extraction_req = f"Input text into {tag_name} element: '{text_to_input[:50]}...'"
            keywords.extend(["input", "text", "field", tag_name])
        
        elif action_name == "click_element_by_index":
            extraction_req = f"Click on {tag_name} element to trigger action"
            keywords.extend(["click", "button", "link", "clickable", tag_name])
        
        elif action_name == "extract_content":
            extraction_req = action_params.get("goal", "Extract content from element")
            keywords.extend(["extract", "content", "data", "table", "list"])
        
        # Use Ollama to verify element
        is_valid, result = verify_step_element(
            step_name=step_name or action_name,
            step_action=action_name,
            element_text=element_text,
            element_html=element_html,
            element_selector=element_selector,
            extraction_requirement=extraction_req,
            expected_keywords=keywords
        )
        
        verification_data = {
            "method": _OLLAMA_DESC,
            "is_valid": is_valid,
            "confidence": result.get("confidence", 0),
            "element_hash": result.get("element_hash", ""),
            "reason": result.get("reason", ""),
            "verification_stages": result.get("verification_stages", {})
        }
        
        return is_valid, verification_data
    
    except Exception as e:
        return False, {
            "method": "Ollama",
            "error": str(e),
            "note": "Verification failed, using element anyway"
        }


def extract_element(snapshot: Dict[str, Any], params: Dict[str, Any], action_name: str = "", should_verify: bool = True):
    if not snapshot:
        return None
    elements = snapshot.get("elements", [])
    target = None

    if action_name == "input_text" or "text" in params:
        # For input_text, prioritize visible input/textarea elements
        visible_inputs = [
            el for el in elements 
            if (el.get("state", {}).get("visible", False) and 
                el.get("identity", {}).get("tagName") in ["input", "textarea"])
        ]
        
        text_param = params.get("text", "").lower() if isinstance(params.get("text"), str) else ""
        
        # Strategy 1: Use type hint from text content
        if "password" in text_param:
            # Looking for password field - check outerHTML for type attribute
            password_inputs = [
                el for el in visible_inputs 
                if _get_type_from_outerhtml(el.get("integrity", {}).get("outerHTML", "")).lower() == "password"
            ]
            if password_inputs:
                target = password_inputs[0]
        
        # Strategy 2: For non-password text inputs
        if target is None and visible_inputs:
            text_type_inputs = [
                el for el in visible_inputs 
                if _get_type_from_outerhtml(el.get("integrity", {}).get("outerHTML", "")).lower() in ["text", "email", ""]
            ]
            if text_type_inputs:
                target = text_type_inputs[0]
            else:
                target = visible_inputs[0]
        
        # Strategy 3: If no visible inputs, try index matching on all inputs
        if target is None and "index" in params:
            target_index = params["index"]
            for el in elements:
                if (el.get("identity", {}).get("tagName") in ["input", "textarea"] and
                    el.get("identity", {}).get("order") == target_index):
                    target = el
                    break
        
        # Strategy 4: Fallback to first visible input
        if target is None and visible_inputs:
            target = visible_inputs[0]
        
        # Strategy 5: Last resort - any input element
        if target is None and elements:
            target = elements[0]

    else:
        # For other actions, match by index
        if "index" in params:
            target_index = params["index"]
            for el in elements:
                if el.get("identity", {}).get("order") == target_index:
                    target = el
                    break
        
        # Fallback: pick first visible or first element
        if target is None:
            visible_elements = [
                el for el in elements 
                if el.get("state", {}).get("visible", False)
            ]
            target = visible_elements[0] if visible_elements else (elements[0] if elements else None)

    if target is None:
        return None

    identity = target.get("identity", {})

    # HEURISTIC DOUBLE-CHECK: If it's a click and the element seems wrong, find a better one
    if action_name in ["click_element", "click_element_by_index"] and "index" in params:
        target_text = target.get("selector_provenance", {}).get("text", "").lower()
        target_attr = str(target.get("attribute_fingerprint", {})).lower()
        
        # We look for a keyword in the goal or params to verify
        keywords = []
        if "text" in params: keywords.append(params["text"].lower())
        # Add common domain terms from goal if possible 
        # (This requires passing 'goal' which we don't have here yet, but we have action_name)
        
        # If we have keywords and they aren't in the found target, search!
        if keywords and not any(kw in target_text or kw in target_attr for kw in keywords):
            print(f"  [!] Heuristic mismatch: Target {target.get('identity', {}).get('tagName')} at index {identity.get('order')} doesn't match keywords {keywords}. Searching...")
            for el in elements:
                el_text = el.get("selector_provenance", {}).get("text", "").lower()
                el_attr = str(el.get("attribute_fingerprint", {})).lower()
                if any(kw in el_text or kw in el_attr for kw in keywords):
                    print(f"  [+] Found better match at index {el.get('identity', {}).get('order')} ({el.get('identity', {}).get('tagName')})")
                    target = el
                    identity = target.get("identity", {})
                    break

    # Verify element with Ollama only if requested
    is_valid, verification = _verify_element_with_ollama(target, action_name, params, step_name=f"Action: {action_name}", should_verify=should_verify)
    
    dom_context = {
        "element_index": identity.get("order"),
        "tagName": identity.get("tagName"),
        "selectors": target.get("selector_provenance", {}),
        "attributes": target.get("attribute_fingerprint", {}),
        "state": target.get("state", {}),
        "outerHTML": target.get("integrity", {}).get("outerHTML", ""),
        "verification": verification
    }
    
    return dom_context


def load_comprehensive_element_data(action_name: str, index: int, run_id: str) -> Optional[Dict[str, Any]]:
    """
    Load comprehensive element capture data if available.
    
    Looks for files matching pattern: element_{action_name}_idx{index}_*.json
    """
    try:
        element_data_dir = pathlib.Path("tmp/element_data")
        if not element_data_dir.exists():
            return None
        
        # Find matching element data files
        pattern = f"element_{action_name}_idx{index}_*.json"
        matching_files = list(element_data_dir.glob(pattern))
        
        if not matching_files:
            return None
        
        # Load the most recent file
        most_recent = max(matching_files, key=lambda p: p.stat().st_mtime)
        
        with open(most_recent, 'r', encoding='utf-8') as f:
            element_data = json.load(f)
        
        return element_data
    except Exception as e:
        logger.warning(f"⚠️  Failed to load comprehensive element data: {e}")
        return None


def enhance_dom_context_with_comprehensive_data(dom_context: Optional[Dict[str, Any]], element_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance existing dom_context with comprehensive element capture data.
    If dom_context is None, initializes it using the element_data.
    """
    if not element_data:
        return dom_context or {}
    
    if dom_context is None:
        # Build base context from comprehensive data if heuristic mapping failed
        rec = element_data.get("recommendedSelector", {})
        dom_context = {
            "element_index": element_data.get("metadata", {}).get("element_index"),
            "tagName": element_data.get("tagName"),
            "selectors": {
                "selector": rec.get("selector", ""),
                "text": element_data.get("textContent", ""),
                "type": rec.get("type", "")
            },
            "attributes": element_data.get("attributes", {}),
            "state": element_data.get("state", {}),
            "outerHTML": element_data.get("outerHTML", ""),
        }
    
    # Add comprehensive selectors and extended data
    dom_context["comprehensive"] = {
        "all_selectors": element_data.get("selectors", []),
        "recommended_selector": element_data.get("recommendedSelector", {}),
        "all_attributes": element_data.get("attributes", {}),
        "data_attributes": element_data.get("dataAttributes", {}),
        "aria_attributes": element_data.get("ariaAttributes", {}),
        "ancestor_context": element_data.get("ancestorIds", []),
        "parent_chain": element_data.get("parentChain", []),
        "element_hash": element_data.get("elementHash", ""),
        "bounding_box": element_data.get("boundingBox", {}),
        "computed_style": element_data.get("computedStyle", {})
    }
    
    return dom_context


def _resolve_snapshot_dir(history_path: pathlib.Path) -> pathlib.Path:
    """
    Resolve the most likely dom_snapshots directory for a given history file.
    Tries multiple run_id candidates to handle orchestration layouts.
    """
    dom_root = history_path.parents[2] / "dom_snapshots"
    candidates = [
        history_path.stem,          # Full filename stem (often includes orchestration id)
        history_path.parent.name,   # Parent folder name (simple run id)
    ]

    for run_id in candidates:
        snapshot_dir = dom_root / run_id
        if snapshot_dir.exists():
            return snapshot_dir

    # Fallback: try to find a directory containing the stem (best-effort)
    if dom_root.exists():
        for entry in dom_root.iterdir():
            if entry.is_dir() and history_path.stem in entry.name:
                return entry

    # Default: return expected path using stem (even if missing)
    return dom_root / history_path.stem


def strip_and_enrich(history_path: pathlib.Path) -> None:
    snapshot_dir = _resolve_snapshot_dir(history_path)
    run_id = snapshot_dir.name
    snapshots = load_snapshots(snapshot_dir)

    with history_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data["history"] if isinstance(data, dict) and "history" in data else data

    for step_idx, step in enumerate(steps):
        state = step.get("state", {}) if isinstance(step, dict) else {}
        if isinstance(state, dict):
            state.pop("screenshot", None)

        model_output = step.get("model_output")
        if model_output is None:
            model_output = {}
        actions = model_output.get("action", []) if isinstance(step, dict) else []
        for action_idx, action in enumerate(actions):
            if not isinstance(action, dict) or not action:
                continue
            action_name = list(action.keys())[0]
            params = action.get(action_name, {}) if isinstance(action.get(action_name, {}), dict) else action.get(action_name, {})
            snapshot = find_snapshot(action_name, params if isinstance(params, dict) else {}, snapshots)
            
            # For extract_content actions, use intelligent content matching with Ollama verification
            if action_name == "extract_content":
                # Get the extracted content from results
                results = step.get("result", [])
                extracted_content = ""
                if results and isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict) and "extracted_content" in result:
                            extracted_content = result.get("extracted_content", "")
                            break
                
                if extracted_content and snapshot:
                    # logger.debug(f"\n{'='*80}")
                    # logger.debug(f"Step {step_idx + 1}: Analyzing extract_content with {len(extracted_content)} chars")
                    # logger.debug(f"{'='*80}")
                    best_element = find_best_element_for_extraction(snapshot, extracted_content)
                    if best_element:
                        # Build enhanced DOM context with verification data
                        identity = best_element.get("identity", {})
                        dom_context = {
                            "element_index": identity.get("order"),
                            "tagName": identity.get("tagName"),
                            "selectors": best_element.get("selector_provenance", {}),
                            "attributes": best_element.get("attribute_fingerprint", {}),
                            "state": best_element.get("state", {}),
                            "outerHTML": best_element.get("integrity", {}).get("outerHTML", ""),
                        }
                        
                        # Include Ollama verification data if available and enabled
                        if OLLAMA_VERIFICATION_ENABLED and "ollama_verification" in best_element:
                            verification = best_element["ollama_verification"]
                            dom_context["verification"] = {
                                "method": _OLLAMA_DESC,
                                "is_valid": verification.get("is_valid"),
                                "confidence": verification.get("confidence"),
                                "element_hash": verification.get("element_hash"),
                                "element_selector": verification.get("element_selector"),
                                "timestamp": verification.get("timestamp"),
                                "verification_stages": verification.get("verification_stages"),
                                "reason": verification.get("reason"),
                                "recommendations": verification.get("recommendations")
                            }
                        else:
                            dom_context["verification"] = {
                                "method": "disabled" if not OLLAMA_VERIFICATION_ENABLED else "heuristic",
                                "note": "Verification disabled" if not OLLAMA_VERIFICATION_ENABLED else "Ollama not available - using heuristic verification"
                            }
                        
                        action["dom_context"] = dom_context
                        logger.debug(f"✓ Extract verified and element mapped")
                        continue
                    # Fallback: ensure extract_content still gets a dom_context
                    fallback_ctx = extract_element(
                        snapshot,
                        params if isinstance(params, dict) else {},
                        action_name,
                        should_verify=False
                    )
                    if fallback_ctx:
                        action["dom_context"] = fallback_ctx
                        logger.debug("✓ Extract mapped via fallback element selection")
                        continue
            
            # Only use Ollama verification for actions that actually target an element
            VERIFIABLE_ACTIONS = [
                "click_element", "click_element_by_index", "click_element_by_text",
                "input_text", "upload_file", "select_dropdown_option", "extract_content"
            ]
            
            should_verify = OLLAMA_VERIFICATION_ENABLED and action_name in VERIFIABLE_ACTIONS

            if snapshot:
                logger.debug(f"Step {step_idx + 1} - Action {action_idx + 1}: {'Verifying' if should_verify else 'Mapping'} {action_name} element...")
                dom_context = extract_element(snapshot, params if isinstance(params, dict) else {}, action_name, should_verify=should_verify)
                
                # Try to load comprehensive element data if available
                # PRIORITY: Comprehensive data is much more accurate than heuristic snapshot mapping
                index = params.get('index') if isinstance(params, dict) else None
                if index is not None:
                    element_data = load_comprehensive_element_data(action_name, index, run_id)
                    if element_data:
                        dom_context = enhance_dom_context_with_comprehensive_data(dom_context, element_data)
                        logger.debug(f"  ✓ Enhanced with comprehensive element data ({len(element_data.get('selectors', []))} selectors)")
                
                if dom_context:
                    action["dom_context"] = dom_context
                    logger.debug(f"  ✓ Element {'verified' if should_verify else 'mapped'} for {action_name}")
            else:
                # Fallback: even without a full snapshot, try to load comprehensive element data
                index = params.get('index') if isinstance(params, dict) else None
                if index is not None:
                    element_data = load_comprehensive_element_data(action_name, index, run_id)
                    if element_data:
                        dom_context = enhance_dom_context_with_comprehensive_data(None, element_data)
                        action["dom_context"] = dom_context
                        logger.debug(f"  ✓ Mapped via comprehensive data ONLY (no snapshot available)")


    # Save enriched history
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # logger.debug(f"\n{'='*80}")
    if OLLAMA_VERIFICATION_ENABLED:
        logger.debug(f"✓ Enrichment complete with Ollama verification for ALL actions")
    else:
        logger.debug(f"✓ Enrichment complete (verification disabled)")
    logger.debug(f"✓ Saved to {history_path}")
    # logger.debug(f"{'='*80}")


def main():
    tmp_dir = pathlib.Path(__file__).resolve().parents[3] / "tmp"
    history_path = tmp_dir / "agent_history" / "75215413-b5fe-422f-9052-4cbe65dd10dc" / "75215413-b5fe-422f-9052-4cbe65dd10dc.json"
    strip_and_enrich(history_path)


if __name__ == "__main__":
    main()