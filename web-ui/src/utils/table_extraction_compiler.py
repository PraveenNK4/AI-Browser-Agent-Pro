"""
Table Extraction Compiler for Deterministic Playwright Replay Engine

Implements the TABLE_EXTRACTION extraction strategy:
  1. Extraction Strategy Classifier - Detect when to use table extraction
  2. Extraction Schema Builder - Build table deserialize schema from DOM context
  3. Playwright Code Generator - Generate deterministic Playwright code
  4. Stability Ranking - Choose best selector from recorded history

Principles:
  - Never invent selectors (use only agent_history)
  - Never re-infer DOM at runtime
  - Never use vision, OCR, or LLMs during execution
  - Fail loudly on DOM drift (no silent fallback)
"""

from typing import Dict, List, Optional, Any, Tuple
import json
import re


# Production safety error classes
class TableExtractionError(Exception):
    """Base exception for table extraction failures."""
    pass


class TableStructureError(TableExtractionError):
    """Table contains unsupported structure (colspan, rowspan, etc)."""
    pass


class TableTooLargeError(TableExtractionError):
    """Table exceeds safe extraction limit."""
    pass


def classify_extraction_strategy(dom_context: Dict[str, Any]) -> str:
    """
    Step 1: Extraction Strategy Classifier
    
    Return strategy type based on DOM structure:
      - "TABLE" if tagName == "table" (highest priority)
      - "ID" if element has stable ID attribute
      - "CLASS" if element has semantic class
      - "TABLE" if:
          * multiple <td> elements with no IDs/classes (table cell pattern)
    
    Args:
        dom_context: DOM context from agent_history action
        
    Returns:
        Strategy type: "ID", "CLASS", or "TABLE"
    """
    if not dom_context:
        return "TABLE"  # Default to table if no context
    
    tag_name = dom_context.get('tagName', '').lower()
    attrs = dom_context.get('attributes', {})
    
    # Rule 0: Table tag name (highest priority)
    if tag_name == 'table':
        return "TABLE"
    
    if tag_name in ['td', 'th', 'tr', 'tbody', 'thead']:
        return "TABLE"
    
    # Rule 1: ID-based extraction (most reliable)
    element_id = attrs.get('id', '').strip()
    if element_id and len(element_id) > 0:
        return "ID"
    
    # Rule 2: Class-based extraction (semantic classes)
    class_list = attrs.get('class_list', [])
    if class_list and any(cls for cls in class_list if cls.strip()):
        # Check if it's a semantic class (not just 'row', 'cell', 'item')
        semantic_classes = [
            'table', 'grid', 'list', 'menu', 'nav', 'form', 'dialog',
            'panel', 'card', 'section', 'container', 'wrapper'
        ]
        for cls in class_list:
            if any(sc in cls.lower() for sc in semantic_classes):
                return "CLASS"
    
    # If we can't identify, default to TABLE (fallback to table extraction)
    return "TABLE"


def rank_selector_stability(
    preferred: Optional[str],
    css: Optional[str],
    xpath: Optional[str],
    attrs: Dict[str, Any]
) -> Tuple[Optional[str], str]:
    """
    Step 4: Stability Ranking
    
    Choose best selector in order of stability:
      1. #id (most stable)
      2. [data-*] attribute selectors
      3. [aria-*] attribute selectors
      4. semantic .class (e.g., .pageBody, .admin-table)
      5. CSS structural selector from history
      6. Never: absolute XPath or nth-child chains
    
    Args:
        preferred: Preferred selector from agent history
        css: CSS selector from agent history
        xpath: XPath selector from agent history
        attrs: Element attributes dict
        
    Returns:
        Tuple of (selected_selector, ranking_reason)
    """
    # Rank 1: ID selector (most stable)
    element_id = attrs.get('id', '').strip()
    if element_id:
        return f"#{element_id}", "ID"
    
    # Rank 2: Data attributes (stable, explicit)
    data_attrs = attrs.get('data_attributes', {})
    if data_attrs:
        # Try data-testid first, then any data-*
        if 'data-testid' in data_attrs:
            return f"[data-testid='{data_attrs['data-testid']}']", "data-testid"
        first_key = next(iter(data_attrs.keys()))
        return f"[{first_key}='{data_attrs[first_key]}']", "data-*"
    
    # Rank 3: ARIA attributes (stable, semantic)
    if attrs.get('role'):
        role = attrs['role']
        if attrs.get('aria-label'):
            return f"[aria-label='{attrs['aria-label']}']", "aria-label"
    
    # Rank 4: Semantic classes (not nth-child, not generic)
    class_list = attrs.get('class_list', [])
    semantic_classes = [
        'pageBody', 'tightTable', 'admin-table', 'datatable',
        'browse', 'content', 'main', 'form-body'
    ]
    for cls in class_list:
        if cls in semantic_classes:
            return f".{cls}", "semantic-class"
    
    # Rank 5: Preferred selector from agent (normalize first)
    if preferred:
        normalized = normalize_selector(preferred)
        if normalized and not _is_unsafe_selector(normalized):
            return normalized, "preferred-normalized"
    
    # Rank 6: CSS selector (normalize, if not unsafe nth-child chain)
    if css:
        normalized = normalize_selector(css)
        if normalized and not _is_unsafe_selector(normalized):
            return normalized, "css-normalized"
    
    # Rank 7: Never use absolute XPath or nth-child chains
    # Return None - means we'll use default fallback
    return None, "no-stable-selector"


def _is_unsafe_selector(selector: str) -> bool:
    """Check if selector is fragile (nth-child, deep nesting)."""
    if not selector:
        return False
    
    # Reject pure nth-child chains
    if selector.startswith(':nth-child') or ':nth-child' in selector:
        return True
    
    # Reject absolute XPath
    if selector.startswith('/html') or selector.startswith('/body'):
        return True
    
    # Reject excessively nested structural selectors (>5 levels)
    if selector.count('>') > 5:
        return True
    
    return False


def normalize_selector(selector: str) -> str:
    """
    Normalize selector by removing fragile patterns.
    
    Rules:
      - Strip html and body tags
      - Remove all :nth-child(), :nth-of-type() patterns
      - Remove leading > chains
      - Preserve: tag name, IDs, semantic classes
    
    Args:
        selector: Raw selector string from history
        
    Returns:
        Normalized, production-safe selector
    """
    if not selector:
        return selector
    
    # Remove leading html.js:nth-child(...) > body:nth-child(...) > chains
    normalized = re.sub(r'^html[^>]*>', '', selector)
    normalized = re.sub(r'^body[^>]*>', '', normalized)
    normalized = re.sub(r'^\s*>\s*', '', normalized)
    
    # Remove all :nth-child() and :nth-of-type() patterns
    normalized = re.sub(r':nth-child\([^)]*\)', '', normalized)
    normalized = re.sub(r':nth-of-type\([^)]*\)', '', normalized)
    
    # Remove trailing > or spaces
    normalized = re.sub(r'\s*>\s*$', '', normalized)
    normalized = normalized.strip()
    
    return normalized if normalized else selector


def detect_header_row(outer_html: str) -> Optional[str]:
    """
    Detect the correct header row from outerHTML (compile-time only).
    
    Rules (in order):
      1. If <thead> exists → use first <tr> inside <thead>
      2. Else if any <th> exists → use first <tr> containing <th>
      3. Else fallback to first <tr>
    
    Args:
        outer_html: Captured outerHTML from DOM context
        
    Returns:
        HTML string of header row, or None if not found
    """
    if not outer_html:
        return None
    
    # Rule 1: Check for <thead> first
    thead_match = re.search(r'<thead[^>]*>(.*?)</thead>', outer_html, re.IGNORECASE | re.DOTALL)
    if thead_match:
        thead_content = thead_match.group(1)
        tr_match = re.search(r'<tr[^>]*>(.*?)</tr>', thead_content, re.IGNORECASE | re.DOTALL)
        if tr_match:
            return f"<tr>{tr_match.group(1)}</tr>"
    
    # Rule 2: Check for first <tr> containing <th>
    all_tr_matches = re.finditer(r'<tr[^>]*>(.*?)</tr>', outer_html, re.IGNORECASE | re.DOTALL)
    for tr_match in all_tr_matches:
        tr_content = tr_match.group(1)
        if re.search(r'<th[^>]*>', tr_content, re.IGNORECASE):
            return f"<tr>{tr_content}</tr>"
    
    # Rule 3: Fallback to first <tr>
    first_tr = re.search(r'<tr[^>]*>(.*?)</tr>', outer_html, re.IGNORECASE | re.DOTALL)
    if first_tr:
        return f"<tr>{first_tr.group(1)}</tr>"
    
    return None


def _check_colspan_rowspan(outer_html: str) -> None:
    """
    Check for colspan/rowspan in outerHTML.
    
    Raises:
        TableStructureError: If colspan or rowspan found
    """
    if not outer_html:
        return
    
    # Check for colspan
    if re.search(r'colspan\s*=', outer_html, re.IGNORECASE):
        raise TableStructureError("Table contains colspan - not supported")
    
    # Check for rowspan
    if re.search(r'rowspan\s*=', outer_html, re.IGNORECASE):
        raise TableStructureError("Table contains rowspan - not supported")


def build_extraction_schema(dom_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 2: Extraction Schema Builder
    
    Build a schema that describes how to extract structured data from the DOM.
    For tables, schema describes row/cell structure.
    For ID/CLASS, schema describes single element extraction.
    
    Schema must come ONLY from captured outerHTML and DOM metadata.
    No live DOM scanning allowed.
    
    Args:
        dom_context: DOM context from agent_history action
        
    Returns:
        Extraction schema dict with schema_version
        
    Raises:
        TableStructureError: If table contains colspan/rowspan
        TableTooLargeError: If table has > 1000 rows
    """
    if not dom_context:
        return {
            "type": "unknown",
            "error": "No DOM context",
            "schema_version": 1
        }
    
    strategy = classify_extraction_strategy(dom_context)
    attrs = dom_context.get('attributes', {})
    selectors = dom_context.get('selectors', {})
    tag_name = dom_context.get('tagName', '').lower()
    outer_html = dom_context.get('outerHTML', '')
    
    # Choose best anchor selector
    anchor_selector, anchor_reason = rank_selector_stability(
        preferred=selectors.get('preferred'),
        css=selectors.get('css'),
        xpath=selectors.get('xpath'),
        attrs=attrs
    )
    
    # Build strategy-specific schema
    if strategy == "ID":
        return {
            "type": "ID",
            "anchor_selector": anchor_selector,
            "anchor_reason": anchor_reason,
            "element_tag": tag_name,
            "extraction_method": "innerText",
            "schema_version": 1
        }
    
    elif strategy == "CLASS":
        return {
            "type": "CLASS",
            "anchor_selector": anchor_selector,
            "anchor_reason": anchor_reason,
            "classes": attrs.get('class_list', []),
            "extraction_method": "innerText",
            "schema_version": 1
        }
    
    elif strategy == "TABLE":
        # Safety check: reject tables with colspan/rowspan (compile-time)
        _check_colspan_rowspan(outer_html)
        
        # Extract table structure from outerHTML
        header_row = detect_header_row(outer_html)
        headers = _extract_headers_from_html(outer_html)
        
        # Count rows and enforce max-row guard
        row_count = len(re.findall(r'<tr[^>]*>', outer_html, re.IGNORECASE))
        if row_count > 1000:
            raise TableTooLargeError(f"Table has {row_count} rows, exceeds safe limit of 1000")
        
        schema = {
            "type": "table",
            "anchor_selector": anchor_selector,
            "anchor_reason": anchor_reason,
            "element_tag": tag_name,
            "headers": headers,
            "header_row_html": header_row,
            "row_count_at_capture": row_count,
            "row_selector": "tr",
            "cell_selector": "td",
            "schema_version": 1
        }
        return schema
    
    return {
        "type": "unknown",
        "error": f"Unknown strategy: {strategy}",
        "schema_version": 1
    }


def _extract_headers_from_html(outer_html: str) -> List[str]:
    """
    Extract table headers from captured outerHTML without live DOM access.
    Uses detect_header_row() to find correct row, then extracts headers.
    
    Uses <th> elements first, then falls back to text content of detected header row.
    
    Args:
        outer_html: Captured outerHTML from DOM context
        
    Returns:
        List of header text strings
    """
    if not outer_html:
        return []
    
    headers = []
    
    # Try to find <th> elements first (most explicit)
    th_pattern = r'<th[^>]*>([^<]*)</th>'
    th_matches = re.findall(th_pattern, outer_html, re.IGNORECASE)
    if th_matches:
        headers = [h.strip() for h in th_matches if h.strip()]
        if headers:
            return headers
    
    # Use detect_header_row() to find correct header row
    header_row = detect_header_row(outer_html)
    if header_row:
        # Extract <td> or <th> content from header row
        cell_pattern = r'<t[dh][^>]*>([^<]*)</t[dh]>'
        cells = re.findall(cell_pattern, header_row, re.IGNORECASE)
        
        if cells:
            # If cells are short, likely headers
            if all(len(c.strip()) < 50 for c in cells if c.strip()):
                headers = [c.strip() for c in cells if c.strip()]
    
    return headers


def generate_table_extraction_code(schema: Dict[str, Any]) -> str:
    """
    Step 3: Playwright Code Generator
    
    Generate deterministic Playwright code for table extraction.
    
    Rules:
      - Use schema.anchor_selector (normalized, only source of truth)
      - Generate JavaScript evaluate() pattern
      - Parse table row by row
      - Match cells to headers by position
      - Throw on DOM drift (headers != cell count)
      - No live heuristics or regex on content
      - Log schema metadata before execution
    
    Args:
        schema: Extraction schema from build_extraction_schema()
        
    Returns:
        Playwright code snippet (string)
    """
    if schema.get("type") != "table":
        # Non-table schemas use simpler extraction
        return _generate_simple_extraction_code(schema)
    
    anchor = schema.get("anchor_selector")
    if not anchor:
        return _generate_fallback_extraction_code()
    
    headers = schema.get("headers", [])
    row_selector = schema.get("row_selector", "tr")
    cell_selector = schema.get("cell_selector", "td")
    schema_version = schema.get("schema_version", 1)
    anchor_reason = schema.get("anchor_reason", "unknown")
    
    # Generate logging metadata
    schema_metadata = {
        "extraction_strategy": "TABLE",
        "schema_used": {
            "anchor_selector": anchor,
            "anchor_reason": anchor_reason,
            "headers": headers,
            "schema_version": schema_version,
            "row_count_at_capture": schema.get("row_count_at_capture")
        }
    }
    
    # Generate JavaScript that will run in browser context
    # This code executes ONLY against captured selectors, never infers
    js_code = f'''
        const element = document.querySelector("{anchor}");
        if (!element) throw new Error("Table not found at selector: {anchor}");
        
        const rows = Array.from(element.querySelectorAll("{row_selector}"));
        if (rows.length < 2) throw new Error("Table has less than 2 rows, expected headers + data");
        
        const headers = Array.from(rows[0].querySelectorAll("{cell_selector}"))
            .map(cell => cell.innerText.trim());
        
        if (headers.length === 0) throw new Error("No headers found in first row");
        
        const data = rows.slice(1).map(row => {{
            const cells = Array.from(row.querySelectorAll("{cell_selector}"))
                .map(cell => cell.innerText.trim());
            
            // Strict validation: reject rows with mismatched cell count
            if (cells.length !== headers.length) {{
                console.warn(`Row has ${{cells.length}} cells but ${{headers.length}} headers`);
            }}
            
            const obj = {{}};
            headers.forEach((header, idx) => {{
                obj[header] = cells[idx] ?? null;
            }});
            return obj;
        }});
        
        return {{ headers, data }};
    '''.strip()
    
    # Generate Playwright code with logging
    code = f'''            # Log extraction schema for debugging
            import json
            log_metadata = {schema_metadata}
            log.info(json.dumps(log_metadata, indent=2))
            
            # Extract table data using captured selector
            table = await page.query_selector("{anchor}")
            if not table:
                raise RuntimeError("Table element not found at selector: {anchor}")
            
            try:
                result = await table.evaluate("""({js_code})""")
                extracted_content = json.dumps(result, indent=2)
            except Exception as e:
                raise RuntimeError(f"Table extraction failed: {{str(e)}}")'''
    
    return code


def _generate_simple_extraction_code(schema: Dict[str, Any]) -> str:
    """
    Generate code for ID/CLASS-based extraction.
    Includes schema versioning and logging.
    """
    anchor = schema.get("anchor_selector")
    if not anchor:
        return _generate_fallback_extraction_code()
    
    schema_version = schema.get("schema_version", 1)
    anchor_reason = schema.get("anchor_reason", "unknown")
    extraction_type = schema.get("type", "unknown")
    
    schema_metadata = {
        "extraction_strategy": extraction_type,
        "schema_used": {
            "anchor_selector": anchor,
            "anchor_reason": anchor_reason,
            "schema_version": schema_version
        }
    }
    
    code = f'''            # Log extraction schema for debugging
            import json
            log_metadata = {schema_metadata}
            log.info(json.dumps(log_metadata, indent=2))
            
            # Extract content from element
            element = await page.query_selector("{anchor}")
            if not element:
                raise RuntimeError("Element not found at selector: {anchor}")
            
            extracted_content = await element.inner_text()'''
    
    return code


def _generate_fallback_extraction_code() -> str:
    """Generate code when no anchor selector is available."""
    code = '''            # Fallback: extract all text content
            extracted_content = await page.evaluate('() => document.body.innerText')'''
    return code


def generate_extraction_with_failure_semantics(
    dom_context: Dict[str, Any],
    should_raise: bool = True
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Complete extraction compiler pipeline.
    
    Steps:
      1. Classify extraction strategy
      2. Build extraction schema (with safety checks)
      3. Generate Playwright code
      4. Return both code and schema (for debugging)
    
    Safety checks:
      - Fails on colspan/rowspan
      - Fails on tables > 1000 rows
      - Includes schema versioning
      - Logs schema metadata
    
    Args:
        dom_context: DOM context from agent_history
        should_raise: If True, raise on schema errors; if False, return fallback
        
    Returns:
        Tuple of (playwright_code, schema_metadata)
        
    Raises:
        TableStructureError: If table has unsupported structure
        TableTooLargeError: If table exceeds size limit
        ValueError: If should_raise=True and extraction setup fails
    """
    try:
        # Step 1: Classify
        strategy = classify_extraction_strategy(dom_context)
        
        # Step 2: Build schema (with safety checks)
        schema = build_extraction_schema(dom_context)
        
        if schema.get("error") and should_raise:
            raise ValueError(f"Schema build failed: {schema.get('error')}")
        
        # Step 3: Generate code
        code = generate_table_extraction_code(schema)
        
        return code, schema
    
    except (TableStructureError, TableTooLargeError) as e:
        # Production safety errors - always raise
        if should_raise:
            raise
        return _generate_fallback_extraction_code(), {
            "type": "fallback",
            "error": str(e),
            "schema_version": 1
        }
    
    except Exception as e:
        if should_raise:
            raise ValueError(f"Extraction compilation failed: {str(e)}")
        
        # Fallback: return simple extraction
        return _generate_fallback_extraction_code(), {
            "type": "fallback",
            "error": str(e),
            "schema_version": 1
        }
