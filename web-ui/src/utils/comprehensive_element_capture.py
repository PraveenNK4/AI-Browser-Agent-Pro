"""
Comprehensive Element Data Capture

Captures ALL available selector data from an element including:
- IDs, classes, data attributes, aria attributes
- Multiple selector types ranked by stability
- Visual state, bounding box, computed styles
- Text content, accessibility labels
- Parent/ancestor context for structural selectors
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from playwright.async_api import Page, ElementHandle
import logging
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class ComprehensiveElementCapture:
    """Captures complete element data for robust selector generation."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Args:
            output_dir: Directory to save comprehensive element JSONs
        """
        self.output_dir = output_dir or Path("tmp/element_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def capture_element_data(
        self, 
        page: Page, 
        element: ElementHandle,
        index: Optional[int] = None,
        action_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Capture comprehensive element data including all possible selectors.
        
        Args:
            page: Playwright page object
            element: Element to capture
            index: Element index (if available)
            action_context: Context of the action (e.g., "click", "input")
        
        Returns:
            Comprehensive element data dictionary
        """
        
        # Execute JavaScript to extract all element data
        element_data = await page.evaluate("""
            (el) => {
                if (!el) return null;
                
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                
                // Extract ALL attributes
                const attributes = {};
                for (let attr of el.attributes) {
                    attributes[attr.name] = attr.value;
                }
                
                // Extract ALL classes
                const classes = Array.from(el.classList);
                
                // Extract ALL data attributes
                const dataAttributes = {};
                for (let key in el.dataset) {
                    dataAttributes[key] = el.dataset[key];
                }
                
                // Extract ALL aria attributes
                const ariaAttributes = {};
                for (let attr of el.attributes) {
                    if (attr.name.startsWith('aria-')) {
                        ariaAttributes[attr.name] = attr.value;
                    }
                }
                
                // Get all possible IDs (element and parents)
                const ancestorIds = [];
                let current = el;
                let depth = 0;
                while (current && depth < 5) {
                    if (current.id) {
                        ancestorIds.push({
                            depth: depth,
                            id: current.id,
                            tag: current.tagName.toLowerCase()
                        });
                    }
                    current = current.parentElement;
                    depth++;
                }
                
                // Get stable parent structure
                const parentChain = [];
                current = el.parentElement;
                depth = 1;
                while (current && depth <= 3) {
                    parentChain.push({
                        depth: depth,
                        tag: current.tagName.toLowerCase(),
                        id: current.id || null,
                        classes: Array.from(current.classList),
                        role: current.getAttribute('role')
                    });
                    current = current.parentElement;
                    depth++;
                }
                
                // Generate multiple selector candidates
                const selectorCandidates = [];
                
                // 1. ID-based selectors
                if (el.id) {
                    selectorCandidates.push({
                        type: 'id',
                        selector: `#${CSS.escape(el.id)}`,
                        stability: 100,
                        description: 'Direct ID selector'
                    });
                }
                
                // 2. Semantic class selectors
                const semanticClasses = classes.filter(c => 
                    c.includes('btn') || c.includes('button') || 
                    c.includes('link') || c.includes('nav') ||
                    c.includes('submit') || c.includes('login') ||
                    c.includes('save') || c.includes('cancel')
                );
                
                if (semanticClasses.length > 0) {
                    selectorCandidates.push({
                        type: 'semantic-class',
                        selector: el.tagName.toLowerCase() + '.' + semanticClasses.map(c => CSS.escape(c)).join('.'),
                        stability: 90,
                        description: 'Semantic class selector'
                    });
                }
                
                // 3. Data attribute selectors
                for (let key in dataAttributes) {
                    selectorCandidates.push({
                        type: 'data-attribute',
                        selector: `${el.tagName.toLowerCase()}[data-${key}="${CSS.escape(dataAttributes[key])}"]`,
                        stability: 80,
                        description: `Data attribute: data-${key}`
                    });
                }
                
                // 4. Aria attribute selectors
                for (let attr in ariaAttributes) {
                    selectorCandidates.push({
                        type: 'aria-attribute',
                        selector: `${el.tagName.toLowerCase()}[${attr}="${CSS.escape(ariaAttributes[attr])}"]`,
                        stability: 80,
                        description: `Aria attribute: ${attr}`
                    });
                }
                
                // 5. Name attribute (for inputs)
                if (attributes.name) {
                    selectorCandidates.push({
                        type: 'name-attribute',
                        selector: `${el.tagName.toLowerCase()}[name="${CSS.escape(attributes.name)}"]`,
                        stability: 85,
                        description: 'Name attribute'
                    });
                }
                
                // 6. Type + Name combo (for inputs)
                if (attributes.type && attributes.name) {
                    selectorCandidates.push({
                        type: 'type-name-combo',
                        selector: `${el.tagName.toLowerCase()}[type="${attributes.type}"][name="${CSS.escape(attributes.name)}"]`,
                        stability: 88,
                        description: 'Type and name combination'
                    });
                }
                
                // 7. Role-based selector
                if (attributes.role) {
                    selectorCandidates.push({
                        type: 'role',
                        selector: `[role="${attributes.role}"]`,
                        stability: 75,
                        description: 'Role attribute'
                    });
                }
                
                // 8. Text-based selector (if meaningful text)
                const innerText = (el.innerText || el.textContent || '').trim();
                if (innerText && innerText.length > 2 && innerText.length < 100) {
                    const escapedText = innerText.replace(/"/g, '\\\\"').substring(0, 50);
                    selectorCandidates.push({
                        type: 'text',
                        selector: `${el.tagName.toLowerCase()}:has-text("${escapedText}")`,
                        stability: 70,
                        description: 'Text content'
                    });
                }
                
                // 9. Parent ID + child structure (no nth-child)
                if (ancestorIds.length > 0) {
                    const parentId = ancestorIds[0];
                    const tag = el.tagName.toLowerCase();
                    selectorCandidates.push({
                        type: 'parent-id-child',
                        selector: `#${CSS.escape(parentId.id)} ${tag}`,
                        stability: 65,
                        description: `Child of parent ID: ${parentId.id}`
                    });
                }
                
                // 10. XPath (for reference)
                const getXPath = (el) => {
                    if (el.id) return `//*[@id="${el.id}"]`;
                    if (el === document.body) return '/html/body';
                    
                    let ix = 0;
                    const siblings = el.parentNode ? el.parentNode.childNodes : [];
                    for (let i = 0; i < siblings.length; i++) {
                        const sibling = siblings[i];
                        if (sibling === el) {
                            const parent = el.parentNode;
                            const parentPath = parent ? getXPath(parent) : '';
                            return `${parentPath}/${el.tagName.toLowerCase()}[${ix + 1}]`;
                        }
                        if (sibling.nodeType === 1 && sibling.tagName === el.tagName) {
                            ix++;
                        }
                    }
                    return '';
                };
                
                const xpath = getXPath(el);
                selectorCandidates.push({
                    type: 'xpath',
                    selector: xpath,
                    stability: 50,
                    description: 'XPath selector'
                });
                
                // 11. Browser-generated CSS path (last resort - may contain nth-child)
                const getCSSPath = (el) => {
                    if (el.id) return `#${CSS.escape(el.id)}`;
                    if (el === document.body) return 'body';
                    
                    const names = [];
                    while (el.parentElement && el !== document.body) {
                        if (el.id) {
                            names.unshift(`#${CSS.escape(el.id)}`);
                            break;
                        } else {
                            let c = 1, e = el;
                            while (e.previousElementSibling) {
                                e = e.previousElementSibling;
                                c++;
                            }
                            names.unshift(`${el.tagName.toLowerCase()}:nth-child(${c})`);
                            el = el.parentElement;
                        }
                    }
                    return names.join(' > ');
                };
                
                const cssPath = getCSSPath(el);
                selectorCandidates.push({
                    type: 'css-path',
                    selector: cssPath,
                    stability: 10,
                    description: 'Browser-generated CSS path (brittle - contains nth-child)'
                });
                
                // Sort candidates by stability (descending)
                selectorCandidates.sort((a, b) => b.stability - a.stability);
                
                return {
                    tagName: el.tagName,
                    textContent: innerText,
                    innerText: innerText,
                    innerHTML: el.innerHTML.substring(0, 500), // Limit size
                    outerHTML: el.outerHTML.substring(0, 1000), // Limit size
                    
                    // All attributes
                    attributes: attributes,
                    classes: classes,
                    dataAttributes: dataAttributes,
                    ariaAttributes: ariaAttributes,
                    
                    // Ancestor context
                    ancestorIds: ancestorIds,
                    parentChain: parentChain,
                    
                    // Visual state
                    state: {
                        visible: !!(rect.width && rect.height),
                        enabled: !el.disabled,
                        focused: el === document.activeElement,
                        checked: el.checked || false,
                        selected: el.selected || false
                    },
                    
                    // Bounding box
                    boundingBox: {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                        top: rect.top,
                        left: rect.left,
                        bottom: rect.bottom,
                        right: rect.right
                    },
                    
                    // Computed styles (key ones)
                    computedStyle: {
                        display: style.display,
                        visibility: style.visibility,
                        opacity: style.opacity,
                        position: style.position,
                        color: style.color,
                        backgroundColor: style.backgroundColor,
                        fontSize: style.fontSize,
                        fontWeight: style.fontWeight
                    },
                    
                    // ALL selector candidates ranked by stability
                    selectors: selectorCandidates,
                    
                    // Best selector (highest stability, no nth-child if possible)
                    recommendedSelector: selectorCandidates.find(s => s.stability > 50) || selectorCandidates[0]
                };
            }
        """, element)
        
        if not element_data:
            logger.warning("Failed to capture element data")
            return {}
        
        # Add metadata
        element_data['metadata'] = {
            'timestamp': datetime.utcnow().isoformat(),
            'page_url': page.url,
            'page_title': await page.title(),
            'element_index': index,
            'action_context': action_context,
            'capture_method': 'comprehensive'
        }
        
        # Generate element hash for tracking
        element_hash = self._generate_element_hash(element_data)
        element_data['elementHash'] = element_hash
        
        return element_data
    
    def _generate_element_hash(self, element_data: Dict[str, Any]) -> str:
        """Generate a unique hash for the element based on stable attributes."""
        # Use stable attributes for hashing
        hash_str = f"{element_data.get('tagName', '')}_" \
                   f"{element_data.get('attributes', {}).get('id', '')}_" \
                   f"{element_data.get('attributes', {}).get('name', '')}_" \
                   f"{element_data.get('textContent', '')[:50]}"
        
        return hashlib.md5(hash_str.encode()).hexdigest()[:12]
    
    async def save_element_data(
        self, 
        element_data: Dict[str, Any],
        filename: Optional[str] = None
    ) -> Path:
        """
        Save element data to JSON file.
        
        Args:
            element_data: Comprehensive element data
            filename: Optional custom filename
        
        Returns:
            Path to saved JSON file
        """
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            element_hash = element_data.get('elementHash', 'unknown')
            filename = f"element_{timestamp}_{element_hash}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(element_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved comprehensive element data to: {filepath}")
        return filepath
    
    def print_selector_summary(self, element_data: Dict[str, Any]):
        """Print a human-readable summary of available selectors."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ELEMENT SELECTOR SUMMARY")
        print("=" * 80)
        
        print(f"\n📌 Element: {element_data.get('tagName', 'unknown')}")
        print(f"📝 Text: {element_data.get('textContent', '')[:60]}")
        print(f"🔖 Hash: {element_data.get('elementHash', 'N/A')}")
        
        print("\n🎯 RECOMMENDED SELECTOR:")
        recommended = element_data.get('recommendedSelector', {})
        print(f"  Type: {recommended.get('type', 'N/A')}")
        print(f"  Selector: {recommended.get('selector', 'N/A')}")
        print(f"  Stability: {recommended.get('stability', 0)}/100")
        print(f"  Note: {recommended.get('description', 'N/A')}")
        
        print("\n📋 ALL AVAILABLE SELECTORS (ranked by stability):")
        selectors = element_data.get('selectors', [])
        for i, sel in enumerate(selectors[:10], 1):  # Top 10
            stability_emoji = "🟢" if sel['stability'] >= 80 else "🟡" if sel['stability'] >= 50 else "🔴"
            print(f"  {i}. {stability_emoji} [{sel['stability']:3d}] {sel['type']:20s} → {sel['selector'][:80]}")
        
        if len(selectors) > 10:
            print(f"  ... and {len(selectors) - 10} more selectors")
        
        print("\n🏷️  ATTRIBUTES:")
        attrs = element_data.get('attributes', {})
        for key, value in list(attrs.items())[:10]:
            print(f"  {key}: {str(value)[:60]}")
        
        print("\n📊 STATE:")
        state = element_data.get('state', {})
        for key, value in state.items():
            emoji = "✅" if value else "❌"
            print(f"  {emoji} {key}: {value}")
        
        print("\n" + "=" * 80)


# Helper function for easy integration
async def capture_and_save_element(
    page: Page,
    element: ElementHandle,
    index: Optional[int] = None,
    action_context: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function to capture and save element data in one call.
    
    Usage:
        element_data = await capture_and_save_element(
            page, 
            element, 
            index=5, 
            action_context="click_button"
        )
    """
    capturer = ComprehensiveElementCapture(output_dir)
    element_data = await capturer.capture_element_data(page, element, index, action_context)
    await capturer.save_element_data(element_data)
    capturer.print_selector_summary(element_data)
    return element_data
