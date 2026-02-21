"""
Ollama Max Accuracy Verifier for DOM Element Selection
========================================================

4-Stage Verification Pipeline using Ollama (qwen2.5:32b)
- Stage 1: Structure Validation (75%+ confidence required)
- Stage 2: Content Matching (75%+ confidence required)
- Stage 3: Step-Specific Validation (75%+ confidence required)
- Stage 4: Cross-Validation with Neighbors (75%+ confidence required)

Final Confidence = MIN(all stages)
Element is valid if MIN >= 90%

Temperature: 0.1 (deterministic, optimal for accuracy)
Timeout: 60 seconds per verification
Model: qwen2.5:32b
"""

import requests
import json
import re
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Global singleton instance to avoid redundant Ollama availability checks
_verifier_instance = None


class OllamaMaxAccuracyVerifier:
    """
    Verifies DOM elements using Ollama's qwen2.5:32b model
    with 4-stage verification pipeline for maximum accuracy.
    """
    
    def __init__(self, skip_availability_check=False):
        self.model = "qwen2.5:14b"  # Faster model for testing/debugging - 14 billion parameters
        self.temperature = 0.1
        self.top_p = 0.95
        self.timeout = 180  # 3 minutes - faster verification for testing
        self.min_confidence = 90
        self.ollama_url = "http://localhost:11434/api/generate"
        self.stage_threshold = 75  # Each stage must have 75%+ confidence
        
        # Test Ollama availability only once
        if not skip_availability_check:
            self._test_ollama_availability()
    
    def _test_ollama_availability(self):
        """Test if Ollama is available at localhost:11434"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                logger.debug("✓ Ollama is available at localhost:11434")
                return True
        except Exception as e:
            logger.warning(f"⚠ Ollama not available: {e}")
            return False
    
    def verify_element_for_step(self, step_name, action_name, element_text, 
                                 html_context, selector, extraction_requirement, 
                                 keywords=None):
        """
        Main verification method using a unified 4-stage pipeline in a SINGLE query.
        """
        try:
            # 🚀 Unified Audit Strategy: Combine all 4 validation stages into one LLM call
            # This is 4x faster and prevents timeouts.
            keyword_text = ", ".join(keywords) if keywords else "None"
            
            prompt = f"""AUDIT ACTION: {action_name} for STEP: {step_name}
REQUIREMENT: {extraction_requirement}
KEYWORDS: {keyword_text}

ELEMENT DATA:
- Tag: {selector}
- Text: {element_text[:500]}
- HTML Snippet: {html_context[:800]}

UNIFIED VERIFICATION TASK:
Perform a 4-stage audit and provide confidence scores (0-100) for each:
1. STRUCTURE: Is the HTML tag and snippet suitable for {action_name}?
2. CONTENT: Does the element text/html actually contain the data for '{extraction_requirement}'?
3. RELEVANCE: Is this the correct specific element for this '{step_name}'?
4. BEST_MATCH: Based on the snippets, is this likely the primary element (not just a small child or wrapper)?

RESPONSE FORMAT: 
Respond with ONLY a JSON object: 
{{
  "structure": score, 
  "content": score, 
  "relevance": score, 
  "best_match": score, 
  "reason": "short explanation"
}}"""

            # Single unified query
            response_json = self._query_ollama_unified(prompt)
            
            # Extract scores
            s1 = response_json.get("structure", 0)
            s2 = response_json.get("content", 0)
            s3 = response_json.get("relevance", 0)
            s4 = response_json.get("best_match", 0)
            
            # Final confidence is the MIN (conservative)
            final_confidence = min(s1, s2, s3, s4)
            is_valid = final_confidence >= self.min_confidence
            
            result = {
                "is_valid": is_valid,
                "confidence": round(final_confidence, 1),
                "element_hash": self._generate_element_hash(selector, element_text),
                "verification_stages": {
                    "structure_validation": {"confidence": s1, "passed": s1 >= self.stage_threshold},
                    "content_matching": {"confidence": s2, "passed": s2 >= self.stage_threshold},
                    "step_specific": {"confidence": s3, "passed": s3 >= self.stage_threshold},
                    "cross_validation": {"confidence": s4, "passed": s4 >= self.stage_threshold}
                },
                "reason": response_json.get("reason", "No reason provided"),
                "method": f"Ollama ({self.model} | Unified Audit)",
                "timestamp": datetime.now().isoformat()
            }
            
            return is_valid, result
            
        except Exception as e:
            logger.error(f"✗ Unified Verification error: {e}")
            return False, {"is_valid": False, "confidence": 0, "reason": str(e)}

    def _query_ollama_unified(self, prompt):
        """Query Ollama and return the parsed JSON object."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": 0.1,
                "stream": False,
                "format": "json" # Force JSON output if the model supports it
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
            
            if response.status_code == 200:
                text = response.json().get("response", "{}")
                # Clean up any markdown blocks if the model ignored "format: json"
                text = re.sub(r'```json\n?|\n?```', '', text).strip()
                return json.loads(text)
            return {}
        except Exception as e:
            logger.warning(f"Ollama query failed: {e}")
            return {}
    
    def _validate_structure(self, html_context, action_name):
        """Stage 1: Validate HTML structure is suitable for the action"""
        
        prompt = f"""Analyze the HTML structure for {action_name} action.

HTML: {html_context[:500]}

Questions:
1. Is the HTML structure valid and well-formed for {action_name}?
2. Does the element have required attributes (id, name, class, role)?
3. Is the element in the expected HTML hierarchy?

Respond with JSON: {{"confidence": 0-100, "reason": "brief reason"}}"""
        
        confidence = self._query_ollama(prompt)
        return confidence
    
    def _verify_content_match(self, element_text, extraction_requirement, keywords):
        """Stage 2: Verify element content matches extraction requirement"""
        
        keyword_text = ", ".join(keywords) if keywords else "None provided"
        
        prompt = f"""Does the element content match the extraction requirement?

Element text: {element_text[:300]}
Required to extract: {extraction_requirement}
Related keywords: {keyword_text}

Respond with JSON: {{"confidence": 0-100, "reason": "brief reason"}}"""
        
        confidence = self._query_ollama(prompt)
        return confidence
    
    def _verify_step_specific(self, step_name, action_name, element_text, extraction_requirement):
        """Stage 3: Verify element is appropriate for the specific step"""
        
        prompt = f"""Is this element appropriate for this step action?

Step: {step_name}
Action: {action_name}
Element text: {element_text[:200]}
Requirement: {extraction_requirement}

Is this the right element to interact with? Respond with JSON: {{"confidence": 0-100, "reason": "brief reason"}}"""
        
        confidence = self._query_ollama(prompt)
        return confidence
    
    def _cross_validate_with_neighbors(self, html_context, element_text, extraction_requirement):
        """Stage 4: Cross-validate this is the best element among alternatives"""
        
        prompt = f"""Is this the BEST element for the extraction requirement?

Element text: {element_text[:200]}
Context: {html_context[:300]}
Requirement: {extraction_requirement}

Rate confidence this is the best element among alternatives. Respond with JSON: {{"confidence": 0-100, "reason": "brief reason"}}"""
        
        confidence = self._query_ollama(prompt)
        return confidence
    
    def _query_ollama(self, prompt):
        """Query Ollama and extract confidence score"""
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": False
            }
            
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                confidence = self._parse_ollama_response(result.get("response", ""))
                return confidence
            else:
                logger.warning(f"Ollama returned status {response.status_code}")
                return 0
                
        except requests.exceptions.Timeout:
            logger.warning(f"Ollama timeout after {self.timeout} seconds")
            return 0
        except requests.exceptions.ConnectionError:
            logger.warning("Cannot connect to Ollama at localhost:11434")
            return 0
        except Exception as e:
            logger.warning(f"Query error: {e}")
            return 0
    
    def _parse_ollama_response(self, response_text):
        """Extract confidence score from Ollama response"""
        
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                confidence = data.get("confidence", 0)
                if isinstance(confidence, (int, float)):
                    return min(100, max(0, confidence))
            return 0
        except:
            return 0
    
    def _generate_element_hash(self, selector, element_text):
        """Generate unique hash for element tracking"""
        
        combined = f"{selector}:{element_text[:100]}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


def verify_step_element(step_name, element_text, element_html, element_selector,
                       extraction_requirement, expected_keywords=None, 
                       nearby_elements=None, step_action=None, **kwargs):
    """
    High-level function to verify a step's element (singleton pattern)
    
    Args:
        step_name: Name of step (e.g., "Extract Content")
        element_text: Text content of element
        element_html: HTML of element
        element_selector: CSS/XPath selector
        extraction_requirement: What should be extracted
        expected_keywords: Keywords related to extraction
        nearby_elements: Alternative elements (for cross-validation)
        step_action: Type of action (extract_table_data, etc)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        (is_valid, verification_result)
    """
    global _verifier_instance
    
    # Create singleton instance only once (on first call)
    if _verifier_instance is None:
        _verifier_instance = OllamaMaxAccuracyVerifier(skip_availability_check=False)
    
    # Reuse same instance for all subsequent calls (no redundant checks)
    verifier = _verifier_instance
    
    return verifier.verify_element_for_step(
        step_name=step_name, 
        action_name=step_action or "extract_content",
        element_text=element_text, 
        html_context=element_html,
        selector=element_selector, 
        extraction_requirement=extraction_requirement, 
        keywords=expected_keywords
    )
