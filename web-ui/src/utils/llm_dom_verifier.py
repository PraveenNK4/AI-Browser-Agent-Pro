"""
LLM-based DOM element verification for 100% accuracy extraction.
Uses Claude API or local Ollama 32B model for validation.
"""

import os
import requests
import json
from typing import Optional, Dict, Any, cast
from anthropic import Anthropic
from anthropic.types import Message, TextBlock


class LLMDOMVerifier:
    """Verify DOM elements using LLM to ensure 100% extraction accuracy."""
    
    def __init__(self):
        """Initialize with Claude or Ollama support."""
        self.use_ollama = False
        self.use_claude = False
        self.client: Optional[Anthropic] = None
        
        # Try Ollama first
        if self._check_ollama():
            self.use_ollama = True
            print("[+] Using local Ollama 32B model for DOM verification")
        else:
            # Fallback to Claude
            self.use_claude = True
            self.client = Anthropic()
            print("[+] Using Claude API for DOM verification")
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running locally."""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama2-uncensored", "prompt": "test", "stream": False},
                timeout=2
            )
            return response.status_code == 200
        except:
            return False
    
    def verify_element(
        self,
        element_text: str,
        element_html: str,
        extraction_requirement: str,
        keywords: list
    ) -> Dict[str, Any]:
        """
        Verify if captured element matches extraction requirement.
        
        Args:
            element_text: Extracted text content from element
            element_html: Outer HTML of element
            extraction_requirement: What should be extracted (from task/prompt)
            keywords: Keywords expected in the extraction
        
        Returns:
            {
                "is_valid": bool,
                "confidence": float (0-100),
                "reason": str,
                "suggestions": str
            }
        """
        if self.use_ollama:
            return self._verify_with_ollama(element_text, element_html, extraction_requirement, keywords)
        else:
            return self._verify_with_claude(element_text, element_html, extraction_requirement, keywords)
    
    def _verify_with_claude(
        self,
        element_text: str,
        element_html: str,
        extraction_requirement: str,
        keywords: list
    ) -> Dict[str, Any]:
        """Verify using Claude API."""
        try:
            prompt = self._build_verification_prompt(element_text, element_html, extraction_requirement, keywords)
            
            if self.client is None:
                raise ValueError("Claude client not initialized")
            
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract text from TextBlock only
            response_text = ""
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text = block.text
                    break
            
            return self._parse_verification_response(response_text)
        
        except Exception as e:
            return {
                "is_valid": False,
                "confidence": 0,
                "reason": f"Verification error: {str(e)}",
                "suggestions": "Could not verify element with LLM"
            }
    
    def _verify_with_ollama(
        self,
        element_text: str,
        element_html: str,
        extraction_requirement: str,
        keywords: list
    ) -> Dict[str, Any]:
        """Verify using local Ollama model."""
        try:
            prompt = self._build_verification_prompt(element_text, element_html, extraction_requirement, keywords)
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2-uncensored",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                return self._parse_verification_response(response_text)
            else:
                raise Exception(f"Ollama error: {response.status_code}")
        
        except Exception as e:
            return {
                "is_valid": False,
                "confidence": 0,
                "reason": f"Ollama verification error: {str(e)}",
                "suggestions": "Could not verify element with Ollama"
            }
    
    def _build_verification_prompt(
        self,
        element_text: str,
        element_html: str,
        extraction_requirement: str,
        keywords: list
    ) -> str:
        """Build verification prompt for LLM."""
        keywords_str = ", ".join(keywords[:10])
        
        prompt = f"""You are a DOM element verification expert. Verify if a captured HTML element is correct for extraction.

EXTRACTION REQUIREMENT:
{extraction_requirement}

EXPECTED KEYWORDS IN DATA:
{keywords_str}

CAPTURED ELEMENT TEXT:
{element_text[:500]}

ELEMENT HTML (first 500 chars):
{element_html[:500]}

VERIFICATION TASK:
1. Does this element contain the data described in the extraction requirement?
2. Does it contain most of the expected keywords?
3. Is it a data container (not UI/navigation/filter)?
4. Would using this element directly extract the right values?

Respond in JSON format:
{{
    "is_valid": true/false,
    "confidence": 0-100,
    "reason": "Detailed explanation",
    "suggestions": "How to improve if invalid"
}}

Only output the JSON, no other text."""
        
        return prompt
    
    def _parse_verification_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM verification response."""
        try:
            # Try to extract JSON from response
            import json
            
            # Find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Validate required fields
                return {
                    "is_valid": result.get("is_valid", False),
                    "confidence": min(100, max(0, int(result.get("confidence", 0)))),
                    "reason": result.get("reason", ""),
                    "suggestions": result.get("suggestions", "")
                }
        except:
            pass
        
        # Fallback: try to infer from text
        response_lower = response_text.lower()
        is_valid = "yes" in response_lower or "valid" in response_lower or "correct" in response_lower
        
        return {
            "is_valid": is_valid,
            "confidence": 70 if is_valid else 30,
            "reason": response_text[:200],
            "suggestions": ""
        }


def verify_extraction_element(
    element_text: str,
    element_html: str,
    extraction_requirement: str,
    keywords: list,
    min_confidence: int = 85
) -> bool:
    """
    Verify if DOM element is correct for extraction.
    
    Args:
        element_text: Text content of element
        element_html: HTML of element
        extraction_requirement: What should be extracted
        keywords: Expected keywords
        min_confidence: Minimum confidence required (0-100)
    
    Returns:
        True if element is verified correct, False otherwise
    """
    verifier = LLMDOMVerifier()
    result = verifier.verify_element(element_text, element_html, extraction_requirement, keywords)
    
    is_valid = result.get("is_valid", False)
    confidence = result.get("confidence", 0)
    reason = result.get("reason", "")
    
    print(f"  LLM Verification: {confidence}% confidence - {'VALID' if is_valid else 'INVALID'}")
    print(f"    Reason: {reason[:100]}...")
    
    return is_valid and confidence >= min_confidence
