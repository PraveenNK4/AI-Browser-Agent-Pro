# ActionRouter: Simplified - Let LLM generate action JSON directly
# Browser-use will handle Playwright script generation via built-in methods

from typing import List, Dict, Any

class ActionRouter:
    """
    Simplified router - enforces retrieve-before-validate pattern.
    LLM generates action JSON, browser-use generates Playwright script.
    """
    def __init__(self):
        pass

    def route(self, user_intent: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Simple routing:
        - retrieve/read/state → retrieve_value_by_element
        - validate/check/verify → retrieve_value_by_element + validate_value
        - Other intents → pass through (let LLM decide)
        """
        intent = user_intent.lower()
        actions = []
        
        retrieve_keywords = {"retrieve", "read", "state", "status", "get", "extract"}
        validate_keywords = {"check", "validate", "verify", "confirm", "ensure"}
        
        if any(word in intent for word in retrieve_keywords):
            actions.append({
                "action": "retrieve_value_by_element",
                "params": params
            })
        elif any(word in intent for word in validate_keywords):
            # Retrieve first, then validate
            actions.append({
                "action": "retrieve_value_by_element",
                "params": params
            })
            actions.append({
                "action": "validate_value",
                "params": {
                    "actual": "<retrieved_value>",
                    "expected": params.get("expected", ""),
                    "operator": params.get("operator", "equals")
                }
            })
        else:
            # Default: truly let LLM decide - return empty to pass through
            pass  # No forced action, LLM chooses
        
        return actions