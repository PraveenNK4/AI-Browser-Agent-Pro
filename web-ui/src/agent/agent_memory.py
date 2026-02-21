
import json
import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class AgentMemory:
    def __init__(self, memory_path: str = "./agent_memory.json"):
        self.memory_path = memory_path
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r') as f:
                    self.memory = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode memory file at {self.memory_path}. Starting fresh.")
                self.memory = self._get_empty_memory_structure()
        else:
            self.memory = self._get_empty_memory_structure()

    def _get_empty_memory_structure(self) -> Dict[str, Any]:
        return {
            "learned_selectors": {}, # {domain: {element_type: [selectors]}}
            "task_history": []       # [{task: str, result: str, timestamp: str}]
        }

    def save_memory(self):
        try:
            with open(self.memory_path, 'w') as f:
                json.dump(self.memory, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def learn_selector(self, domain: str, element_type: str, selector: str):
        """Record successful selectors for a domain"""
        if domain not in self.memory["learned_selectors"]:
            self.memory["learned_selectors"][domain] = {}
        
        if element_type not in self.memory["learned_selectors"][domain]:
            self.memory["learned_selectors"][domain][element_type] = []
            
        if selector not in self.memory["learned_selectors"][domain][element_type]:
            logger.info(f"🧠 Learning new selector for {domain} ({element_type}): {selector}")
            self.memory["learned_selectors"][domain][element_type].append(selector)
            self.save_memory()

    def add_task_history(self, task: str, result: str):
        """Record a completed task"""
        import datetime
        entry = {
            "task": task,
            "result": result,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.memory["task_history"].append(entry)
        self.save_memory()

    def get_similar_tasks(self, task_description: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Find relevant history using simple keyword overlap (Vector DB would be better but this is 'Vault' MVP)"""
        # Simple overlap score
        def score(t1, t2):
            words1 = set(t1.lower().split())
            words2 = set(t2.lower().split())
            return len(words1.intersection(words2))

        scored_tasks = []
        for entry in self.memory["task_history"]:
            s = score(task_description, entry["task"])
            if s > 0:
                scored_tasks.append((s, entry))
        
        scored_tasks.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored_tasks[:limit]]

    def get_learned_selectors(self, domain: str) -> Dict[str, List[str]]:
        """Get all learned selectors for a domain"""
        return self.memory["learned_selectors"].get(domain, {})

# Global instance
agent_memory = AgentMemory()
