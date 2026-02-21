"""Parse user prompts into structured processes."""
import re
from typing import List
from .process_models import Process

PROCESS_HEADER = re.compile(r'^(Mandatory Process|Process \d+):', re.IGNORECASE)


def parse_processes(prompt: str) -> List[Process]:
    """
    Parse user prompt into structured processes.
    
    Format:
        Mandatory Process: Description
        - Step 1
        - Step 2
        
        Process 1: Description
        - Step 1
        - Step 2
    
    Args:
        prompt: User's process description text
        
    Returns:
        List of parsed Process objects
    """
    processes: List[Process] = []
    current = None

    for line in prompt.splitlines():
        line = line.strip()
        if not line:
            continue

        if PROCESS_HEADER.match(line):
            if current:
                processes.append(current)

            is_mandatory = line.lower().startswith("mandatory")
            current = Process(
                name=line.replace(":", "").strip(),
                steps=[],
                is_mandatory=is_mandatory
            )
        else:
            if current:
                current.steps.append(line.lstrip("- ").strip())

    if current:
        processes.append(current)

    # If no process headers found, treat entire prompt as single mandatory process
    if not processes:
        processes.append(Process(
            name="Mandatory Process",
            steps=[prompt],
            is_mandatory=True
        ))

    return processes
