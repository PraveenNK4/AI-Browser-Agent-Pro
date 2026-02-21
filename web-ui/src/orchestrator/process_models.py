"""Data models for process orchestration."""
from dataclasses import dataclass
from typing import List, Optional


class ProcessFailed(Exception):
    """Exception raised when a mandatory process fails."""
    pass


@dataclass
class Process:
    """Represents a single process to be executed."""
    name: str
    steps: List[str]
    is_mandatory: bool = False
    success_assertions: Optional[List[str]] = None


@dataclass
class ProcessResult:
    """Result of executing a single process."""
    process: Process
    history: Optional[object] = None  # AgentHistoryList
    success: bool = False
    error: Optional[str] = None
    docx_path: Optional[str] = None
    gif_path: Optional[str] = None
    playwright_script_path: Optional[str] = None
    history_path: Optional[str] = None
