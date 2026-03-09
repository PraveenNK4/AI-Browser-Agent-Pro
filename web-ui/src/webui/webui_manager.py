import json
from collections.abc import Generator
from typing import TYPE_CHECKING
import os
import gradio as gr
from datetime import datetime
from typing import Optional, Dict, List
import uuid
import asyncio
import time

from gradio.components import Component
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.agent.service import Agent
from src.browser.custom_browser import CustomBrowser
from src.browser.custom_context import CustomBrowserContext
from src.controller.custom_controller import CustomController
# from src.agent.deep_research.deep_research_agent import DeepResearchAgent


class WebuiManager:
    def __init__(self, settings_save_dir: str = "./tmp/webui_settings"):
        self.id_to_component: dict[str, Component] = {}
        self.component_to_id: dict[Component, str] = {}

        self.settings_save_dir = settings_save_dir
        os.makedirs(self.settings_save_dir, exist_ok=True)

    def init_browser_use_agent(self) -> None:
        """
        init browser use agent
        """
        self.bu_agent: Optional[Agent] = None
        self.bu_browser: Optional[CustomBrowser] = None
        self.bu_browser_context: Optional[CustomBrowserContext] = None
        self.bu_controller: Optional[CustomController] = None
        self.bu_chat_history: List[Dict[str, Optional[str]]] = []
        self.bu_response_event: Optional[asyncio.Event] = None
        self.bu_user_help_response: Optional[str] = None
        self.bu_current_task: Optional[asyncio.Task] = None
        self.bu_agent_task_id: Optional[str] = None
        self.bu_docx_path: Optional[str] = None
        self.bu_step_start_time: Optional[float] = None
        self.bu_previous_tokens: int = 0
        # Failure and hallucination tracking
        self.bu_step_failures: Dict[int, int] = {}  # Maps step number to failure count
        self.bu_consecutive_failures: int = 0  # Counter for consecutive failures
        self.bu_last_action: Optional[str] = None  # Track last action for hallucination
        self.bu_repeated_action_count: int = 0  # Count consecutive repeated actions
        self.bu_hallucination_triggered: bool = False  # Flag if hallucination stop was triggered
        self.bu_should_stop_agent: bool = False  # Flag to stop agent execution
        # Step progression tracking
        self.bu_highest_step_reached: int = 0  # Track highest step number reached to detect backward steps
        self.bu_step_backtrack_count: int = 0  # Count how many times agent went backward

    # def init_deep_research_agent(self) -> None:
    #     """
    #     init deep research agent
    #     """
    #     self.dr_agent: Optional[DeepResearchAgent] = None
    #     self.dr_current_task = None
    #     self.dr_agent_task_id: Optional[str] = None
    #     self.dr_save_dir: Optional[str] = None

    def add_components(self, tab_name: str, components_dict: dict[str, "Component"]) -> None:
        """
        Add tab components
        """
        for comp_name, component in components_dict.items():
            comp_id = f"{tab_name}.{comp_name}"
            self.id_to_component[comp_id] = component
            self.component_to_id[component] = comp_id

    def get_components(self) -> list["Component"]:
        """
        Get all components
        """
        return list(self.id_to_component.values())

    def get_component_by_id(self, comp_id: str) -> "Component":
        """
        Get component by id
        """
        return self.id_to_component[comp_id]

    def get_id_by_component(self, comp: "Component") -> str:
        """
        Get id by component
        """
        return self.component_to_id[comp]

    def save_config(self, components: Dict["Component", str]) -> None:
        """
        Save config
        """
        cur_settings = {}
        for comp in components:
            if not isinstance(comp, gr.Button) and not isinstance(comp, gr.File) and str(
                    getattr(comp, "interactive", True)).lower() != "false":
                comp_id = self.get_id_by_component(comp)
                cur_settings[comp_id] = components[comp]

        config_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(os.path.join(self.settings_save_dir, f"{config_name}.json"), "w") as fw:
            json.dump(cur_settings, fw, indent=4)

        return os.path.join(self.settings_save_dir, f"{config_name}.json")

    def load_config(self, config_path: str):
        """
        Load config
        """
        if config_path is None:
            config_status = self.id_to_component["load_save_config.config_status"]
            yield {config_status: config_status.__class__(value="No config file selected.")}
            return

        with open(config_path, "r") as fr:
            ui_settings = json.load(fr)

        update_components = {}
        for comp_id, comp_val in ui_settings.items():
            if comp_id in self.id_to_component:
                comp = self.id_to_component[comp_id]
                if comp.__class__.__name__ == "Chatbot":
                    update_components[comp] = comp.__class__(value=comp_val)
                else:
                    update_components[comp] = comp.__class__(value=comp_val)
                    if comp_id == "agent_settings.planner_llm_provider":
                        yield update_components  # yield provider, let callback run
                        time.sleep(0.1)  # wait for Gradio UI callback

        config_status = self.id_to_component["load_save_config.config_status"]
        update_components.update(
            {
                config_status: config_status.__class__(value=f"Successfully loaded config: {config_path}")
            }
        )
        yield update_components

    def check_step_failure(self, step_num: int, failed: bool = True) -> bool:
        """
        Track step failures. Returns True if step exceeded 3 failures and should stop.
        
        Args:
            step_num: Current step number
            failed: Whether this step failed (default True)
            
        Returns:
            True if step failures exceeded limit (3), False otherwise
        """
        if failed:
            if step_num not in self.bu_step_failures:
                self.bu_step_failures[step_num] = 0
            self.bu_step_failures[step_num] += 1
            self.bu_consecutive_failures += 1
            
            if self.bu_step_failures[step_num] >= 3:
                return True  # Stop: 3 failures on this step
        else:
            self.bu_consecutive_failures = 0
        
        return False

    def check_hallucination(self, current_action: Optional[str]) -> bool:
        """
        Detect hallucination: same core action repeated 3+ times consecutively.
        
        Normalizes actions by extracting the key action name + args,
        ignoring padding actions like scroll_page, wait, etc.
        
        Args:
            current_action: The current action being taken
            
        Returns:
            True if hallucination detected (3+ repeated actions), False otherwise
        """
        if not current_action:
            self.bu_last_action = None
            self.bu_repeated_action_count = 0
            return False
        
        # Normalize: extract core action signature for comparison
        # This strips noise like scroll_page padding that makes actions look different
        import re
        _noise = r'(scroll_page|wait|sleep)\([^)]*\),?\s*'
        normalized = re.sub(_noise, '', current_action).strip().rstrip(',')
        if not normalized:
            normalized = current_action  # fallback if all actions were noise
        
        # Check if core action is the same as last action
        if normalized == self.bu_last_action:
            self.bu_repeated_action_count += 1
        else:
            self.bu_last_action = normalized
            self.bu_repeated_action_count = 1
        
        # If same action repeated 3+ times, it's hallucination
        if self.bu_repeated_action_count >= 3:
            self.bu_hallucination_triggered = True
            return True
        
        return False

    def reset_failure_tracking(self) -> None:
        """Reset failure and hallucination tracking for a new task."""
        self.bu_step_failures = {}
        self.bu_consecutive_failures = 0
        self.bu_last_action = None
        self.bu_repeated_action_count = 0
        self.bu_hallucination_triggered = False
        self.bu_highest_step_reached = 0
        self.bu_step_backtrack_count = 0

    def check_step_progression(self, current_step: int) -> bool:
        """
        Track step progression and detect if agent is going backward unnecessarily.
        
        Args:
            current_step: The current step number
            
        Returns:
            True if agent is making backward progress excessively, False if progressing normally
        """
        # Update highest step reached
        if current_step > self.bu_highest_step_reached:
            self.bu_highest_step_reached = current_step
            return False  # Normal forward progression
        
        # If we're at a step we've been to before, check if it's going backward
        if current_step < self.bu_highest_step_reached:
            self.bu_step_backtrack_count += 1
            # NOTE: Removed noisy warning - false positives during multi-process orchestration
            
            # If agent backtracks more than 2 times in multi-step tasks, it might be confused
            if self.bu_step_backtrack_count > 2:
                # logger.error(f"❌ Agent backtracked {self.bu_step_backtrack_count} times. Likely confused about workflow.")
                return True  # Signal that backward progression is excessive
        
        return False
        self.bu_should_stop_agent = False

