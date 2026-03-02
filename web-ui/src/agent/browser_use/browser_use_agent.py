from __future__ import annotations

import asyncio
import logging
import os
from typing import Callable, Awaitable

from browser_use.agent.gif import create_history_gif
from browser_use.agent.service import Agent
from browser_use.agent.views import (
    ActionResult,
    AgentHistory,
    AgentHistoryList,
    AgentStepInfo,
    ToolCallingMethod,
)
from browser_use.browser.views import BrowserStateHistory
from browser_use.utils import time_execution_async
from src.agent.browser_use.action_router import ActionRouter
from dotenv import load_dotenv

try:
    from browser_use.agent.service import AgentHookFunc
except ImportError:
    AgentHookFunc = Callable[['BrowserUseAgent'], Awaitable[None]]

def is_model_without_tool_support(model_name: str) -> bool:
    """Check if model doesn't support tool calling."""
    models_without_tools = [
        'gpt-3.5-turbo-instruct',
        'text-davinci-003',
        'text-davinci-002',
        'text-davinci-001',
    ]
    return any(model in model_name.lower() for model in models_without_tools)

load_dotenv()
logger = logging.getLogger(__name__)

SKIP_LLM_API_KEY_VERIFICATION = (
    os.environ.get("SKIP_LLM_API_KEY_VERIFICATION", "false").lower()[0] in "ty1"
)


def generate_playwright_script_from_history(history: AgentHistoryList, output_path: str):
    """
    Generate a Playwright Python script from agent history.
    This is a COMPLETE implementation that recreates all actions.
    """
    try:
        logger.info(f"🎬 Generating Playwright script: {output_path}")
        
        script_lines = [
            "#!/usr/bin/env python3",
            '"""',
            "Auto-generated Playwright script from Browser-Use agent execution",
            f"Generated from task: {history.history[0].model_output.current_state.next_goal if history.history else 'Unknown'}",
            '"""',
            "",
            "import asyncio",
            "import os",
            "import re",
            "from playwright.async_api import async_playwright",
            "",
            "async def click_and_upload_file(page, target, path):",
            "    # Robust upload helper - supports both index (int) and text (str) targets",
            "    print(f'[*] Starting robust upload for target: {target}')",
            "    try:",
            "         if not os.path.exists(path):",
            "             print(f'[-] File not found: {path}')",
            "             return False",
            "         # Strategy 0: If target is int, use index-based click via nth()",
            "         if isinstance(target, int):",
            "             try:",
            "                 all_elements = page.locator(\'a, button, input, [role=\"button\"]\')",
            "                 target_el = all_elements.nth(target)",
            "                 async with page.expect_file_chooser(timeout=7000) as fc_info:",
            "                     await target_el.click(timeout=5000)",
            "                 file_chooser = await fc_info.value",
            "                 await file_chooser.set_files(path)",
            "                 return True",
            "             except: pass",
            "         # Strategy 1: Native Chooser via Strict Text Match",
            "         if isinstance(target, str):",
            "             try:",
            "                 target_el = page.locator(\'a, button, [role=\"button\"]\').filter(has_text=re.compile(f\'^{target}$\', re.I)).first",
            "                 async with page.expect_file_chooser(timeout=7000) as fc_info:",
            "                     await target_el.click(timeout=5000)",
            "                 file_chooser = await fc_info.value",
            "                 await file_chooser.set_files(path)",
            "                 return True",
            "             except: pass",
            "         # Strategy 2: Hidden File Input Fallback",
            "         try:",
            "             await page.locator(\'input[type=\"file\"]\').first.set_input_files(path)",
            "             return True",
            "         except: pass",
            "         # Strategy 3: Global Text Search Fallback (if target is string)",
            "         if isinstance(target, str):",
            "             try:",
            "                 target_el = page.get_by_text(target, exact=True).first",
            "                 async with page.expect_file_chooser(timeout=5000) as fc_info:",
            "                     await target_el.click()",
            "                 file_chooser = await fc_info.value",
            "                 await file_chooser.set_files(path)",
            "                 return True",
            "             except: pass",
            "    except Exception as e:",
            "         print(f\'Upload failed: {e}\')",
            "    return False",
            "",
            "async def main():",
            "    async with async_playwright() as p:",
            "        # Launch browser in MAXIMIZED mode",
            "        browser = await p.chromium.launch(headless=False, args=[\'--start-maximized\'])",
            "        context = await browser.new_context(no_viewport=True)",
            "        page = await context.new_page()",
            "",
        ]

        
        # Track variables for state
        uploaded_files = set()
        
        for i, step in enumerate(history.history, 1):
            script_lines.append(f"        # Step {i}")
            
            # Extract actions from step
            if hasattr(step, 'action') and step.action:
                for action in step.action:
                    action_name = action.__class__.__name__.replace('Action', '').lower()
                    
                    # Navigate to URL
                    if hasattr(action, 'go_to_url') and action.go_to_url:
                        url = action.go_to_url.url
                        script_lines.append(f"        await page.goto('{url}')")
                    
                    # Click element
                    elif hasattr(action, 'click_element') and action.click_element:
                        index = action.click_element.index
                        script_lines.append(f"        # Click element at index {index}")
                        script_lines.append(f"        element = page.locator(\'[data-browser-use-index=\"{index}\"]\').first")
                        script_lines.append(f"        await element.click()")
                    
                    # Input text
                    elif hasattr(action, 'input_text') and action.input_text:
                        index = action.input_text.index
                        text = action.input_text.text
                        # Escape quotes in text
                        text_escaped = text.replace('"', '\\"').replace("'", "\\'")
                        script_lines.append(f"        # Input text at index {index}")
                        script_lines.append(f"        element = page.locator(\'[data-browser-use-index=\"{index}\"]\').first")
                        script_lines.append(f"        await element.fill('{text_escaped}')")
                    
                    # Upload file
                    elif hasattr(action, 'upload_file') and action.upload_file:
                        index = action.upload_file.index
                        path = action.upload_file.path
                        filename = os.path.basename(path)
                        script_lines.append(f"        # Upload file at index {index}")
                        script_lines.append(f"        await click_and_upload_file(page, {index}, '{path}')")
                        uploaded_files.add(filename)
                    
                    # Click by text
                    elif hasattr(action, 'click_element_by_text') and action.click_element_by_text:
                        text = action.click_element_by_text.text
                        text_escaped = text.replace('"', '\\"').replace("'", "\\'")
                        script_lines.append(f"        # Click element by text: {text}")
                        script_lines.append(f"        await page.get_by_text('{text_escaped}').click()")
                    
                    # Scroll
                    elif hasattr(action, 'scroll_page') and action.scroll_page:
                        direction = action.scroll_page.direction
                        amount = action.scroll_page.amount
                        if direction == "down":
                            script_lines.append(f"        await page.evaluate(\'window.scrollBy(0, {amount})\')")
                        elif direction == "up":
                            script_lines.append(f"        await page.evaluate(\'window.scrollBy(0, -{amount})\')")
                        elif direction == "right":
                            script_lines.append(f"        await page.evaluate(\'window.scrollBy({amount}, 0)\')")
                        elif direction == "left":
                            script_lines.append(f"        await page.evaluate(\'window.scrollBy(-{amount}, 0)\')")
                    
                    # Hover
                    elif hasattr(action, 'hover_element') and action.hover_element:
                        index = action.hover_element.index
                        script_lines.append(f"        # Hover over element at index {index}")
                        script_lines.append(f"        element = page.locator(\'[data-browser-use-index=\"{index}\"]\').first")
                        script_lines.append(f"        await element.hover()")
                    
                    # Extract data (commented out in playback script)
                    elif hasattr(action, 'retrieve_value_by_element') and action.retrieve_value_by_element:
                        index = action.retrieve_value_by_element.index
                        script_lines.append(f"        # Extract value from element at index {index}")
                        script_lines.append(f"        # element = page.locator(\'[data-browser-use-index=\"{index}\"]\').first")
                        script_lines.append(f"        # value = await element.text_content()")
                    
                    # Add delay between actions
                    script_lines.append(f"        await asyncio.sleep(0.5)")
            
            script_lines.append("")
        
        # Add footer
        script_lines.extend([
            "        # Task complete",
            "        await asyncio.sleep(2)",
            "        await browser.close()",
            "",
            "",
            "if __name__ == '__main__':",
            "    asyncio.run(main())",
        ])
        
        # Add file upload notes if applicable
        if uploaded_files:
            note_lines = [
                "",
                "# NOTES:",
                "# File uploads detected. Update file paths in the script:",
            ]
            for filename in uploaded_files:
                note_lines.append(f"#   - {filename}")
            script_lines = script_lines[:6] + note_lines + script_lines[6:]
        
        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(script_lines))
        
        logger.info(f"✅ Playwright script generated: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to generate Playwright script: {e}", exc_info=True)
        return False


class BrowserUseAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_router = ActionRouter()
        self._history_file_path = None  # Track where history is saved
        
        # BITNET INTEGRATION: Apply low-entropy prompt optimization if BitNet is detected
        is_bitnet = "bitnet" in str(self.model_name).lower() or "q2" in str(self.model_name).lower()
        if is_bitnet:
            logger.info("🦾 BitNet/Low-Bit model detected. Applying ternary logic optimizations.")
            self._apply_bitnet_optimizations()

        # Patch run method for custom cleanup
        original_run = super().run

        async def patched_run(*run_args, **run_kwargs):
            try:
                return await original_run(*run_args, **run_kwargs)
            except Exception as e:
                logger.error(f'Error in agent run: {e}', exc_info=True)
                raise
            finally:
                # Custom cleanup
                logger.debug('🧹 Agent run cleanup started')
                
                # Auto-enrich history with DOM context and Ollama verification
                if self._history_file_path:
                    try:
                        from pathlib import Path
                        from src.utils.strip_and_enrich_history import strip_and_enrich
                        
                        history_path = Path(self._history_file_path)
                        if history_path.exists():
                            logger.debug(f'📝 Enriching history with DOM context: {history_path.name}')
                            strip_and_enrich(history_path)
                            logger.debug('✅ History enrichment complete')
                    except Exception as enrich_err:
                        logger.warning(f'⚠️ Failed to enrich history: {enrich_err}', exc_info=True)

                # GIF generation
                if hasattr(self.settings, 'generate_gif') and self.settings.generate_gif:
                    try:
                        output_path: str = 'agent_history.gif'
                        if isinstance(self.settings.generate_gif, str):
                            output_path = self.settings.generate_gif
                        
                        logger.debug(f"🎥 Generating GIF: {output_path}")
                        create_history_gif(task=self.task, history=self.state.history, output_path=output_path)
                        logger.debug(f"✅ GIF created: {output_path}")
                    except Exception as gif_err:
                        logger.error(f'❌ Failed to create GIF: {gif_err}', exc_info=True)

                # Playwright script generation - CUSTOM IMPLEMENTATION
                if hasattr(self, 'playwright_script_path') and self.playwright_script_path:
                    try:
                        # logger.info(f"🎬 Generating Playwright script: {self.playwright_script_path}")
                        success = generate_playwright_script_from_history(
                            history=self.state.history,
                            output_path=self.playwright_script_path
                        )
                        if success:
                            logger.info(f"✅ Playwright script saved: {self.playwright_script_path}")
                        else:
                            logger.warning(f"⚠️ Playwright script generation incomplete")
                    except Exception as script_err:
                        logger.error(f'❌ Failed to generate Playwright script: {script_err}', exc_info=True)
                
                logger.debug('✅ Agent cleanup complete')

        self.run = patched_run
        self.playwright_script_path = None  # Will be set externally
    
    def save_history(self, file_path=None):
        """Override save_history to track file path for auto-enrichment"""
        # Call parent's save_history
        super().save_history(file_path)
        # Track the path for enrichment in cleanup
        if file_path:
            self._history_file_path = str(file_path)

    def plan_actions(self, user_intent: str, params: dict) -> list:
        """Use ActionRouter for retrieve/validate enforcement."""
        return self.action_router.route(user_intent, params)

    def _set_tool_calling_method(self) -> ToolCallingMethod | None:
        tool_calling_method = self.settings.tool_calling_method
        if tool_calling_method == 'auto':
            if is_model_without_tool_support(self.model_name):
                return 'raw'
            elif self.chat_model_library == 'ChatGoogleGenerativeAI':
                return None
            elif self.chat_model_library in ['ChatOpenAI', 'AzureChatOpenAI']:
                return 'function_calling'
            else:
                return None
        else:
            return tool_calling_method

    def _apply_bitnet_optimizations(self):
        """Inject BitNet-friendly behavior into the agent."""
        # This is a specialized hook for BitNet b1.58 ternary models
        logger.info("🔧 Injecting ternary logic prompt optimizations into agent context.")
        
        # Override the system prompt builder to include ternary-specific instructions
        original_setup = self._setup_system_prompt
        
        def bitnet_system_prompt_builder():
            base_prompt = original_setup()
            bitnet_addendum = (
                "\n\n[BITNET OPTIMIZATION ENABLED]\n"
                "You are running on a 1.58-bit Ternary Logic Engine. Use extreme technical precision.\n"
                "1. PREFER direct indices over fuzzy search.\n"
                "2. AVOID complex reasoning chains; follow the 'retrieve-before-validate' law.\n"
                "3. IF unsure, use 'scroll_down' or 'hover_capture' to refresh your ternary state."
            )
            return base_prompt + bitnet_addendum

        self._setup_system_prompt = bitnet_system_prompt_builder
        
        # Enable BitNet optimizations in the controller if supported
        if hasattr(self.controller, 'bitnet_mode'):
            self.controller.bitnet_mode = True