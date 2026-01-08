import asyncio
import base64
import io
import json
import logging
import os
import platform
import shutil
import uuid
import xml.etree.ElementTree as ET
from typing import Any, AsyncGenerator, Dict, Optional

from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from PIL import Image

import gradio as gr

# from browser_use.agent.service import Agent
from browser_use.agent.views import (
    AgentHistoryList,
    AgentOutput,
)
from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.browser.views import BrowserState
from gradio.components import Component
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.browser.custom_browser import CustomBrowser
from src.controller.custom_controller import CustomController
from src.utils import llm_provider
from src.webui.webui_manager import WebuiManager

logger = logging.getLogger(__name__)


def _detect_browser():
    """Detect available browsers and prefer Chrome for automation."""
    system = platform.system()

    if system == "Windows":
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Users\pnandank\AppData\Local\Google\Chrome\Application\chrome.exe",  # User install
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        edge_paths = [
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        ]
    elif system == "Darwin":  # macOS
        chrome_paths = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
        edge_paths = ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
    else:  # Linux
        chrome_paths = ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/opt/google/chrome/chrome"]
        edge_paths = ["/usr/bin/microsoft-edge", "/usr/bin/microsoft-edge-stable"]

    # Prefer Chrome over Edge for automation
    for path in chrome_paths + edge_paths:
        if os.path.exists(path):
            logger.info(f"Detected browser for automation: {path}")
            return path

    logger.warning("No supported browser found. Will use system default.")
    return None


def load_credentials_from_xml():
    """Load credentials from XML file."""
    credentials_file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'credentials.xml')
    credentials = {}

    if os.path.exists(credentials_file):
        try:
            tree = ET.parse(credentials_file)
            root = tree.getroot()

            for site in root.findall('site'):
                site_name = site.get('name')
                if site_name:
                    credentials[site_name] = {
                        'username': site.find('username').text if site.find('username') is not None else '',
                        'password': site.find('password').text if site.find('password') is not None else '',
                        'url': site.find('url').text if site.find('url') is not None else ''
                    }
            logger.info(f"Loaded {len(credentials)} site credentials from XML")
        except Exception as e:
            logger.error(f"Error loading credentials from XML: {e}")
    else:
        logger.warning(f"Credentials file not found: {credentials_file}")

    return credentials


def replace_credential_placeholders(task_text, credentials):
    """Replace credential placeholders in task text with secure references."""
    import re

    # Store credentials in webui_manager for secure access during execution
    # Instead of exposing credentials in the task text, we'll use placeholders

    # First, check if any stored site names appear in the task
    task_lower = task_text.lower()
    for site_name, creds in credentials.items():
        site_name_lower = site_name.lower()
        # Check if the site name appears in the task
        if site_name_lower in task_lower:
            # If we find a site reference, add login instruction with placeholders
            if 'login' not in task_lower and 'log in' not in task_lower:
                # Get the URL from credentials if available
                site_url = creds.get('url', f'https://{site_name}')
                # Use secure placeholders instead of actual credentials
                login_instruction = f"First, navigate to {site_url} and log in to {site_name} using the stored credentials for this site. "
                task_text = login_instruction + task_text
                logger.info(f"Added automatic login instruction with URL for site: {site_name} -> {site_url} (credentials will be injected securely)")
                break

    # Also check for the original patterns as fallback
    patterns = [
        r'login to (\w+)',
        r'log into (\w+)',
        r'(\w+)\.com',
        r'(\w+) website',
        r'(\w+) site'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, task_text, re.IGNORECASE)
        for match in matches:
            site_name = match.lower()
            if site_name in credentials:
                # Replace generic credential requests with secure placeholders
                task_text = re.sub(
                    r'credentials?:?\s*(username|user|login)?:?\s*\w+\s*(password|pass)?:?\s*\w+',
                    f'Use the stored credentials for {site_name}',
                    task_text,
                    flags=re.IGNORECASE
                )
                logger.info(f"Replaced credential request with secure reference for site: {site_name}")
                break

    return task_text


def setup_secure_credentials(credentials):
    """
    Set up credentials securely in environment variables without exposing to LLM.
    Returns a mapping of site names to secure references.
    """
    credential_mapping = {}

    for site_name, creds in credentials.items():
        # Set environment variables that the agent can access securely
        os.environ[f'AUTOMATION_{site_name.upper()}_USERNAME'] = creds['username']
        os.environ[f'AUTOMATION_{site_name.upper()}_PASSWORD'] = creds['password']
        os.environ[f'AUTOMATION_{site_name.upper()}_URL'] = creds.get('url', f'https://{site_name}')

        # Create a secure reference for the LLM
        credential_mapping[site_name.lower()] = f"stored_credentials_{site_name.lower()}"

        logger.info(f"Securely stored credentials for {site_name} in environment variables")

    return credential_mapping


def inject_secure_credential_references(task_text, credential_mapping):
    """
    Replace credential references with secure placeholders that don't expose actual values.
    """
    for site_name, secure_ref in credential_mapping.items():
        if site_name in task_text.lower():
            # Replace actual credential exposure with secure reference
            task_text = task_text.replace(
                f"using the stored credentials for this site",
                f"using {secure_ref}"
            )
            break

    return task_text


# --- Helper Functions --- (Defined at module level)


async def _initialize_llm(
        provider: Optional[str],
        model_name: Optional[str],
        temperature: float,
        base_url: Optional[str],
        api_key: Optional[str],
        num_ctx: Optional[int] = None,
) -> Optional[BaseChatModel]:
    """Initializes the LLM based on settings. Returns None if provider/model is missing."""
    if not provider or not model_name:
        logger.info("LLM Provider or Model Name not specified, LLM will be None.")
        return None
    try:
        # Use your actual LLM provider logic here
        logger.info(
            f"Initializing LLM: Provider={provider}, Model={model_name}, Temp={temperature}"
        )
        # Example using a placeholder function
        llm = llm_provider.get_llm_model(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            base_url=base_url or None,
            api_key=api_key or None,
            # Add other relevant params like num_ctx for ollama
            num_ctx=num_ctx if provider == "ollama" else None,
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        gr.Warning(
            f"Failed to initialize LLM '{model_name}' for provider '{provider}'. Please check settings. Error: {e}"
        )
        return None


def _get_config_value(
        webui_manager: WebuiManager,
        comp_dict: Dict[gr.components.Component, Any],
        comp_id_suffix: str,
        default: Any = None,
) -> Any:
    """Safely get value from component dictionary using its ID suffix relative to the tab."""
    # Assumes component ID format is "tab_name.comp_name"
    tab_name = "browser_use_agent"  # Hardcode or derive if needed
    comp_id = f"{tab_name}.{comp_id_suffix}"
    # Need to find the component object first using the ID from the manager
    try:
        comp = webui_manager.get_component_by_id(comp_id)
        return comp_dict.get(comp, default)
    except KeyError:
        # Try accessing settings tabs as well
        for prefix in ["agent_settings", "browser_settings"]:
            try:
                comp_id = f"{prefix}.{comp_id_suffix}"
                comp = webui_manager.get_component_by_id(comp_id)
                return comp_dict.get(comp, default)
            except KeyError:
                continue
        logger.warning(
            f"Component with suffix '{comp_id_suffix}' not found in manager for value lookup."
        )
        return default


def _format_agent_output(model_output: AgentOutput) -> str:
    """Formats AgentOutput for display in the chatbot using JSON."""
    content = ""
    if model_output:
        try:
            # Directly use model_dump if actions and current_state are Pydantic models
            action_dump = [
                action.model_dump(exclude_none=True) for action in model_output.action
            ]

            state_dump = model_output.current_state.model_dump(exclude_none=True)
            model_output_dump = {
                "current_state": state_dump,
                "action": action_dump,
            }
            # Dump to JSON string with indentation
            json_string = json.dumps(model_output_dump, indent=4, ensure_ascii=False)
            # Wrap in <pre><code> for proper display in HTML
            content = f"<pre><code class='language-json'>{json_string}</code></pre>"

        except AttributeError as ae:
            logger.error(
                f"AttributeError during model dump: {ae}. Check if 'action' or 'current_state' or their items support 'model_dump'."
            )
            content = f"<pre><code>Error: Could not format agent output (AttributeError: {ae}).\nRaw output: {str(model_output)}</code></pre>"
        except Exception as e:
            logger.error(f"Error formatting agent output: {e}", exc_info=True)
            # Fallback to simple string representation on error
            content = f"<pre><code>Error formatting agent output.\nRaw output:\n{str(model_output)}</code></pre>"

    return content.strip()


# --- Updated Callback Implementation ---


async def _handle_new_step(
        webui_manager: WebuiManager, state: BrowserState, output: AgentOutput, step_num: int
):
    """Callback for each step taken by the agent, including screenshot display."""

    # Use the correct chat history attribute name from the user's code
    if not hasattr(webui_manager, "bu_chat_history"):
        logger.error(
            "Attribute 'bu_chat_history' not found in webui_manager! Cannot add chat message."
        )
        # Initialize it maybe? Or raise an error? For now, log and potentially skip chat update.
        webui_manager.bu_chat_history = []  # Initialize if missing (consider if this is the right place)
        # return # Or stop if this is critical
    step_num -= 1
    logger.info(f"Step {step_num} completed.")

    # Calculate token usage for this step
    current_tokens = webui_manager.bu_agent.state.history.total_input_tokens()
    if not hasattr(webui_manager, 'bu_previous_tokens'):
        webui_manager.bu_previous_tokens = 0
    step_tokens = current_tokens - webui_manager.bu_previous_tokens
    webui_manager.bu_previous_tokens = current_tokens

    # Debug logging
    logger.info(f"[TOKEN DEBUG] Step {step_num-1}: current_tokens={current_tokens}, previous_tokens={webui_manager.bu_previous_tokens - step_tokens}, step_tokens={step_tokens}")

    # --- Screenshot Handling ---
    screenshot_html = ""
    # Ensure state.screenshot exists and is not empty before proceeding
    # Use getattr for safer access
    screenshot_data = getattr(state, "screenshot", None)
    if screenshot_data:
        try:
            # Basic validation: check if it looks like base64
            if (
                    isinstance(screenshot_data, str) and len(screenshot_data) > 100
            ):  # Arbitrary length check
                # *** UPDATED STYLE: Removed centering, adjusted width ***
                img_tag = f'<img src="data:image/jpeg;base64,{screenshot_data}" alt="Step {step_num} Screenshot" style="max-width: 800px; max-height: 600px; object-fit:contain;" />'
                screenshot_html = (
                        img_tag + "<br/>"
                )  # Use <br/> for line break after inline-block image
            else:
                logger.warning(
                    f"Screenshot for step {step_num} seems invalid (type: {type(screenshot_data)}, len: {len(screenshot_data) if isinstance(screenshot_data, str) else 'N/A'})."
                )
                screenshot_html = "**[Invalid screenshot data]**<br/>"

        except Exception as e:
            logger.error(
                f"Error processing or formatting screenshot for step {step_num}: {e}",
                exc_info=True,
            )
            screenshot_html = "**[Error displaying screenshot]**<br/>"
    else:
        logger.debug(f"No screenshot available for step {step_num}.")

    # --- Format Agent Output ---
    formatted_output = _format_agent_output(output)  # Use the updated function

    # --- Combine and Append to Chat ---
    step_header = f"--- **Step {step_num}** (Tokens: {step_tokens}) ---"
    # Combine header, image (with line break), and JSON block
    final_content = step_header + "<br/>" + screenshot_html + formatted_output

    chat_message = {
        "role": "assistant",
        "content": final_content.strip(),  # Remove leading/trailing whitespace
    }

    # Append to the correct chat history list
    webui_manager.bu_chat_history.append(chat_message)

    await asyncio.sleep(0.05)


def generate_docx_report(webui_manager: WebuiManager, history: AgentHistoryList, task: str):
    """Generate a well-formatted DOCX report with task details and screenshots."""
    try:
        # Create document
        doc = Document()
        doc.add_heading('AI Browser Agent Execution Report', 0)

        # Add task summary
        doc.add_heading('Task Summary', level=1)
        task_summary = doc.add_paragraph()
        task_summary.add_run('Original Task: ').bold = True
        task_summary.add_run(task)

        duration = history.total_duration_seconds()
        tokens = history.total_input_tokens()
        final_result = history.final_result()

        summary_table = doc.add_table(rows=4, cols=2)
        summary_table.style = 'Table Grid'
        cells = summary_table.rows[0].cells
        cells[0].text = 'Duration'
        cells[1].text = '.2f'
        cells = summary_table.rows[1].cells
        cells[0].text = 'Total Tokens Used'
        cells[1].text = str(tokens)
        cells = summary_table.rows[2].cells
        cells[0].text = 'Steps Executed'
        cells[1].text = str(len(history.history))
        cells = summary_table.rows[3].cells
        cells[0].text = 'Status'
        cells[1].text = 'Completed' if final_result else 'Failed'

        # Add final result
        if final_result:
            doc.add_heading('Final Result', level=1)
            result_para = doc.add_paragraph()
            result_para.add_run(final_result)

        # Add step-by-step execution
        doc.add_heading('Step-by-Step Execution', level=1)

        for i, step in enumerate(history.history, 1):
            doc.add_heading(f'Step {i}', level=2)

            # Step info
            step_para = doc.add_paragraph()
            step_para.add_run('Action: ').bold = True
            if hasattr(step, 'action') and step.action:
                action_text = str(step.action[0]) if step.action else 'No action'
                step_para.add_run(action_text)

            # Add screenshot if available - preserve quality
            if hasattr(step, 'state') and hasattr(step.state, 'screenshot') and step.state.screenshot:
                try:
                    # Decode base64 screenshot
                    screenshot_data = step.state.screenshot
                    if isinstance(screenshot_data, str) and screenshot_data.startswith('data:image'):
                        # Extract base64 data
                        screenshot_data = screenshot_data.split(',')[1]

                    image_data = base64.b64decode(screenshot_data)
                    image = Image.open(io.BytesIO(image_data))

                    # Calculate optimal size while preserving quality (max 6 inches wide)
                    max_width_inches = 6
                    dpi = 150  # Higher DPI for better quality
                    max_width_pixels = max_width_inches * dpi

                    if image.width > max_width_pixels:
                        ratio = max_width_pixels / image.width
                        new_size = (int(image.width * ratio), int(image.height * ratio))
                        image = image.resize(new_size, Image.Resampling.LANCZOS)

                    # Convert to RGB if necessary (DOCX doesn't support some formats)
                    if image.mode in ("RGBA", "LA", "P"):
                        image = image.convert("RGB")

                    # Save with high quality
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='JPEG', quality=95, optimize=True)
                    img_buffer.seek(0)

                    # Add to document
                    doc.add_picture(img_buffer, width=Inches(min(max_width_inches, image.width / dpi)))
                    doc.add_paragraph(f'Screenshot after Step {i}').alignment = WD_ALIGN_PARAGRAPH.CENTER

                except Exception as e:
                    logger.warning(f"Failed to add screenshot for step {i}: {e}")

            # Add step result - extract readable content from ActionResult
            if hasattr(step, 'result') and step.result:
                result_para = doc.add_paragraph()
                result_para.add_run('Result: ').bold = True

                # Try to extract meaningful content from ActionResult
                result_text = str(step.result)
                if hasattr(step.result, 'extracted_content') and step.result.extracted_content:
                    result_text = str(step.result.extracted_content)
                elif hasattr(step.result, 'success') and step.result.success is not None:
                    if step.result.success:
                        result_text = "✓ Action completed successfully"
                    else:
                        result_text = "✗ Action failed"
                        if hasattr(step.result, 'error') and step.result.error:
                            result_text += f": {step.result.error}"

                result_para.add_run(result_text)

        # Add errors if any
        errors = history.errors()
        if errors and any(errors):
            doc.add_heading('Errors Encountered', level=1)
            for error in errors:
                if error:
                    error_para = doc.add_paragraph()
                    error_para.add_run('Error: ').bold = True
                    error_para.add_run(str(error))

        # Save the document
        task_id = webui_manager.bu_agent_task_id
        docx_path = os.path.join(
            os.getenv("AGENT_HISTORY_PATH", "./tmp/agent_history"),
            task_id,
            f"{task_id}_report.docx"
        )

        doc.save(docx_path)
        logger.info(f"DOCX report saved to: {docx_path}")
        return docx_path

    except Exception as e:
        logger.error(f"Failed to generate DOCX report: {e}")
        return None


def enhance_playwright_script(script_content: str, task_id: str) -> str:
    """Enhance the generated Playwright script with sequential execution and error handling."""

    # Add imports and setup
    enhanced_script = f'''import asyncio
import logging
from playwright.async_api import async_playwright, expect
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create screenshots directory
screenshot_dir = Path(f"./tmp/agent_history/{task_id}/screenshots")
screenshot_dir.mkdir(parents=True, exist_ok=True)

async def run_automation():
    """Run the automation with sequential step execution and error handling."""

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={{"width": 1280, "height": 1024}})
        page = await context.new_page()

        step_results = []
        current_step = 0

        try:
'''

    # Parse the original script and add step-by-step execution
    lines = script_content.split('\n')
    in_async_function = False
    indent_level = 0
    step_actions = []

    for i, line in enumerate(lines):
        if 'async def run(' in line or 'async def main(' in line:
            in_async_function = True
            continue
        elif in_async_function and line.strip().startswith('await '):
            # This is an action that should be wrapped with error handling
            action_line = line.strip()
            step_actions.append(action_line)

            # Add enhanced step execution
            enhanced_script += f'''
            # Step {len(step_actions)}
            current_step = {len(step_actions)}
            logger.info(f"Executing Step {{current_step}}: {{action_line}}")

            try:
                # Take screenshot before action
                screenshot_path = screenshot_dir / f"step_{{current_step}}_before.png"
                await page.screenshot(path=str(screenshot_path))
                logger.info(f"Screenshot saved: {{screenshot_path}}")

                # Execute the action
                {action_line}

                # Wait for page to be stable
                await page.wait_for_load_state('networkidle')
                await asyncio.sleep(1)

                # Take screenshot after action
                screenshot_path = screenshot_dir / f"step_{{current_step}}_after.png"
                await page.screenshot(path=str(screenshot_path))
                logger.info(f"Screenshot saved: {{screenshot_path}}")

                # Validate step success (basic check)
                step_results.append({{
                    "step": current_step,
                    "action": "{action_line}",
                    "status": "success",
                    "screenshot_before": str(screenshot_path),
                    "screenshot_after": str(screenshot_path)
                }})

                logger.info(f"Step {{current_step}} completed successfully")

            except Exception as e:
                logger.error(f"Step {{current_step}} failed: {{e}}")

                # Take error screenshot
                error_screenshot = screenshot_dir / f"step_{{current_step}}_error.png"
                await page.screenshot(path=str(error_screenshot))

                step_results.append({{
                    "step": current_step,
                    "action": "{action_line}",
                    "status": "failed",
                    "error": str(e),
                    "error_screenshot": str(error_screenshot)
                }})

                logger.error(f"Stopping execution due to step {{current_step}} failure")
                break

            # Brief pause between steps
            await asyncio.sleep(0.5)
'''

    # Add completion and cleanup
    enhanced_script += '''
        finally:
            # Generate summary report
            logger.info("Generating execution summary...")

            success_count = sum(1 for r in step_results if r["status"] == "success")
            total_steps = len(step_results)

            logger.info(f"Execution Summary:")
            logger.info(f"- Total Steps: {total_steps}")
            logger.info(f"- Successful: {success_count}")
            logger.info(f"- Failed: {total_steps - success_count}")

            # Save detailed results
            results_file = screenshot_dir.parent / "execution_results.json"
            import json
            with open(results_file, 'w') as f:
                json.dump(step_results, f, indent=2)
            logger.info(f"Results saved to: {results_file}")

            # Close browser
            await browser.close()

            if success_count == total_steps:
                logger.info("🎉 All steps completed successfully!")
                return True
            else:
                logger.error(f"❌ Execution failed: {total_steps - success_count} steps failed")
                return False

if __name__ == "__main__":
    success = asyncio.run(run_automation())
    exit(0 if success else 1)
'''

    return enhanced_script


def _handle_done(webui_manager: WebuiManager, history: AgentHistoryList):
    """Callback when the agent finishes the task (success or failure)."""
    logger.info(
        f"Agent task finished. Duration: {history.total_duration_seconds():.2f}s, Tokens: {history.total_input_tokens()}"
    )

    # Generate DOCX report
    task = getattr(webui_manager.bu_agent, 'task', 'Unknown task') if webui_manager.bu_agent else 'Unknown task'
    docx_path = generate_docx_report(webui_manager, history, task)

    final_summary = "**Task Completed**\n"
    final_summary += f"- Duration: {history.total_duration_seconds():.2f} seconds\n"
    final_summary += f"- Total Input Tokens: {history.total_input_tokens()}\n"  # Or total tokens if available

    final_result = history.final_result()
    if final_result:
        final_summary += f"- Final Result: {final_result}\n"

    errors = history.errors()
    if errors and any(errors):
        final_summary += f"- **Errors:**\n```\n{errors}\n```\n"
    else:
        final_summary += "- Status: Success\n"

    if docx_path:
        final_summary += f"- 📄 **DOCX Report Generated:** {os.path.basename(docx_path)}\n"

    # Check if summary is already added to avoid duplicates
    if not any("**Task Completed**" in msg.get("content", "") for msg in webui_manager.bu_chat_history if msg.get("role") == "assistant"):
        webui_manager.bu_chat_history.append(
            {"role": "assistant", "content": final_summary}
        )


async def _ask_assistant_callback(
        webui_manager: WebuiManager, query: str, browser_context: BrowserContext
) -> Dict[str, Any]:
    """Callback triggered by the agent's ask_for_assistant action."""
    logger.info("Agent requires assistance. Waiting for user input.")

    if not hasattr(webui_manager, "_chat_history"):
        logger.error("Chat history not found in webui_manager during ask_assistant!")
        return {"response": "Internal Error: Cannot display help request."}

    webui_manager.bu_chat_history.append(
        {
            "role": "assistant",
            "content": f"**Need Help:** {query}\nPlease provide information or perform the required action in the browser, then type your response/confirmation below and click 'Submit Response'.",
        }
    )

    # Use state stored in webui_manager
    webui_manager.bu_response_event = asyncio.Event()
    webui_manager.bu_user_help_response = None  # Reset previous response

    try:
        logger.info("Waiting for user response event...")
        await asyncio.wait_for(
            webui_manager.bu_response_event.wait(), timeout=3600.0
        )  # Long timeout
        logger.info("User response event received.")
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for user assistance.")
        webui_manager.bu_chat_history.append(
            {
                "role": "assistant",
                "content": "**Timeout:** No response received. Trying to proceed.",
            }
        )
        webui_manager.bu_response_event = None  # Clear the event
        return {"response": "Timeout: User did not respond."}  # Inform the agent

    response = webui_manager.bu_user_help_response
    webui_manager.bu_chat_history.append(
        {"role": "user", "content": response}
    )  # Show user response in chat
    webui_manager.bu_response_event = (
        None  # Clear the event for the next potential request
    )
    return {"response": response}


# --- Core Agent Execution Logic --- (Needs access to webui_manager)


async def run_agent_task(
        webui_manager: WebuiManager, components: Dict[gr.components.Component, Any]
) -> AsyncGenerator[Dict[gr.components.Component, Any], None]:
    """Handles the entire lifecycle of initializing and running the agent."""

    # --- Get Components ---
    # Need handles to specific UI components to update them
    user_input_comp = webui_manager.get_component_by_id("browser_use_agent.user_input")
    run_button_comp = webui_manager.get_component_by_id("browser_use_agent.run_button")
    stop_button_comp = webui_manager.get_component_by_id(
        "browser_use_agent.stop_button"
    )
    pause_resume_button_comp = webui_manager.get_component_by_id(
        "browser_use_agent.pause_resume_button"
    )
    clear_button_comp = webui_manager.get_component_by_id(
        "browser_use_agent.clear_button"
    )
    chatbot_comp = webui_manager.get_component_by_id("browser_use_agent.chatbot")
    history_file_comp = webui_manager.get_component_by_id(
        "browser_use_agent.agent_history_file"
    )
    gif_comp = webui_manager.get_component_by_id("browser_use_agent.recording_gif")
    playwright_script_comp = webui_manager.get_component_by_id("browser_use_agent.playwright_script")
    docx_report_comp = webui_manager.get_component_by_id("browser_use_agent.docx_report")
    browser_view_comp = webui_manager.get_component_by_id(
        "browser_use_agent.browser_view"
    )

    # Initialize docx_path variable
    docx_path = None

    # --- Browser Settings (from environment variables) ---
    browser_binary_path = os.getenv("BROWSER_PATH") or None
    browser_user_data_dir = os.getenv("BROWSER_USER_DATA") or None
    use_own_browser = os.getenv("USE_OWN_BROWSER", "true").lower() == "true"
    keep_browser_open = os.getenv("KEEP_BROWSER_OPEN", "true").lower() == "true"
    headless = os.getenv("HEADLESS", "false").lower() == "true"
    disable_security = os.getenv("DISABLE_SECURITY", "false").lower() == "true"
    window_w = int(os.getenv("WINDOW_WIDTH", "1280"))
    window_h = int(os.getenv("WINDOW_HEIGHT", "1100"))
    cdp_url = os.getenv("BROWSER_CDP") or None
    wss_url = os.getenv("WSS_URL") or None
    save_recording_path = os.getenv("RECORDING_PATH") or None
    save_trace_path = os.getenv("TRACE_PATH") or None
    save_agent_history_path = os.getenv("AGENT_HISTORY_PATH", "./tmp/agent_history")
    save_download_path = os.getenv("DOWNLOADS_PATH", "./tmp/downloads")

    # Initialize directories early so file processing can use them
    os.makedirs(save_agent_history_path, exist_ok=True)
    if save_recording_path:
        os.makedirs(save_recording_path, exist_ok=True)
    if save_trace_path:
        os.makedirs(save_trace_path, exist_ok=True)
    if save_download_path:
        os.makedirs(save_download_path, exist_ok=True)

    # --- 1. Get Task and Initial UI Update ---
    task = components.get(user_input_comp, "").strip()
    if not task:
        gr.Warning("Please enter a task.")
        yield {run_button_comp: gr.update(interactive=True)}
        return

    # Generate task ID early for file processing
    webui_manager.bu_agent_task_id = str(uuid.uuid4())

    # Get uploaded context files
    context_files_comp = webui_manager.get_component_by_id("browser_use_agent.context_files")
    uploaded_files = components.get(context_files_comp, [])

    logger.info(f"Context files component: {context_files_comp}")
    logger.info(f"Uploaded files count: {len(uploaded_files) if uploaded_files else 0}")
    logger.info(f"Uploaded files type: {type(uploaded_files)}")

    # Process uploaded files for context
    context_info = ""
    if uploaded_files and len(uploaded_files) > 0:
        context_info = "\n\nContext files uploaded:"
        logger.info(f"Processing {len(uploaded_files)} uploaded files")

        for i, file_obj in enumerate(uploaded_files):
            logger.info(f"Processing file {i+1}: {file_obj}")
            logger.info(f"File object type: {type(file_obj)}")

            # Gradio file uploads are file paths, not file objects
            if isinstance(file_obj, str):
                # It's a file path from Gradio upload
                filename = os.path.basename(file_obj)
                logger.info(f"Gradio file path: {file_obj}")

                context_info += f"\n- {filename}"

                # Copy file to temporary location for agent access
                temp_dir = os.path.join(save_agent_history_path, webui_manager.bu_agent_task_id, "context_files")
                os.makedirs(temp_dir, exist_ok=True)
                temp_file_path = os.path.join(temp_dir, filename)

                try:
                    shutil.copy2(file_obj, temp_file_path)
                    context_info += f" (saved to {temp_file_path})"
                    logger.info(f"Saved context file: {temp_file_path}")

                    # Read first few lines for context (only for text files)
                    try:
                        # Check if it's a text file by extension
                        text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.csv', '.xml', '.yaml', '.yml'}
                        file_ext = os.path.splitext(filename)[1].lower()

                        if file_ext in text_extensions:
                            with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                first_lines = []
                                for line_num, line in enumerate(f):
                                    if line_num < 3:  # First 3 lines
                                        first_lines.append(line.strip())
                                    else:
                                        break
                                context_info += f"\n  Preview: {' | '.join(first_lines)}"
                        else:
                            # For binary files like .docx, .pdf, etc.
                            context_info += f"\n  Preview: [Binary file - {file_ext} format]"
                    except Exception as e:
                        logger.debug(f"Could not read file preview: {e}")
                        context_info += f"\n  Preview: [Could not preview file]"

                except Exception as e:
                    logger.error(f"Failed to save context file {filename}: {e}")
                    context_info += f" (failed to save: {e})"

            elif hasattr(file_obj, 'name') or hasattr(file_obj, 'filename'):
                # Fallback for other file object types
                filename = getattr(file_obj, 'name', getattr(file_obj, 'filename', 'unknown_file'))
                logger.info(f"File object filename: {filename}")

                context_info += f"\n- {filename}"

                # Save file to temporary location for agent access
                temp_dir = os.path.join(save_agent_history_path, webui_manager.bu_agent_task_id, "context_files")
                os.makedirs(temp_dir, exist_ok=True)
                temp_file_path = os.path.join(temp_dir, filename)

                try:
                    # Reset file pointer if possible
                    if hasattr(file_obj, 'seek'):
                        file_obj.seek(0)

                    with open(temp_file_path, 'wb') as f:
                        if hasattr(file_obj, 'read'):
                            content = file_obj.read()
                            f.write(content)
                        else:
                            logger.warning(f"File object has no read method: {filename}")

                    context_info += f" (saved to {temp_file_path})"
                    logger.info(f"Saved context file: {temp_file_path}")

                    # Read first few lines for context
                    try:
                        with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            first_lines = []
                            for line_num, line in enumerate(f):
                                if line_num < 3:  # First 3 lines
                                    first_lines.append(line.strip())
                                else:
                                    break
                            context_info += f"\n  Preview: {' | '.join(first_lines)}"
                    except Exception as e:
                        logger.debug(f"Could not read file preview: {e}")

                except Exception as e:
                    logger.error(f"Failed to save context file {filename}: {e}")
                    context_info += f" (failed to save: {e})"
            else:
                logger.warning(f"Unsupported file object type: {type(file_obj)}")
                context_info += f"\n- Unsupported file type: {type(file_obj)}"
    else:
        logger.info("No context files uploaded")
        context_info = "\n\nNo context files uploaded"

    logger.info(f"Final context_info: {context_info}")
    task += context_info

    # Load credentials and replace placeholders in task
    credentials = load_credentials_from_xml()
    logger.info(f"Loaded credentials for sites: {list(credentials.keys())}")
    original_task = task
    task = replace_credential_placeholders(task, credentials)
    if task != original_task:
        logger.info(f"Task modified with credentials: '{original_task}' -> '{task}'")
    else:
        logger.info(f"No credential replacement needed for task: '{task}'")

    # Set up secure credentials in environment variables
    credential_mapping = setup_secure_credentials(credentials)

    # Inject secure credential references into task
    task = inject_secure_credential_references(task, credential_mapping)

    # Set running state indirectly via _current_task
    webui_manager.bu_chat_history.append({"role": "user", "content": task})

    yield {
        user_input_comp: gr.Textbox(
            value="", interactive=False, placeholder="Agent is running..."
        ),
        run_button_comp: gr.Button(value="⏳ Running...", interactive=False),
        stop_button_comp: gr.Button(interactive=True),
        pause_resume_button_comp: gr.Button(value="⏸️ Pause", interactive=True),
        clear_button_comp: gr.Button(interactive=False),
        chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
        history_file_comp: gr.update(value=None),
        gif_comp: gr.update(value=None),
    }

    # --- Agent Settings (from environment variables) ---
    override_system_prompt = os.getenv("OVERRIDE_SYSTEM_PROMPT")
    extend_system_prompt = os.getenv("EXTEND_SYSTEM_PROMPT")
    llm_provider_name = os.getenv("DEFAULT_LLM", "ollama")
    llm_model_name = os.getenv("LLM_MODEL_NAME", "qwen2.5:7b")
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.6"))
    use_vision = os.getenv("USE_VISION", "false").lower() == "true"
    ollama_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16000"))
    llm_base_url = os.getenv("LLM_BASE_URL")
    llm_api_key = os.getenv("LLM_API_KEY")
    max_steps = int(os.getenv("MAX_STEPS", "25"))
    max_actions = int(os.getenv("MAX_ACTIONS", "3"))
    max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", "128000"))
    tool_calling_str = os.getenv("TOOL_CALLING_METHOD", "auto")
    # For OpenAI provider models (OpenRouter), use json_mode to avoid tool_choice issues
    if llm_provider_name == "openai":
        tool_calling_method = "json_mode"
    elif llm_model_name and llm_model_name.startswith("nvidia/"):
        tool_calling_method = "tools"
    else:
        tool_calling_method = tool_calling_str if tool_calling_str != "None" else None
    mcp_server_config_comp = webui_manager.id_to_component.get(
        "agent_settings.mcp_server_config"
    )
    mcp_server_config_str = (
        components.get(mcp_server_config_comp) if mcp_server_config_comp else None
    )
    mcp_server_config = (
        json.loads(mcp_server_config_str) if mcp_server_config_str else None
    )

    # Memory settings - from environment variables
    enable_memory = os.getenv("ENABLE_MEMORY", "false").lower() == "true"

    # Planner LLM Settings (Optional) - from environment variables
    planner_llm_provider_name = os.getenv("PLANNER_LLM_PROVIDER")
    planner_llm = None
    planner_use_vision = False
    if planner_llm_provider_name:
        planner_llm_model_name = os.getenv("PLANNER_LLM_MODEL_NAME")
        planner_llm_temperature = float(os.getenv("PLANNER_LLM_TEMPERATURE", "0.6"))
        planner_ollama_num_ctx = int(os.getenv("PLANNER_OLLAMA_NUM_CTX", "16000"))
        planner_llm_base_url = os.getenv("PLANNER_LLM_BASE_URL")
        planner_llm_api_key = os.getenv("PLANNER_LLM_API_KEY")
        planner_use_vision = os.getenv("PLANNER_USE_VISION", "false").lower() == "true"

        planner_llm = await _initialize_llm(
            planner_llm_provider_name,
            planner_llm_model_name,
            planner_llm_temperature,
            planner_llm_base_url,
            planner_llm_api_key,
            planner_ollama_num_ctx if planner_llm_provider_name == "ollama" else None,
        )

    # Truly dynamic prompt generation - analyzes task and generates adaptive rules
    def analyze_task_and_generate_rules(task_text: str) -> str:
        """Analyze the task and generate truly dynamic, context-aware automation rules."""

        # Parse task components to understand what needs to be done
        task_lower = task_text.lower()

        # Extract key actions and objects from the task
        actions = []
        objects = []

        # Action keywords
        action_keywords = ['login', 'click', 'fill', 'submit', 'search', 'find', 'extract', 'navigate', 'select', 'add', 'remove', 'sort', 'filter']
        for keyword in action_keywords:
            if keyword in task_lower:
                actions.append(keyword)

        # Object keywords
        object_keywords = ['form', 'button', 'input', 'field', 'cart', 'product', 'item', 'page', 'link', 'menu', 'table', 'data']
        for keyword in object_keywords:
            if keyword in task_lower:
                objects.append(keyword)

        # Generate dynamic rules based on detected actions and objects
        rules = ["IMPORTANT: Stay focused on the exact task requested. Do not perform additional actions beyond what is specifically asked."]

        # Dynamic rule generation based on task analysis
        if 'multiple' in task_lower or 'all' in task_lower or any(word in task_lower for word in ['2', '3', '4', '5', 'several']):
            rules.append("MULTI-ITEM TASK: When handling multiple items, maintain your position on the main listing page and avoid navigation to individual item pages unless specifically required.")

        if any(action in actions for action in ['login', 'submit', 'fill']):
            rules.append("FORM INTERACTION: For form-based actions, use text-based element selection and complete all required fields before submission.")

        if any(obj in objects for obj in ['cart', 'product', 'item']):
            rules.append("COMMERCE WORKFLOW: Stay on inventory/product listing pages when adding items - avoid individual product page navigation.")

        if any(action in actions for action in ['search', 'find', 'filter', 'sort']):
            rules.append("DATA OPERATIONS: Use available filters and sorting options to organize data before performing actions on specific items.")

        if any(obj in objects for obj in ['table', 'data', 'information']):
            rules.append("DATA HANDLING: Extract or manipulate data systematically, ensuring you're on the correct page with the target information.")

        # Universal web automation principles (always included)
        rules.extend([
            "",
            "UNIVERSAL WEB AUTOMATION PRINCIPLES:",
            "- Prefer text-based element selection over positional/index-based selection for reliability",
            "- Use descriptive identifiers (button text, field labels, link text) over generic selectors",
            "- If elements aren't visible, try scrolling, waiting, or zooming to reveal content",
            "- Maintain context awareness - know which page/section you're currently on",
            "- Handle dynamic content by using stable, descriptive selectors",
            "- When actions fail, adapt by trying alternative approaches or selectors",
            "- Wait for page loads and element interactions to complete naturally",
            "- Use context clues (surrounding text, positioning) to identify correct elements among similar ones"
        ])

        return "\n".join(rules)

    # Generate dynamic rules based on task content
    prompt_addition = "\n\n" + analyze_task_and_generate_rules(task)
    if extend_system_prompt:
        extend_system_prompt += prompt_addition
    else:
        extend_system_prompt = prompt_addition

    # --- Browser Settings (from environment variables) ---
    browser_binary_path = os.getenv("BROWSER_PATH") or None
    browser_user_data_dir = os.getenv("BROWSER_USER_DATA") or None
    use_own_browser = os.getenv("USE_OWN_BROWSER", "true").lower() == "true"
    keep_browser_open = os.getenv("KEEP_BROWSER_OPEN", "true").lower() == "true"
    headless = os.getenv("HEADLESS", "false").lower() == "true"
    disable_security = os.getenv("DISABLE_SECURITY", "false").lower() == "true"
    window_w = int(os.getenv("WINDOW_WIDTH", "1280"))
    window_h = int(os.getenv("WINDOW_HEIGHT", "1100"))
    cdp_url = os.getenv("BROWSER_CDP") or None
    wss_url = os.getenv("WSS_URL") or None
    save_recording_path = os.getenv("RECORDING_PATH") or None
    save_trace_path = os.getenv("TRACE_PATH") or None
    save_agent_history_path = os.getenv("AGENT_HISTORY_PATH", "./tmp/agent_history")
    save_download_path = os.getenv("DOWNLOADS_PATH", "./tmp/downloads")

    # Initialize directories early so file processing can use them
    os.makedirs(save_agent_history_path, exist_ok=True)
    if save_recording_path:
        os.makedirs(save_recording_path, exist_ok=True)
    if save_trace_path:
        os.makedirs(save_trace_path, exist_ok=True)
    if save_download_path:
        os.makedirs(save_download_path, exist_ok=True)

    stream_vw = 70
    stream_vh = int(70 * window_h // window_w)

    # --- 2. Initialize LLM ---
    main_llm = await _initialize_llm(
        llm_provider_name,
        llm_model_name,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        ollama_num_ctx if llm_provider_name == "ollama" else None,
    )

    # Pass the webui_manager instance to the callback when wrapping it
    async def ask_callback_wrapper(
            query: str, browser_context: BrowserContext
    ) -> Dict[str, Any]:
        return await _ask_assistant_callback(webui_manager, query, browser_context)

    if not webui_manager.bu_controller:
        webui_manager.bu_controller = CustomController(
            ask_assistant_callback=ask_callback_wrapper
        )
        await webui_manager.bu_controller.setup_mcp_client(mcp_server_config)

    # --- 4. Initialize Browser and Context ---
    should_close_browser_on_finish = not keep_browser_open

    try:
        # Always close existing resources to prevent state confusion
        logger.info("Closing any existing browser resources to prevent state confusion.")
        if webui_manager.bu_browser_context:
            logger.info("Closing previous browser context.")
            await webui_manager.bu_browser_context.close()
            webui_manager.bu_browser_context = None
        if webui_manager.bu_browser:
            logger.info("Closing previous browser.")
            await webui_manager.bu_browser.close()
            webui_manager.bu_browser = None

        # Small delay to ensure resources are fully cleaned up
        await asyncio.sleep(0.5)

        # Create Browser if needed
        if not webui_manager.bu_browser:
            logger.info("Launching new browser instance.")
            extra_args = ["--start-maximized"]  # Default to full screen
            if use_own_browser:
                browser_binary_path = os.getenv("BROWSER_PATH", None) or browser_binary_path
                if browser_binary_path == "":
                    browser_binary_path = None

                # Log which browser path we're using
                if browser_binary_path:
                    logger.info(f"Using configured BROWSER_PATH: {browser_binary_path}")
                else:
                    # Auto-detect Chrome browser if no path specified
                    browser_binary_path = _detect_browser()
                    if browser_binary_path:
                        logger.info(f"Using auto-detected browser: {browser_binary_path}")

                browser_user_data = browser_user_data_dir or os.getenv("BROWSER_USER_DATA", None)
                if browser_user_data:
                    extra_args += [f"--user-data-dir={browser_user_data}"]
            else:
                browser_binary_path = None

            webui_manager.bu_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    browser_binary_path=browser_binary_path,
                    extra_browser_args=extra_args,
                    wss_url=wss_url,
                    cdp_url=cdp_url,
                    new_context_config=BrowserContextConfig(
                        window_width=window_w,
                        window_height=window_h,
                    )
                )
            )

        # Create Context if needed
        if not webui_manager.bu_browser_context:
            logger.info("Creating new browser context.")
            context_config = BrowserContextConfig(
                trace_path=save_trace_path if save_trace_path else None,
                save_recording_path=save_recording_path
                if save_recording_path
                else None,
                save_downloads_path=save_download_path if save_download_path else None,
                window_height=window_h,
                window_width=window_w,
            )
            if not webui_manager.bu_browser:
                raise ValueError("Browser not initialized, cannot create context.")
            webui_manager.bu_browser_context = (
                await webui_manager.bu_browser.new_context(config=context_config)
            )

        # --- 5. Initialize or Update Agent ---
        webui_manager.bu_agent_task_id = str(uuid.uuid4())  # New ID for this task run
        os.makedirs(
            os.path.join(save_agent_history_path, webui_manager.bu_agent_task_id),
            exist_ok=True,
        )
        history_file = os.path.join(
            save_agent_history_path,
            webui_manager.bu_agent_task_id,
            f"{webui_manager.bu_agent_task_id}.json",
        )
        video_path = os.path.join(
            save_agent_history_path,
            webui_manager.bu_agent_task_id,
            f"{webui_manager.bu_agent_task_id}.mp4",
        )
        playwright_script_path = os.path.join(
            save_agent_history_path,
            webui_manager.bu_agent_task_id,
            f"{webui_manager.bu_agent_task_id}_playwright.py",
        )

        # Reset token counter for new task
        webui_manager.bu_previous_tokens = 0

        # Pass the webui_manager to callbacks when wrapping them
        async def step_callback_wrapper(
                state: BrowserState, output: AgentOutput, step_num: int
        ):
            await _handle_new_step(webui_manager, state, output, step_num)

        def done_callback_wrapper(history: AgentHistoryList):
            _handle_done(webui_manager, history)

        if not webui_manager.bu_agent:
            logger.info(f"Initializing new agent for task: {task}")
            if not webui_manager.bu_browser or not webui_manager.bu_browser_context:
                raise ValueError(
                    "Browser or Context not initialized, cannot create agent."
                )
            webui_manager.bu_agent = BrowserUseAgent(
                task=task,
                llm=main_llm,
                browser=webui_manager.bu_browser,
                browser_context=webui_manager.bu_browser_context,
                controller=webui_manager.bu_controller,
                register_new_step_callback=step_callback_wrapper,
                register_done_callback=done_callback_wrapper,
                use_vision=use_vision,
                override_system_message=override_system_prompt,
                extend_system_message=extend_system_prompt,
                max_input_tokens=max_input_tokens,
                max_actions_per_step=max_actions,
                tool_calling_method=tool_calling_method,
                planner_llm=planner_llm,
                use_vision_for_planner=planner_use_vision if planner_llm else False,
                source="webui",
            )
            webui_manager.bu_agent.state.agent_id = webui_manager.bu_agent_task_id
            webui_manager.bu_agent.settings.generate_gif = video_path
            webui_manager.bu_agent.settings.save_playwright_script_path = playwright_script_path
        else:
            webui_manager.bu_agent.state.agent_id = webui_manager.bu_agent_task_id
            webui_manager.bu_agent.add_new_task(task)
            webui_manager.bu_agent.settings.generate_gif = video_path
            webui_manager.bu_agent.browser = webui_manager.bu_browser
            webui_manager.bu_agent.browser_context = webui_manager.bu_browser_context
            webui_manager.bu_agent.controller = webui_manager.bu_controller

        # --- 6. Run Agent Task and Stream Updates ---
        agent_run_coro = webui_manager.bu_agent.run(max_steps=max_steps)
        agent_task = asyncio.create_task(agent_run_coro)
        webui_manager.bu_current_task = agent_task  # Store the task

        last_chat_len = len(webui_manager.bu_chat_history)
        while not agent_task.done():
            is_paused = webui_manager.bu_agent.state.paused
            is_stopped = webui_manager.bu_agent.state.stopped

            # Check for pause state
            if is_paused:
                yield {
                    pause_resume_button_comp: gr.update(
                        value="▶️ Resume", interactive=True
                    ),
                    stop_button_comp: gr.update(interactive=True),
                }
                # Wait until pause is released or task is stopped/done
                while is_paused and not agent_task.done():
                    # Re-check agent state in loop
                    is_paused = webui_manager.bu_agent.state.paused
                    is_stopped = webui_manager.bu_agent.state.stopped
                    if is_stopped:  # Stop signal received while paused
                        break
                    await asyncio.sleep(0.2)

                if (
                        agent_task.done() or is_stopped
                ):  # If stopped or task finished while paused
                    break

                # If resumed, yield UI update
                yield {
                    pause_resume_button_comp: gr.update(
                        value="⏸️ Pause", interactive=True
                    ),
                    run_button_comp: gr.update(
                        value="⏳ Running...", interactive=False
                    ),
                }

            # Check if agent stopped itself or stop button was pressed (which sets agent.state.stopped)
            if is_stopped:
                logger.info("Agent has stopped (internally or via stop button).")
                if not agent_task.done():
                    # Ensure the task coroutine finishes if agent just set flag
                    try:
                        await asyncio.wait_for(
                            agent_task, timeout=1.0
                        )  # Give it a moment to exit run()
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Agent task did not finish quickly after stop signal, cancelling."
                        )
                        agent_task.cancel()
                    except Exception:  # Catch task exceptions if it errors on stop
                        pass
                break  # Exit the streaming loop

            # Check if agent is asking for help (via response_event)
            update_dict = {}
            if webui_manager.bu_response_event is not None:
                update_dict = {
                    user_input_comp: gr.update(
                        placeholder="Agent needs help. Enter response and submit.",
                        interactive=True,
                    ),
                    run_button_comp: gr.update(
                        value="✔️ Submit Response", interactive=True
                    ),
                    pause_resume_button_comp: gr.update(interactive=False),
                    stop_button_comp: gr.update(interactive=False),
                    chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
                }
                last_chat_len = len(webui_manager.bu_chat_history)
                yield update_dict
                # Wait until response is submitted or task finishes
                await webui_manager.bu_response_event.wait()

                # Restore UI after response submitted or if task ended unexpectedly
                if not agent_task.done():
                    yield {
                        user_input_comp: gr.update(
                            placeholder="Agent is running...", interactive=False
                        ),
                        run_button_comp: gr.update(
                            value="⏳ Running...", interactive=False
                        ),
                        pause_resume_button_comp: gr.update(interactive=True),
                        stop_button_comp: gr.update(interactive=True),
                    }
                else:
                    break  # Task finished while waiting for response

            # Update Chatbot if new messages arrived via callbacks
            if len(webui_manager.bu_chat_history) > last_chat_len:
                update_dict[chatbot_comp] = gr.update(
                    value=webui_manager.bu_chat_history
                )
                last_chat_len = len(webui_manager.bu_chat_history)

            # Update Browser View
            if headless and webui_manager.bu_browser_context:
                try:
                    screenshot_b64 = (
                        await webui_manager.bu_browser_context.take_screenshot()
                    )
                    if screenshot_b64:
                        html_content = f'<img src="data:image/jpeg;base64,{screenshot_b64}" style="width:{stream_vw}vw; height:{stream_vh}vh ; border:1px solid #ccc;">'
                        update_dict[browser_view_comp] = gr.update(
                            value=html_content, visible=True
                        )
                    else:
                        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                        update_dict[browser_view_comp] = gr.update(
                            value=html_content, visible=True
                        )
                except Exception as e:
                    logger.debug(f"Failed to capture screenshot: {e}")
                    update_dict[browser_view_comp] = gr.update(
                        value="<div style='...'>Error loading view...</div>",
                        visible=True,
                    )
            else:
                update_dict[browser_view_comp] = gr.update(visible=False)

            # Yield accumulated updates
            if update_dict:
                yield update_dict

            await asyncio.sleep(0.5)  # Polling interval - increased to reduce API calls

        # --- 7. Task Finalization ---
        webui_manager.bu_agent.state.paused = False
        webui_manager.bu_agent.state.stopped = False
        final_update = {}
        try:
            logger.info("Agent task completing...")
            # Await the task ensure completion and catch exceptions if not already caught
            if not agent_task.done():
                await agent_task  # Retrieve result/exception
            elif agent_task.exception():  # Check if task finished with exception
                agent_task.result()  # Raise the exception to be caught below
            logger.info("Agent task completed processing.")

            logger.info(f"Explicitly saving agent history to: {history_file}")
            webui_manager.bu_agent.save_history(history_file)

            if os.path.exists(history_file):
                final_update[history_file_comp] = gr.File(value=history_file)

            if gif_path and os.path.exists(gif_path):
                logger.info(f"GIF found at: {gif_path}")
                final_update[gif_comp] = gr.Image(value=gif_path)

            if playwright_script_path and os.path.exists(playwright_script_path):
                logger.info(f"Playwright script found at: {playwright_script_path}")
                try:
                    with open(playwright_script_path, 'r', encoding='utf-8') as f:
                        script_content = f.read()

                    # Enhance the Playwright script with sequential execution and error handling
                    enhanced_script = enhance_playwright_script(script_content, webui_manager.bu_agent_task_id)

                    final_update[playwright_script_comp] = gr.Code(value=enhanced_script, language="python")
                except Exception as e:
                    logger.error(f"Failed to read/enhance Playwright script: {e}")
                    final_update[playwright_script_comp] = gr.Code(value=f"Error reading script: {e}", language="text")

            if docx_path and os.path.exists(docx_path):
                logger.info(f"DOCX report found at: {docx_path}")
                final_update[docx_report_comp] = gr.File(value=docx_path)

        except asyncio.CancelledError:
            logger.info("Agent task was cancelled.")
            if not any(
                    "Cancelled" in msg.get("content", "")
                    for msg in webui_manager.bu_chat_history
                    if msg.get("role") == "assistant"
            ):
                webui_manager.bu_chat_history.append(
                    {"role": "assistant", "content": "**Task Cancelled**."}
                )
            final_update[chatbot_comp] = gr.update(value=webui_manager.bu_chat_history)
        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            error_message = (
                f"**Agent Execution Error:**\n```\n{type(e).__name__}: {e}\n```"
            )
            if not any(
                    error_message in msg.get("content", "")
                    for msg in webui_manager.bu_chat_history
                    if msg.get("role") == "assistant"
            ):
                webui_manager.bu_chat_history.append(
                    {"role": "assistant", "content": error_message}
                )
            # Show task summary even on error
            if webui_manager.bu_agent and hasattr(webui_manager.bu_agent, 'state') and webui_manager.bu_agent.state.history:
                _handle_done(webui_manager, webui_manager.bu_agent.state.history)
            final_update[chatbot_comp] = gr.update(value=webui_manager.bu_chat_history)
            gr.Error(f"Agent execution failed: {e}")

        finally:
            webui_manager.bu_current_task = None  # Clear the task reference

            # Close browser/context if requested
            if should_close_browser_on_finish:
                if webui_manager.bu_browser_context:
                    logger.info("Closing browser context after task.")
                    await webui_manager.bu_browser_context.close()
                    webui_manager.bu_browser_context = None
                if webui_manager.bu_browser:
                    logger.info("Closing browser after task.")
                    await webui_manager.bu_browser.close()
                    webui_manager.bu_browser = None

            # --- 8. Final UI Update ---
            final_update.update(
                {
                    user_input_comp: gr.update(
                        value="",
                        interactive=True,
                        placeholder="Enter your next task...",
                    ),
                    run_button_comp: gr.update(value="▶️ Submit Task", interactive=True),
                    stop_button_comp: gr.update(value="⏹️ Stop", interactive=False),
                    pause_resume_button_comp: gr.update(
                        value="⏸️ Pause", interactive=False
                    ),
                    clear_button_comp: gr.update(interactive=True),
                    # Ensure final chat history is shown
                    chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
                }
            )
            yield final_update

    except Exception as e:
        # Catch errors during setup (before agent run starts)
        logger.error(f"Error setting up agent task: {e}", exc_info=True)
        webui_manager.bu_current_task = None  # Ensure state is reset
        yield {
            user_input_comp: gr.update(
                interactive=True, placeholder="Error during setup. Enter task..."
            ),
            run_button_comp: gr.update(value="▶️ Submit Task", interactive=True),
            stop_button_comp: gr.update(value="⏹️ Stop", interactive=False),
            pause_resume_button_comp: gr.update(value="⏸️ Pause", interactive=False),
            clear_button_comp: gr.update(interactive=True),
            chatbot_comp: gr.update(
                value=webui_manager.bu_chat_history
                      + [{"role": "assistant", "content": f"**Setup Error:** {e}"}]
            ),
        }


# --- Button Click Handlers --- (Need access to webui_manager)


async def handle_submit(
        webui_manager: WebuiManager, components: Dict[gr.components.Component, Any]
):
    """Handles clicks on the main 'Submit' button."""
    user_input_comp = webui_manager.get_component_by_id("browser_use_agent.user_input")
    user_input_value = components.get(user_input_comp, "").strip()

    # Check if waiting for user assistance
    if webui_manager.bu_response_event and not webui_manager.bu_response_event.is_set():
        logger.info(f"User submitted assistance: {user_input_value}")
        webui_manager.bu_user_help_response = (
            user_input_value if user_input_value else "User provided no text response."
        )
        webui_manager.bu_response_event.set()
        # UI updates handled by the main loop reacting to the event being set
        yield {
            user_input_comp: gr.update(
                value="",
                interactive=False,
                placeholder="Waiting for agent to continue...",
            ),
            webui_manager.get_component_by_id(
                "browser_use_agent.run_button"
            ): gr.update(value="⏳ Running...", interactive=False),
        }
    # Check if a task is currently running (using _current_task)
    elif webui_manager.bu_current_task and not webui_manager.bu_current_task.done():
        logger.warning(
            "Submit button clicked while agent is already running and not asking for help."
        )
        gr.Info("Agent is currently running. Please wait or use Stop/Pause.")
        yield {}  # No change
    else:
        # Handle submission for a new task
        logger.info("Submit button clicked for new task.")
        # Use async generator to stream updates from run_agent_task
        async for update in run_agent_task(webui_manager, components):
            yield update


async def handle_stop(webui_manager: WebuiManager):
    """Handles clicks on the 'Stop' button."""
    logger.info("Stop button clicked.")
    agent = webui_manager.bu_agent
    task = webui_manager.bu_current_task

    if agent and task and not task.done():
        # Signal the agent to stop by setting its internal flag
        agent.state.stopped = True
        agent.state.paused = False  # Ensure not paused if stopped
        return {
            webui_manager.get_component_by_id(
                "browser_use_agent.stop_button"
            ): gr.update(interactive=False, value="⏹️ Stopping..."),
            webui_manager.get_component_by_id(
                "browser_use_agent.pause_resume_button"
            ): gr.update(interactive=False),
            webui_manager.get_component_by_id(
                "browser_use_agent.run_button"
            ): gr.update(interactive=False),
        }
    else:
        logger.warning("Stop clicked but agent is not running or task is already done.")
        # Reset UI just in case it's stuck
        return {
            webui_manager.get_component_by_id(
                "browser_use_agent.run_button"
            ): gr.update(interactive=True),
            webui_manager.get_component_by_id(
                "browser_use_agent.stop_button"
            ): gr.update(interactive=False),
            webui_manager.get_component_by_id(
                "browser_use_agent.pause_resume_button"
            ): gr.update(interactive=False),
            webui_manager.get_component_by_id(
                "browser_use_agent.clear_button"
            ): gr.update(interactive=True),
        }


async def handle_pause_resume(webui_manager: WebuiManager):
    """Handles clicks on the 'Pause/Resume' button."""
    agent = webui_manager.bu_agent
    task = webui_manager.bu_current_task

    if agent and task and not task.done():
        if agent.state.paused:
            logger.info("Resume button clicked.")
            agent.resume()
            # UI update happens in main loop
            return {
                webui_manager.get_component_by_id(
                    "browser_use_agent.pause_resume_button"
                ): gr.update(value="⏸️ Pause", interactive=True)
            }  # Optimistic update
        else:
            logger.info("Pause button clicked.")
            agent.pause()
            return {
                webui_manager.get_component_by_id(
                    "browser_use_agent.pause_resume_button"
                ): gr.update(value="▶️ Resume", interactive=True)
            }  # Optimistic update
    else:
        logger.warning(
            "Pause/Resume clicked but agent is not running or doesn't support state."
        )
        return {}  # No change


async def handle_clear(webui_manager: WebuiManager):
    """Handles clicks on the 'Clear' button."""
    logger.info("Clear button clicked.")

    # Stop any running task first
    task = webui_manager.bu_current_task
    if task and not task.done():
        logger.info("Clearing requires stopping the current task.")
        webui_manager.bu_agent.stop()
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)  # Wait briefly
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception as e:
            logger.warning(f"Error stopping task on clear: {e}")
    webui_manager.bu_current_task = None

    if webui_manager.bu_controller:
        await webui_manager.bu_controller.close_mcp_client()
        webui_manager.bu_controller = None
    webui_manager.bu_agent = None

    # Reset state stored in manager
    webui_manager.bu_chat_history = []
    webui_manager.bu_response_event = None
    webui_manager.bu_user_help_response = None
    webui_manager.bu_agent_task_id = None

    logger.info("Agent state and browser resources cleared.")

    # Reset UI components
    return {
        webui_manager.get_component_by_id("browser_use_agent.chatbot"): gr.update(
            value=[]
        ),
        webui_manager.get_component_by_id("browser_use_agent.user_input"): gr.update(
            value="", placeholder="Enter your task here..."
        ),
        webui_manager.get_component_by_id(
            "browser_use_agent.agent_history_file"
        ): gr.update(value=None),
        webui_manager.get_component_by_id("browser_use_agent.recording_gif"): gr.update(
            value=None
        ),
        webui_manager.get_component_by_id("browser_use_agent.browser_view"): gr.update(
            value="<div style='...'>Browser Cleared</div>"
        ),
        webui_manager.get_component_by_id("browser_use_agent.run_button"): gr.update(
            value="▶️ Submit Task", interactive=True
        ),
        webui_manager.get_component_by_id("browser_use_agent.stop_button"): gr.update(
            interactive=False
        ),
        webui_manager.get_component_by_id(
            "browser_use_agent.pause_resume_button"
        ): gr.update(value="⏸️ Pause", interactive=False),
        webui_manager.get_component_by_id("browser_use_agent.clear_button"): gr.update(
            interactive=True
        ),
    }


# --- Tab Creation Function ---


def create_browser_use_agent_tab(webui_manager: WebuiManager):
    """
    Create the run agent tab, defining UI, state, and handlers.
    """
    webui_manager.init_browser_use_agent()

    # --- Define UI Components ---
    tab_components = {}
    with gr.Column():
        user_input = gr.Textbox(
            label="Your Task or Response",
            placeholder="Enter your task here or provide assistance when asked.",
            lines=3,
            interactive=True,
            elem_id="user_input",
        )

        # File upload for context
        context_files = gr.File(
            label="Upload Context Files (PDF, TXT, etc.)",
            file_types=[".pdf", ".txt", ".docx", ".md", ".csv", ".json"],
            file_count="multiple",
            interactive=True,
            elem_id="context_files",
        )

        with gr.Row():
            stop_button = gr.Button(
                "⏹️ Stop", interactive=False, variant="stop", scale=2
            )
            pause_resume_button = gr.Button(
                "⏸️ Pause", interactive=False, variant="secondary", scale=2, visible=True
            )
            clear_button = gr.Button(
                "🗑️ Clear", interactive=True, variant="secondary", scale=2
            )
            run_button = gr.Button("▶️ Submit Task", variant="primary", scale=3)

        chatbot = gr.Chatbot(
            lambda: webui_manager.bu_chat_history,  # Load history dynamically
            elem_id="browser_use_chatbot",
            label="Agent Interaction",
            type="messages",
            height=600,
            show_copy_button=True,
        )

        browser_view = gr.HTML(
            value="<div style='width:100%; height:50vh; display:flex; justify-content:center; align-items:center; border:1px solid #ccc; background-color:#f0f0f0;'><p>Browser View (Requires Headless=True)</p></div>",
            label="Browser Live View",
            elem_id="browser_view",
            visible=False,
        )
        with gr.Column():
            gr.Markdown("### Task Outputs")
            agent_history_file = gr.File(label="Agent History JSON", interactive=False)
            recording_gif = gr.Video(
                label="Task Recording MP4",
                format="mp4",
                interactive=False,
            )
            playwright_script = gr.Code(
                label="Generated Playwright Script",
                language="python",
                interactive=False,
                lines=20,
            )
            docx_report = gr.File(label="DOCX Report", interactive=False)

    # --- Store Components in Manager ---
    tab_components.update(
        dict(
            chatbot=chatbot,
            user_input=user_input,
            context_files=context_files,
            clear_button=clear_button,
            run_button=run_button,
            stop_button=stop_button,
            pause_resume_button=pause_resume_button,
            agent_history_file=agent_history_file,
            recording_gif=recording_gif,
            browser_view=browser_view,
            playwright_script=playwright_script,
            docx_report=docx_report,
        )
    )
    webui_manager.add_components(
        "browser_use_agent", tab_components
    )  # Use "browser_use_agent" as tab_name prefix

    all_managed_components = set(
        webui_manager.get_components()
    )  # Get all components known to manager
    run_tab_outputs = list(tab_components.values())

    async def submit_wrapper(
            components_dict: Dict[Component, Any],
    ) -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_submit that yields its results."""
        async for update in handle_submit(webui_manager, components_dict):
            yield update

    async def stop_wrapper() -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_stop."""
        update_dict = await handle_stop(webui_manager)
        yield update_dict

    async def pause_resume_wrapper() -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_pause_resume."""
        update_dict = await handle_pause_resume(webui_manager)
        yield update_dict

    async def clear_wrapper() -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_clear."""
        update_dict = await handle_clear(webui_manager)
        yield update_dict

    # --- Connect Event Handlers using the Wrappers --
    run_button.click(
        fn=submit_wrapper, inputs=all_managed_components, outputs=run_tab_outputs, trigger_mode="multiple"
    )
    user_input.submit(
        fn=submit_wrapper, inputs=all_managed_components, outputs=run_tab_outputs
    )
    stop_button.click(fn=stop_wrapper, inputs=None, outputs=run_tab_outputs)
    pause_resume_button.click(
        fn=pause_resume_wrapper, inputs=None, outputs=run_tab_outputs
    )
    clear_button.click(fn=clear_wrapper, inputs=None, outputs=run_tab_outputs)
