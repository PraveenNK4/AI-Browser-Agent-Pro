import logging
import inspect
import asyncio
import os
from typing import Optional, Type, Callable, Dict, Any, Union, Awaitable, TypeVar
from pydantic import BaseModel
import builtins
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller
from browser_use.controller.registry.service import Registry, RegisteredAction
from langchain_core.language_models.chat_models import BaseChatModel
from browser_use.agent.views import ActionModel, ActionResult
from src.utils.mcp_client import create_tool_param_model, setup_mcp_client_and_tools
from browser_use.utils import time_execution_sync
from src.utils.dom_snapshot import capture_dom_snapshot
from src.utils.comprehensive_element_capture import ComprehensiveElementCapture
from pathlib import Path
from src.utils.config import VAULT_CREDENTIAL_PREFIX

logger = logging.getLogger(__name__)

# GLOBAL SAFETY: Make logger available in builtins to prevent
# 'logger is not defined' errors in nested closures or callbacks.
# This is a defensive measure for edge cases where scope is lost.
if not hasattr(builtins, 'logger'):
    builtins.logger = logger

Context = TypeVar('Context')


class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [],
                 output_model: Optional[Type[BaseModel]] = None,
                 ask_assistant_callback: Optional[Union[Callable[[str, BrowserContext], Dict[str, Any]], Callable[
                     [str, BrowserContext], Awaitable[Dict[str, Any]]]]] = None,
                 extraction_llm: Optional[BaseChatModel] = None,
                 ):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()
        self.ask_assistant_callback = ask_assistant_callback
        self.extraction_llm = extraction_llm
        
        # Initialize comprehensive element capture
        element_capture_dir = Path("tmp/element_data")
        self.element_capturer = ComprehensiveElementCapture(output_dir=element_capture_dir)
        logger.info(f"✓ Comprehensive element capture initialized: {element_capture_dir}")
        self.extraction_llm = extraction_llm
        self.mcp_client = None
        self.mcp_server_config = None
        self.state_history = []
        self.sensitive_data = {}
        self.available_file_paths = []  # Track available files for upload
        self.last_page_with_tables = None
        
        # Track consecutive action failures for smarter error guidance
        self.action_failure_tracker = {}  # {(action_name, param_key): failure_count}
        
        # BITNET MODE: Enable aggressive optimizations for ternary models
        self.bitnet_mode = False

    async def _execute_smart_login(self, browser, username: str, password: str, button_text: str = "Sign in"):
        """Internal helper to execute smart login when auto-detected from input_text."""
        try:
            page = await browser.get_current_page()
            
            # FULLY GENERIC: Dynamic selectors for any login form
            # Priority order: specific patterns first, generic fallbacks last
            un_selectors = [
                'input[autocomplete="username"]',
                'input[name*="user" i]',
                'input[id*="user" i]',
                'input[name*="login" i]',
                'input[id*="login" i]',
                'input[name*="email" i]',
                'input[type="email"]',
                'input[type="text"]:visible'  # Last resort: first visible text input
            ]
            pw_selectors = [
                'input[type="password"]:visible',
                'input[autocomplete="current-password"]',
                'input[name*="pass" i]',
                'input[id*="pass" i]'
            ]
            
            # Fill username - find first matching visible field
            username_filled = False
            for selector in un_selectors:
                try:
                    el = page.locator(selector).first
                    if await el.count() > 0 and await el.is_visible(timeout=1000):
                        await el.fill(username)
                        logger.info(f"✓ Filled username into {selector}")
                        username_filled = True
                        break
                except:
                    continue
            
            if not username_filled:
                logger.warning("⚠ Could not find username field")
            
            # Fill password - find first matching visible field
            password_filled = False
            for selector in pw_selectors:
                try:
                    el = page.locator(selector).first
                    if await el.count() > 0 and await el.is_visible(timeout=1000):
                        await el.fill(password)
                        logger.info(f"✓ Filled password into {selector}")
                        password_filled = True
                        break
                except:
                    continue
            
            if not password_filled:
                logger.warning("⚠ Could not find password field")
            
            # Click login button - try multiple patterns
            btn_selectors = [
                f'button:has-text("{button_text}")',
                f'a:has-text("{button_text}")',
                'button:has-text("Log in")',
                'button:has-text("Login")',
                'button:has-text("Submit")',
                'button[type="submit"]',
                'input[type="submit"]',
                'button:visible >> nth=0'  # Last resort: first visible button
            ]
            
            button_clicked = False
            for selector in btn_selectors:
                try:
                    el = page.locator(selector).first
                    if await el.count() > 0 and await el.is_visible(timeout=1000):
                        await el.click()
                        logger.info(f"✓ Clicked login button: {selector}")
                        button_clicked = True
                        break
                except:
                    continue
            
            if not button_clicked:
                logger.warning("⚠ Could not find login button")
            
            # Wait for navigation
            await page.wait_for_load_state("networkidle", timeout=10000)
            
            return ActionResult(
                extracted_content="✅ AUTO-LOGIN COMPLETE (smart_login forced from input_text). Login sequence executed atomically.",
                include_in_memory=True
            )
        except Exception as e:
            return ActionResult(error=f"Auto-login failed: {str(e)}")

    def _register_custom_actions(self):
        """Register all custom browser actions"""

        @self.registry.action(
            "Retrieve value from element. Use for all data extraction (retrieve, read, check, validate, status)."
        )
        async def retrieve_value_by_element(
            index: int,
            browser: BrowserContext,
        ):
            try:
                dom_el = await browser.get_dom_element_by_index(index)
                if not dom_el:
                    return ActionResult(error=f"No element found at index {index}")

                element = await browser.get_locate_element(dom_el)
                if not element:
                    return ActionResult(error=f"Unable to locate element at index {index}")

                # Extract text/value with fallbacks
                value = (
                    await element.text_content()
                    or await element.inner_text()
                    or await element.get_attribute("value")
                    or await element.get_attribute("aria-label")
                    or await element.get_attribute("title")
                    or await element.get_attribute("alt")
                    or ""
                )

                value = value.strip()
                
                # Check SVG/image status indicators if no text
                if not value:
                    page = await browser.get_current_page()
                    svg_data = await page.evaluate(f"""
                        (function() {{
                            const el = document.evaluate(
                                "//*[@data-browser-use-index='{index}']",
                                document, null,
                                XPathResult.FIRST_ORDERED_NODE_TYPE, null
                            ).singleNodeValue;
                            if (!el) return '';
                            
                            const svg = el.querySelector('svg');
                            if (svg) {{
                                const title = svg.querySelector('title');
                                if (title) return title.textContent.trim();
                                const classes = svg.getAttribute('class') || '';
                                if (classes.includes('success') || classes.includes('check') || classes.includes('running'))
                                    return '✓ RUNNING';
                                else if (classes.includes('error') || classes.includes('failed') || classes.includes('stop'))
                                    return '✗ ERROR';
                            }}
                            
                            const img = el.querySelector('img');
                            if (img) return img.getAttribute('alt') || img.getAttribute('title') || '';
                            
                            return el.getAttribute('aria-label') || el.getAttribute('title') || '';
                        }})()
                    """)
                    if svg_data:
                        value = svg_data.strip()

                return ActionResult(
                    extracted_content={"index": index, "value": value},
                    include_in_memory=True,
                )
            except Exception as e:
                return ActionResult(error=str(e))

        @self.registry.action(
            "Validate retrieved value. MUST use plain text from retrieve_value_by_element. Supports operators: equals, contains, not_equals, running, stopped."
        )
        async def validate_value(
            actual: str,
            expected: str,
            operator: str = "equals",
        ):
            try:
                # Reject HTML markup
                if actual and ('<' in actual and '>' in actual):
                    return ActionResult(
                        error=f"❌ CRITICAL: validate_value received HTML markup: '{actual}'\n"
                               f"REQUIRED: Call retrieve_value_by_element FIRST, THEN validate_value with extracted text."
                    )
                
                # Normalize for comparison
                def normalize(text: str) -> str:
                    text = text.strip().lower()
                    if '✓' in text or 'check' in text or 'running' in text or 'active' in text or 'success' in text:
                        return '✓ running'
                    elif '✗' in text or 'error' in text or 'failed' in text or 'inactive' in text or 'stop' in text:
                        return '✗ error'
                    return text
                
                actual_norm = normalize(actual)
                expected_norm = normalize(expected)

                if operator == "equals":
                    passed = actual_norm == expected_norm
                elif operator == "contains":
                    passed = expected_norm in actual_norm
                elif operator == "not_equals":
                    passed = actual_norm != expected_norm
                elif operator == "running":
                    passed = '✓' in actual or 'running' in actual_norm or 'active' in actual_norm
                elif operator == "stopped":
                    passed = '✗' in actual or 'error' in actual_norm or 'inactive' in actual_norm or 'stop' in actual_norm
                else:
                    return ActionResult(error=f"Unsupported operator: {operator}")

                result = {
                    "actual": actual,
                    "expected": expected,
                    "operator": operator,
                    "passed": passed,
                }

                if not passed:
                    return ActionResult(
                        extracted_content=result,
                        error=f"Validation failed: {actual} {operator} {expected}",
                    )

                return ActionResult(extracted_content=result, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))

        @self.registry.action(
            "Ask human for help when blocked (missing credentials, CAPTCHA, subjective judgment needed)."
        )
        async def ask_for_assistant(query: str, browser: BrowserContext):
            if self.ask_assistant_callback:
                if inspect.iscoroutinefunction(self.ask_assistant_callback):
                    user_response = await self.ask_assistant_callback(query, browser)
                else:
                    user_response = self.ask_assistant_callback(query, browser)
                msg = f"AI ask: {query}. User response: {user_response['response']}"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            return ActionResult(extracted_content="Human cannot help. Try another way.", include_in_memory=True)

        @self.registry.action('Input text into form field. DO NOT USE FOR FILE UPLOADS - use upload_file instead.')
        async def input_text(index: int, text: str, browser: BrowserContext, **kwargs: bool):
            try:
                # FORCE SMART_LOGIN: Detect login form and redirect to smart_login
                is_credential = any(cred in text.upper() for cred in ['USERNAME', 'PASSWORD', 'USER', 'PASS', VAULT_CREDENTIAL_PREFIX, 'LOGIN', 'ADMIN'])
                
                if is_credential:
                    try:
                        page = await browser.get_current_page()
                        # Check if we're on a login page (password field visible)
                        password_visible = await page.locator('input[type="password"]').count() > 0
                        username_visible = await page.locator('input[type="text"], input[name*="user"], input[id*="user"]').count() > 0
                        
                        if password_visible and username_visible:
                            # LOGIN FORM DETECTED - Force redirect to smart_login
                            logger.info("🔐 LOGIN FORM DETECTED - Forcing smart_login instead of input_text")
                            
                            # Extract credentials from sensitive_data
                            username = None
                            password = None
                            if self.sensitive_data:
                                for key, val in self.sensitive_data.items():
                                    if 'USER' in key.upper():
                                        username = val
                                    elif 'PASS' in key.upper():
                                        password = val
                            
                            if username and password:
                                # Call smart_login directly
                                return await self._execute_smart_login(browser, username, password)
                            else:
                                return ActionResult(
                                    extracted_content=f"🔐 LOGIN FORM DETECTED but credentials not in vault. Use smart_login(username='{{{{{VAULT_CREDENTIAL_PREFIX}_USERNAME}}}}', password='{{{{{VAULT_CREDENTIAL_PREFIX}_PASSWORD}}}}') action instead.",
                                    include_in_memory=True
                                )
                        elif not password_visible:
                            return ActionResult(
                                extracted_content="✅ LOGIN ALREADY COMPLETE - No login form on this page. You are already authenticated. Call done() with success=true.",
                                include_in_memory=True
                            )
                    except Exception as detect_err:
                        logger.debug(f"Login detection failed: {detect_err}")
                
                dom_el = await browser.get_dom_element_by_index(index)
                if not dom_el:
                    return ActionResult(error=f'No element at index {index}')

                element = await browser.get_locate_element(dom_el)
                if not element:
                    return ActionResult(error=f'No element at index {index}')
                
                # Comprehensive element capture
                # Comprehensive element capture (Precision-mode enabled by default)
                if os.getenv("COMPREHENSIVE_CAPTURE", "false").lower() == "true":
                    try:
                        page = await browser.get_current_page()
                        element_data = await self.element_capturer.capture_element_data(
                            page=page,
                            element=element,
                            index=index,
                            action_context="input_text"
                        )
                        await self.element_capturer.save_element_data(element_data)
                    except Exception as e:
                        logger.warning(f"Feature capture failed: {e}")
                
                # CRITICAL: Block file input attempts
                element_type = await element.get_attribute('type')
                tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                
                if element_type == 'file' or tag_name == 'input[type="file"]':
                    return ActionResult(
                        error=f'🚫 CRITICAL ERROR: input_text() called on FILE INPUT element at index {index}.\n'
                              f'❌ input_text() CANNOT upload files.\n'
                              f'✅ REQUIRED ACTION: Use upload_file(index={index}, path="<file_path>") instead.\n'
                              f'Available files: {self.available_file_paths}'
                    )

                # Substitute parameters (e.g., {{PREFIX_USERNAME}}) from sensitive_data
                # Also strip any <secret> tags if the LLM hallucinated them, including typos
                text = text.replace("<secret>", "").replace("</secret>", "")
                text = text.replace("</secrett>", "")  # Common Qwen typo
                
                # Strip trailing JSON garbage from malformed LLM output
                # Common patterns: }}, {  or }},  or },  or trailing quotes
                import re
                text = re.sub(r'\}\s*,?\s*\{.*$', '', text)  # Remove }}, { and anything after
                text = re.sub(r'\}\s*,\s*$', '', text)  # Remove trailing }},
                text = re.sub(r'^\s*["\']+|["\']+\s*$', '', text)  # Strip surrounding quotes
                text = text.strip()
                
                if self.sensitive_data:
                    for key, val in self.sensitive_data.items():
                        # Try both common placeholder formats
                        placeholders = [f"{{{{{key}}}}}", f"{{{key}}}", key]
                        for placeholder in placeholders:
                            if placeholder in text:
                                text = text.replace(placeholder, val)

                # Mask sensitive data in logs using the same logic
                log_text = text
                if self.sensitive_data:
                    for secret_val in self.sensitive_data.values():
                        if secret_val and isinstance(secret_val, str) and len(secret_val) > 1:
                            log_text = log_text.replace(secret_val, "******")
                        elif secret_val and not isinstance(secret_val, str):
                            log_text = log_text.replace(str(secret_val), "******")
                
                # Double-check with global filter if available
                if hasattr(builtins, 'redacting_filter_values'):
                    for val in builtins.redacting_filter_values:
                        if val and str(val) in log_text:
                            log_text = log_text.replace(str(val), "******")
                
                await element.fill(text)
                msg = f'Input {log_text} into index {index}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f'Failed to input text: {str(e)}')

        @self.registry.action('Upload file to file input element. REQUIRED for <input type="file"> elements. This is the ONLY way to upload files - input_text will NOT work for file uploads.')
        async def upload_file(index: int, path: str, browser: BrowserContext):
            """
            Upload a file to a file input element.
            
            Args:
                index: The element index of the file input
                path: Full file path from available_file_paths
            """
            logger.info(f"🔍 upload_file called: index={index}, path={path}")
            logger.info(f"📂 Available file paths: {self.available_file_paths}")
            
            # Normalize path separators
            path_normalized = os.path.normpath(path)
            
            # Check if file is available
            file_available = False
            matched_path = None
            
            for available_path in self.available_file_paths:
                available_normalized = os.path.normpath(available_path)
                # Check exact match or if the filename matches
                if available_normalized == path_normalized or os.path.basename(available_normalized) == os.path.basename(path_normalized):
                    file_available = True
                    matched_path = available_normalized
                    break
            
            if not file_available:
                # RELAXATION: If running locally and file exists, allow it.
                if os.path.exists(path_normalized):
                     logger.warning(f"⚠️ Allowing local file upload not in context list: {path_normalized}")
                     matched_path = path_normalized
                else:
                     return ActionResult(
                        error=f'❌ File not available: {path}\n'
                              f'📂 Available files:\n' + '\n'.join([f'  - {p}' for p in self.available_file_paths]) +
                              f'\n(Or provide a valid absolute path to a file that exists on this machine)'
                    )
            
            # Use the matched path
            actual_path = matched_path
            logger.info(f"✅ Using file path: {actual_path}")
            
            if not os.path.exists(actual_path):
                return ActionResult(error=f'❌ File does not exist: {actual_path}')

            try:
                dom_el = await browser.get_dom_element_by_index(index)
                if not dom_el:
                    return ActionResult(error=f'❌ No element at index {index}')
                
                # Get file upload element
                # Attempt 1: Check if it's a file input or has one associated
                file_upload_dom_el = dom_el.get_file_upload_element()
                
                if file_upload_dom_el:
                    file_upload_el = await browser.get_locate_element(file_upload_dom_el)
                    if file_upload_el:
                        await file_upload_el.set_input_files(actual_path)
                        logger.info(f'✅ Successfully uploaded file (direct input): {os.path.basename(actual_path)} to index {index}')
                        return ActionResult(
                            extracted_content=f'✅ Uploaded file: {os.path.basename(actual_path)} to index {index}',
                            include_in_memory=True
                        )

                # Attempt 2: Fallback - Click and handle file chooser (common for "Upload" buttons)
                logger.info(f"ℹ️ Element {index} is not a direct file input. Attempting click + file chooser handling...")
                
                element = await browser.get_locate_element(dom_el)
                if not element:
                     return ActionResult(error=f'❌ Cannot locate element at index {index} for click interaction.')

                try:
                    page = await browser.get_current_page() # DEFINE PAGE HERE
                    async with page.expect_file_chooser(timeout=30000) as fc_info:
                        await element.click()
                    
                    file_chooser = await fc_info.value
                    await file_chooser.set_files(actual_path)
                    
                    logger.info(f'✅ Successfully uploaded file (via chooser): {os.path.basename(actual_path)}')
                    return ActionResult(
                        extracted_content=f'✅ Uploaded file via chooser: {os.path.basename(actual_path)} (clicked index {index})',
                        include_in_memory=True
                    )
                except asyncio.TimeoutError:
                     return ActionResult(error=f'❌ Failed: Clicked element {index}, but no file chooser dialog opened within 30s. Is this the right upload button?')

            except Exception as e:
                logger.error(f'❌ Upload failed: {str(e)}', exc_info=True)
                return ActionResult(error=f'❌ Upload failed: {str(e)}')

        @self.registry.action('Click element')
        async def click_element(index: int, browser: BrowserContext):
            try:
                dom_el = await browser.get_dom_element_by_index(index)
                if not dom_el:
                    return ActionResult(error=f'No element at index {index}')
                
                element = await browser.get_locate_element(dom_el)
                if not element:
                    return ActionResult(error=f'No element found at index {index}')

                # SMART LINK PREFERENCE: Check if this is a menu trigger instead of actual link
                page = await browser.get_current_page()
                try:
                    tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
                    role = await element.get_attribute("role") or ""
                    aria_haspopup = await element.get_attribute("aria-haspopup") or ""
                    title = await element.get_attribute("title") or ""
                    
                    # Detect menu triggers: role="menu", aria-haspopup, or "Function menu" in title
                    is_menu_trigger = (
                        role.lower() == "menu" or 
                        aria_haspopup.lower() == "true" or 
                        "function menu" in title.lower() or
                        "menu for" in title.lower()
                    )
                    
                    if is_menu_trigger and tag_name != "a":
                        # Extract text to find the actual link
                        element_text = await element.inner_text()
                        # Clean up: often format is "Function menu for X" -> extract "X"
                        search_text = title.replace("Function menu for", "").strip() if "Function menu for" in title else element_text.strip()
                        
                        if search_text:
                            logger.info(f"🔀 Menu trigger detected at index {index}. Searching for link '{search_text}'...")
                            # Try to find an <a> tag with this text
                            link_locator = page.locator(f"a:has-text('{search_text}')").first
                            try:
                                if await link_locator.is_visible(timeout=2000):
                                    await link_locator.click(timeout=3000)
                                    msg = f'🔀 Clicked navigation link "{search_text}" (redirected from menu trigger at index {index})'
                                    logger.info(msg)
                                    return ActionResult(extracted_content=msg, include_in_memory=True)
                            except:
                                logger.debug(f"Link search failed, falling back to original element")
                except Exception as redirect_err:
                    logger.debug(f"Smart redirect check skipped: {redirect_err}")

                # Comprehensive element capture (Precision-mode enabled by default)
                if os.getenv("COMPREHENSIVE_CAPTURE", "false").lower() == "true":
                    try:
                        element_data = await self.element_capturer.capture_element_data(
                            page=page,
                            element=element,
                            index=index,
                            action_context="click_element"
                        )
                        await self.element_capturer.save_element_data(element_data)
                    except Exception as e:
                        logger.warning(f"Feature capture failed: {e}")

                await element.click()
                msg = f'Clicked element at index {index}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f'Failed to click element: {str(e)}')

        @self.registry.action('Click element by visible text (when indices unreliable)')
        async def click_element_by_text(text: str, browser: BrowserContext):
            try:
                page = await browser.get_current_page()
                await page.wait_for_load_state('networkidle')
                await asyncio.sleep(0.5)

                locator = page.get_by_text(text, exact=False)
                await locator.click()
                return ActionResult(extracted_content=f'Clicked element: {text}', include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f'Failed to click "{text}": {str(e)}')

        @self.registry.action('Select dropdown option by text')
        async def select_dropdown_option(index: int, text: str, browser: BrowserContext):
            try:
                dom_el = await browser.get_dom_element_by_index(index)
                dropdown_el = await browser.get_locate_element(dom_el)
                await dropdown_el.click()
                await asyncio.sleep(0.5)

                page = await browser.get_current_page()
                option_locator = page.get_by_text(text, exact=True)
                await option_locator.click()
                return ActionResult(extracted_content=f'Selected "{text}" from dropdown {index}', include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f'Failed to select "{text}": {str(e)}')

        @self.registry.action('Zoom page to percentage (e.g., 80 for 80%)')
        async def zoom_page(percentage: int, browser: BrowserContext):
            try:
                page = await browser.get_current_page()
                await page.evaluate(f"document.body.style.zoom = '{percentage}%'")
                return ActionResult(extracted_content=f'Zoomed to {percentage}%', include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f'Zoom failed: {str(e)}')

        @self.registry.action('Scroll page. Direction: up/down/left/right. Amount in pixels.')
        async def scroll_page(direction: str = "down", amount: int = 500, browser: BrowserContext = None):
            try:
                page = await browser.get_current_page()
                scroll_mapping = {
                    "down": f"window.scrollBy(0, {amount})",
                    "up": f"window.scrollBy(0, -{amount})",
                    "right": f"window.scrollBy({amount}, 0)",
                    "left": f"window.scrollBy(-{amount}, 0)"
                }
                await page.evaluate(scroll_mapping.get(direction, scroll_mapping["down"]))
                return ActionResult(extracted_content=f'Scrolled {direction} by {amount}px', include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f'Scroll failed: {str(e)}')

        @self.registry.action('Go back to the previous page in history')
        async def go_back(browser: BrowserContext):
            try:
                page = await browser.get_current_page()
                await page.go_back()
                return ActionResult(extracted_content='Navigated back', include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f'Failed to go back: {str(e)}')

        @self.registry.action('Go forward to the next page in history')
        async def go_forward(browser: BrowserContext):
            try:
                page = await browser.get_current_page()
                await page.go_forward()
                return ActionResult(extracted_content='Navigated forward', include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f'Failed to go forward: {str(e)}')

        @self.registry.action('Reload the current page')
        async def refresh_page(browser: BrowserContext):
            try:
                page = await browser.get_current_page()
                await page.reload()
                return ActionResult(extracted_content='Page refreshed', include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f'Failed to refresh page: {str(e)}')

        @self.registry.action('Hover over element to reveal tooltips or menus')
        async def hover_element(index: int, browser: BrowserContext):
            try:
                dom_el = await browser.get_dom_element_by_index(index)
                if not dom_el:
                    return ActionResult(error=f'No element at index {index}')
                
                element = await browser.get_locate_element(dom_el)
                if not element:
                    return ActionResult(error=f'Unable to locate element at index {index}')
                
                await element.hover()
                await asyncio.sleep(0.3)  # Wait for hover effects
                return ActionResult(extracted_content=f'Hovered over element {index}', include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f'Hover failed: {str(e)}')

        @self.registry.action('Hover and capture DOM/state for status icons and tooltips')
        async def hover_capture(index: int, browser: BrowserContext):
            """Hover an element, then capture DOM/attr/style snapshot for icon/status reads."""
            try:
                dom_el = await browser.get_dom_element_by_index(index)
                if not dom_el:
                    return ActionResult(error=f'No element at index {index}')

                element = await browser.get_locate_element(dom_el)
                if not element:
                    return ActionResult(error=f'Unable to locate element at index {index}')

                page = await browser.get_current_page()

                await element.hover()
                await asyncio.sleep(2.0)  # Allow slower tooltips/state to render

                hover_data = await page.evaluate(
                    """
                    (idx) => {
                        const el = document.evaluate(
                            "//*[@data-browser-use-index='" + idx + "']",
                            document, null,
                            XPathResult.FIRST_ORDERED_NODE_TYPE, null
                        ).singleNodeValue;
                        if (!el) return { error: 'element not found' };

                        const rect = el.getBoundingClientRect();
                        const style = window.getComputedStyle(el);

                        // Try to find a nearby tooltip if present
                        const tooltip = document.querySelector('[role="tooltip"], .tooltip, .ant-tooltip-inner, .mat-tooltip');

                        return {
                            outerHTML: el.outerHTML,
                            innerText: el.innerText || el.textContent || '',
                            title: el.getAttribute('title'),
                            alt: el.getAttribute('alt'),
                            ariaLabel: el.getAttribute('aria-label'),
                            role: el.getAttribute('role'),
                            src: el.getAttribute('src'),
                            dataStatus: el.getAttribute('data-status'),
                            className: el.className,
                            computedColor: style.color,
                            computedBackground: style.backgroundColor,
                            boundingBox: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                            tooltipText: tooltip ? (tooltip.innerText || tooltip.textContent || '').trim() : null
                        };
                    }
                    """,
                    index,
                )

                try:
                    await capture_dom_snapshot(page, reason=f"post_action:hover_capture", action_params={"index": index})
                except Exception:
                    # Snapshot failures should not block main action
                    pass

                return ActionResult(
                    extracted_content={
                        "index": index,
                        "hover_capture": hover_data,
                    },
                    include_in_memory=True,
                )
            except Exception as e:
                return ActionResult(error=f'Hover capture failed: {str(e)}')

        @self.registry.action('Hover relative to indexed element in any direction (left/right/up/down). Captures tooltip and state. Useful for non-indexed icons, badges, status indicators.')
        async def hover_with_offset(index: int, browser: BrowserContext, direction: str = "left", distance: int = 30, capture_state: bool = True):
            """
            Hover an element positioned in any direction from an indexed reference element.
            Perfect for status icons, badges, indicators that aren't independently indexed.
            
            Args:
                index: Index of the reference element
                direction: Direction to move from element ("left", "right", "up", "down")
                distance: Distance in pixels to move (default 30)
                capture_state: Whether to capture tooltip and DOM state (default True)
            
            Example:
                {"hover_with_offset": {"index": 18, "direction": "left", "distance": 30}}
                → Hovers 30px to the left of element 18, captures tooltip
            """
            try:
                dom_el = await browser.get_dom_element_by_index(index)
                if not dom_el:
                    return ActionResult(error=f'No element at index {index}')

                element = await browser.get_locate_element(dom_el)
                if not element:
                    return ActionResult(error=f'Unable to locate element at index {index}')

                page = await browser.get_current_page()

                # Get the element's bounding box
                box = await element.bounding_box()
                if not box:
                    return ActionResult(error=f'Cannot get bounding box for element {index}')

                # Calculate hover position based on direction
                direction_lower = direction.lower().strip()
                
                if direction_lower == "left":
                    hover_x = box['x'] - distance
                    hover_y = box['y'] + box['height'] / 2
                elif direction_lower == "right":
                    hover_x = box['x'] + box['width'] + distance
                    hover_y = box['y'] + box['height'] / 2
                elif direction_lower == "up":
                    hover_x = box['x'] + box['width'] / 2
                    hover_y = box['y'] - distance
                elif direction_lower == "down":
                    hover_x = box['x'] + box['width'] / 2
                    hover_y = box['y'] + box['height'] + distance
                else:
                    return ActionResult(error=f'Invalid direction "{direction}". Use: left, right, up, or down')

                # Move mouse to calculated position
                await page.mouse.move(hover_x, hover_y)
                await asyncio.sleep(0.5)  # Let hover effects render

                tooltip_text = None
                attributes = {}
                
                if capture_state:
                    # Try to capture any tooltip that appears
                    tooltip_text = await page.evaluate(
                        """
                        () => {
                            const tooltip = document.querySelector('[role="tooltip"], .tooltip, .ant-tooltip-inner, .mat-tooltip');
                            if (tooltip) {
                                return (tooltip.innerText || tooltip.textContent || '').trim();
                            }
                            return null;
                        }
                        """
                    )
                    
                    # Try to get element at hover position for attributes
                    attributes = await page.evaluate(
                        """
                        (x, y) => {
                            const el = document.elementFromPoint(x, y);
                            if (!el) return {};
                            
                            return {
                                tagName: el.tagName,
                                className: el.className,
                                title: el.getAttribute('title'),
                                ariaLabel: el.getAttribute('aria-label'),
                                role: el.getAttribute('role'),
                                dataTestId: el.getAttribute('data-testid'),
                            };
                        }
                        """,
                        hover_x,
                        hover_y,
                    )

                try:
                    await capture_dom_snapshot(
                        page, 
                        reason=f"post_action:hover_with_offset",
                        action_params={"index": index, "direction": direction, "distance": distance}
                    )
                except Exception:
                    pass

                # Build result message
                result_msg = f"Hovered {distance}px {direction} of element {index}"
                if tooltip_text:
                    result_msg += f". Tooltip: {tooltip_text}"

                return ActionResult(
                    extracted_content={
                        "index": index,
                        "direction": direction,
                        "distance": distance,
                        "position": {"x": hover_x, "y": hover_y},
                        "tooltip": tooltip_text,
                        "element_attributes": attributes,
                        "message": result_msg
                    },
                    include_in_memory=True,
                )
            except Exception as e:
                return ActionResult(error=f'Hover with offset failed: {str(e)}')


        @self.registry.action('Audit current site security connection and SSL/TLS certificates.')
        async def get_site_security_info(browser: BrowserContext):
            """
            Extract security details of the current page's connection, 
            including SSL/TLS certificate information.
            """
            try:
                page = await browser.get_current_page()
                url = page.url
                is_https = url.startswith('https://')
                
                # Basic security metadata from the DOM/Window context
                security_meta = await page.evaluate('''() => {
                    return {
                        protocol: window.location.protocol,
                        origin: window.location.origin,
                        isSecureContext: window.isSecureContext
                    }
                }''')
                
                msg = f"🛡️ Site Security Audit for: {url}\n"
                msg += f"--------------------------------------------------\n"
                msg += f"Connection: {'✅ SECURE (HTTPS)' if is_https else '❌ INSECURE (HTTP)'}\n"
                msg += f"Context: {'✅ Secure Context' if security_meta.get('isSecureContext') else '❌ Non-Secure Context'}\n"
                
                # Fetch detailed certificate info via page.request (OOB check to same URL)
                # This is more reliable than trying to intercept the main frame's response 
                # after the fact if browser-use doesn't provide it directly.
                try:
                    # Using the page's context to maintain cookies/sessions
                    request_context = page.context.request
                    response = await request_context.get(url)
                    details = response.security_details()
                    
                    if details:
                        msg += f"\nCertificate Details:\n"
                        msg += f" • Issuer: {details.get('issuer', 'N/A')}\n"
                        msg += f" • Subject: {details.get('subject', 'N/A')}\n"
                        msg += f" • Protocol: {details.get('protocol', 'N/A')}\n"
                        msg += f" • Valid From: {details.get('validFrom', 'N/A')}\n"
                        msg += f" • Valid To: {details.get('validTo', 'N/A')}\n"
                        
                        # Add a sanity check for expiration
                        from datetime import datetime
                        try:
                            # validTo is usually a timestamp or ISO string depending on version
                            # but Playwright typically returns a timestamp (seconds/ms) 
                            # or we can parse it if it's a string.
                            expired = False
                            valid_to = details.get('validTo')
                            if isinstance(valid_to, (int, float)):
                                # Playwright usually provides decimal seconds
                                expiry_dt = datetime.fromtimestamp(valid_to)
                                if expiry_dt < datetime.now():
                                    expired = True
                                msg += f" • Status: {'❌ EXPIRED' if expired else '✅ VALID'}\n"
                        except:
                            pass
                    else:
                        msg += "\n⚠️ No certificate details found. This usually happens for local/broken connections.\n"
                except Exception as cert_err:
                    msg += f"\n⚠️ Deep certificate scan failed: {str(cert_err)}\n"

                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f'Security audit failed: {str(e)}')


        @self.registry.action('Check if checkbox/radio is checked; click only if unchecked. Works for checkboxes, radio buttons, and toggle inputs.')
        async def safe_check_and_click(index: int, browser: BrowserContext):
            """
            Check the state of a checkbox, radio button, or toggle input.
            Only click if currently unchecked. Prevents unnecessary repeated clicks.
            """
            try:
                dom_el = await browser.get_dom_element_by_index(index)
                if not dom_el:
                    return ActionResult(error=f'No element at index {index}')
                
                element = await browser.get_locate_element(dom_el)
                if not element:
                    return ActionResult(error=f'Unable to locate element at index {index}')
                
                # Get element type for logging
                element_type = await element.get_attribute('type')
                element_tag = await element.evaluate('el => el.tagName.toLowerCase()')
                element_role = await element.get_attribute('role')
                
                # Try to check current checked state - handle both input elements and other controls
                try:
                    is_checked = await element.is_checked()
                except Exception:
                    # Fallback: check for 'checked' attribute or aria-checked if is_checked() fails
                    is_checked = await element.evaluate('''
                        el => {
                            if (el.checked !== undefined) return el.checked;
                            if (el.getAttribute('aria-checked') === 'true') return true;
                            return false;
                        }
                    ''')
                
                logger.info(f"✓ safe_check_and_click at index {index}: type={element_type}, tag={element_tag}, checked={is_checked}")
                
                if is_checked:
                    # Already checked, skip click
                    result_msg = f"✓ Radio/checkbox at index {index} already checked (skipped click)"
                    logger.info(result_msg)
                    return ActionResult(
                        extracted_content=result_msg,
                        include_in_memory=True
                    )
                
                # Not checked, click once
                await element.click()
                await asyncio.sleep(0.3)
                
                # Verify final state after click
                try:
                    final_checked = await element.is_checked()
                except Exception:
                    final_checked = await element.evaluate('''
                        el => {
                            if (el.checked !== undefined) return el.checked;
                            if (el.getAttribute('aria-checked') === 'true') return true;
                            return false;
                        }
                    ''')
                
                result_msg = f"✓ Radio/checkbox at index {index} clicked. Final state: {'checked' if final_checked else 'unchecked'}"
                logger.info(result_msg)
                
                return ActionResult(
                    extracted_content=result_msg,
                    include_in_memory=True
                )
            except Exception as e:
                logger.error(f"✗ safe_check_and_click failed at index {index}: {str(e)}", exc_info=True)
                return ActionResult(error=f'Safe check and click failed: {str(e)}')



        @self.registry.action('Navigate directly to URL')
        async def go_to_url(url: str, browser: BrowserContext):
            try:
                page = await browser.get_current_page()
                await page.goto(url)
                await capture_dom_snapshot(page, reason="go_to_url")
                return ActionResult(extracted_content=f"Navigated to {url}", include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))

        @self.registry.action(
            '✅ PRIMARY LOGIN METHOD: Atomically enters username & password and clicks Sign In. '
            'SAVES TOKENS and reliability. ALWAYS use this instead of individual inputs for login. '
            f'Pass {{{{{VAULT_CREDENTIAL_PREFIX}_USERNAME}}}} and {{{{{VAULT_CREDENTIAL_PREFIX}_PASSWORD}}}}.'
        )
        async def smart_login(
            username: str, 
            password: str, 
            browser: BrowserContext,
            button_text: Optional[str] = "Sign in"
        ):
            """
            Atomic login action that handles UN/PW/Click in one turn.
            """
            try:
                page = await browser.get_current_page()
                
                # 1. Identify Username field
                un_selectors = [
                    'input[id*="username" i]', 'input[id*="user" i]', 'input[id*="login" i]', 'input[name*="user" i]', 
                    'input[placeholder*="sername" i]', 'input[aria-label*="sername" i]'
                ]
                username_el = None
                for selector in un_selectors:
                    el = page.locator(selector).first
                    if await el.count() > 0 and await el.is_visible():
                        username_el = el
                        break
                
                # 2. Identify Password field
                pw_selectors = [
                    'input[type="password"]', 'input[id*="password" i]', 'input[id*="pass" i]', 'input[name*="pass" i]'
                ]
                password_el = None
                for selector in pw_selectors:
                    el = page.locator(selector).first
                    if await el.count() > 0 and await el.is_visible():
                        password_el = el
                        break
                
                # 3. Identify Submit button
                btn_selectors = [
                    f'button:has-text("{button_text}")', f'input[type="submit"]',
                    f'input[value*="{button_text}" i]', f'button[id*="login" i]'
                ]
                submit_btn = None
                for selector in btn_selectors:
                    el = page.locator(selector).first
                    if await el.is_visible():
                        submit_btn = el
                        break

                if not username_el or not password_el:
                    return ActionResult(error="Could not find username or password field automatically. Use manual indexing.")

                # Handle sensitive data substitution
                un_val = username
                pw_val = password
                if self.sensitive_data:
                    for k, v in self.sensitive_data.items():
                        un_val = un_val.replace(f"{{{{{k}}}}}", v).replace(f"{{{k}}}", v)
                        pw_val = pw_val.replace(f"{{{{{k}}}}}", v).replace(f"{{{k}}}", v)

                # Execute interactions
                await username_el.fill(un_val)
                await password_el.fill(pw_val)
                
                msg = f"✓ Atomic login initiated (Username and Password fields found and filled)"
                
                if submit_btn:
                    await submit_btn.click()
                    msg += f" and '{button_text}' button clicked."
                else:
                    await page.keyboard.press("Enter")
                    msg += " and Enter pressed (button not found)."

                # CRITICAL: Wait for page to transition after login
                # This prevents the agent from getting confused on the post-login page
                try:
                    # Wait for navigation or network idle
                    await page.wait_for_load_state("networkidle", timeout=10000)
                    
                    # Verify login succeeded: password field should no longer be visible
                    await asyncio.sleep(1)  # Small buffer for SPA transitions
                    password_visible = False
                    for selector in pw_selectors:
                        try:
                            el = page.locator(selector).first
                            if await el.count() > 0 and await el.is_visible(timeout=1000):
                                password_visible = True
                                break
                        except:
                            pass
                    
                    if not password_visible:
                        msg += " ✓ Login successful - redirected to authenticated page."
                    else:
                        msg += " ⚠ Login form still visible - credentials may be incorrect."
                        
                except Exception as wait_err:
                    msg += f" (Navigation check: {wait_err})"

                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f"Smart login failed: {str(e)}")

        @self.registry.action('Extract page content. Best for general text extraction or unstructured data.')
        async def extract_content(goal: str, browser: BrowserContext, should_strip_link_urls: bool = True):
            """
            Extract specific information from the current page.
            """
            try:
                page = await browser.get_current_page()
                
                # Performance: detect if goal is about a table, redirect if so
                table_keywords = ['table', 'columns', 'rows', 'data', 'list', 'status', 'id']
                if any(kw in goal.lower() for kw in table_keywords):
                    # Try to find a table and extract it
                    tables = await page.query_selector_all('table')
                    if tables:
                        msg = "Goal mentions tabular data. Suggest using extract_table for better results if this general extraction is insufficient."
                        logger.info(msg)
                
                # Standard extraction logic (generic)
                content = await page.evaluate("""() => {
                    // Generic extraction: get all visible text but prioritize clean structure
                    return document.body.innerText;
                }""")
                
                return ActionResult(
                    extracted_content=f"Page content for goal '{goal}':\n\n{content[:5000]}",
                    include_in_memory=True
                )
            except Exception as e:
                return ActionResult(error=str(e))

        @self.registry.action('Extract structured table data. REQUIRED for reports containing lists, IDs, or status grids.')
        async def extract_table(
            goal: str, 
            browser: BrowserContext, 
            header_keywords: Optional[list[str]] = None,
            table_index: int = 0
        ):
            """
            Extract structured data from tables. 
            Provide header_keywords to filter columns.
            """
            try:
                page = await browser.get_current_page()
                
                # Robust extraction logic aligned with script_helpers.py
                # 1. Find the table
                tables = await page.query_selector_all('table')
                if not tables:
                    return ActionResult(error="No tables found on the current page.")
                
                if table_index >= len(tables):
                    table_index = 0 # Fallback to first
                
                table = tables[table_index]
                
                # 2. Extract rows
                rows = await table.query_selector_all('tr')
                if not rows:
                    return ActionResult(error="Table found but contains no rows (tr elements).")
                
                # 3. Resolve headers
                col_map = {}
                header_row_idx = -1
                
                # Scan first 10 rows for headers if not found in first
                num_to_scan = min(10, len(rows))
                
                headers_found = []
                for i in range(num_to_scan):
                    cells = await rows[i].query_selector_all('th, td')
                    texts = [(await c.inner_text()).strip() for c in cells]
                    
                    if header_keywords:
                        # Find indices for requested keywords
                        current_map = {}
                        for kw in header_keywords:
                            idx = next((j for j, t in enumerate(texts) if kw.lower() in t.lower()), -1)
                            if idx != -1:
                                current_map[kw] = idx
                        
                        if len(current_map) >= len(header_keywords) * 0.5: # 50% match
                            col_map = current_map
                            header_row_idx = i
                            headers_found = texts
                            break
                    else:
                        # Use first row with > 1 columns as header
                        if len(texts) > 1:
                            headers_found = texts
                            header_row_idx = i
                            col_map = {h: idx for idx, h in enumerate(texts)}
                            break
                
                if not headers_found:
                    # Fallback: invent headers Col 1, Col 2...
                    header_row_idx = -1
                    cells = await rows[0].query_selector_all('td, th')
                    headers_found = [f"Col {idx+1}" for idx in range(len(cells))]
                    col_map = {h: idx for idx, h in enumerate(headers_found)}

                # 4. Extract data
                data_results = []
                for i in range(header_row_idx + 1, len(rows)):
                    cells = await rows[i].query_selector_all('td')
                    if len(cells) < len(headers_found):
                        continue
                        
                    row_data = {}
                    for h, idx in col_map.items():
                        if idx < len(cells):
                            row_data[h] = (await cells[idx].inner_text()).strip()
                    
                    if row_data:
                        data_results.append(row_data)
                
                if not data_results:
                    return ActionResult(error=f"Table found with headers {headers_found}, but no data rows followed.")

                # Format as JSON string
                import json
                json_data = json.dumps(data_results, indent=2)
                
                # Keep it under token limit
                if len(json_data) > 3000:
                    json_data = json_data[:3000] + "... [TRUNCATED]"

                return ActionResult(
                    extracted_content=f"Extracted {len(data_results)} rows from table (Goal: {goal}):\n\n```json\n{json_data}\n```",
                    include_in_memory=True
                )
            except Exception as e:
                return ActionResult(error=f"Table extraction failed: {str(e)}")

    @time_execution_sync('--act')
    async def act(
        self,
        action: ActionModel,
        browser_context: Optional[BrowserContext] = None,
        page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        available_file_paths: Optional[list[str]] = None,
        context: Context | None = None,
    ) -> ActionResult:
        """Execute action with security and loop detection"""
        try:
            # Update available file paths if provided
            if available_file_paths is not None:
                self.available_file_paths = available_file_paths
                logger.info(f"📂 Updated available file paths: {self.available_file_paths}")
            
            # Page stability check (Optimized for speed)
            if browser_context:
                try:
                    page = await browser_context.get_current_page()
                    # CRITICAL SPEED OPTIMIZATION: Reduce timeout for load state checks
                    # Performance: Use 'domcontentloaded' instead of 'networkidle' for faster turns where safe
                    await asyncio.wait_for(page.wait_for_load_state('domcontentloaded', timeout=500), timeout=0.6)
                except (asyncio.TimeoutError, Exception):
                    pass

            # Loop detection
            if browser_context:
                try:
                    page = await browser_context.get_current_page()
                    url = page.url
                    title = await page.title()
                    content_len = len(await page.content())
                    state_hash = f"{url}:{title}:{content_len}"
                    self.state_history.append(state_hash)
                    if len(self.state_history) > 20:
                        self.state_history.pop(0)
                        
                    # Looser loop detection for non-mutating actions (threshold 5)
                    action_data = action.model_dump(exclude_unset=True)
                    # Add 'wait' and 'extract_table' to whitelist to prevent premature loop detection
                    non_mutating_actions = ['extract_content', 'extract_table', 'scroll_page', 'scroll_to_text', 'wait_for_element', 'done', 'retrieve_value_by_element', 'wait']
                    is_non_mutating = any(name in action_data for name in non_mutating_actions)
                    
                    max_repeats = 5 if is_non_mutating else 3
                    
                    if self.state_history.count(state_hash) > max_repeats:
                        if 'done' in action_data:
                            logger.info(f"Bypassing loop detection for 'done' action at {state_hash}")
                        else:
                            logger.warning(f"Loop detected: {state_hash} (Repeats: {self.state_history.count(state_hash)})")
                            return ActionResult(error=f"Loop detected: Same page state repeated {max_repeats}+ times. Try moving the page (scroll, click) or use a different approach.")
                except Exception:
                    pass

            # Credential substitution
            def substitute_sensitive_data(params: dict, sensitive_data: Optional[Dict[str, str]]) -> dict:
                if not sensitive_data:
                    return params
                
                def substitute_in_value(value):
                    """Recursively substitute sensitive data in nested structures"""
                    if isinstance(value, str):
                        # Replace all placeholders in string values
                        for s_key, s_val in sensitive_data.items():
                            if s_key in value:
                                value = value.replace(s_key, s_val)
                        return value
                    elif isinstance(value, dict):
                        # Recursively process nested dictionaries
                        return {k: substitute_in_value(v) for k, v in value.items()}
                    elif isinstance(value, list):
                        # Recursively process lists
                        return [substitute_in_value(v) for v in value]
                    else:
                        # Return non-string, non-container values as-is
                        return value
                
                new_params = {}
                for key, value in params.items():
                    new_params[key] = substitute_in_value(value)
                return new_params

            result_obj: ActionResult | None = None
            executed_action_name: str | None = None

            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is None:
                    continue

                executed_action_name = action_name

                # Capture DOM BEFORE performing any action
                if browser_context:
                    try:
                        page = await browser_context.get_current_page()
                        # Include action_name in params for Ollama verification
                        capture_params = {**params, "action_name": action_name}
                        await capture_dom_snapshot(page, reason=f"pre_action:{action_name}", action_params=capture_params)
                        logger.debug(f"✓ Captured DOM before {action_name}")
                    except Exception as snap_exc:  # pragma: no cover
                        logger.debug(f"DOM snapshot before action skipped: {snap_exc}")

                data_to_use = sensitive_data or self.sensitive_data
                if data_to_use:
                    params = substitute_sensitive_data(params, data_to_use)

                # Auto-convert click_element_by_index to safe_check_and_click for checkbox/radio
                # Conversion triggers if EITHER (a) goal mentions checking status of radio/checkbox/toggle AND element is radio/checkbox
                # OR (b) element is radio/checkbox regardless of goal text (fallback safety)
                if action_name == "click_element_by_index":
                    context_text = ""
                    if context is not None and hasattr(context, 'current_state') and hasattr(context.current_state, 'next_goal'):
                        context_text = str(context.current_state.next_goal).lower()

                    action_keywords = ['check', 'verify', 'retrieve', 'read', 'status', 'state', 'selected']
                    element_keywords = ['radio', 'checkbox', 'toggle']

                    has_action = any(word in context_text for word in action_keywords)
                    has_element = any(word in context_text for word in element_keywords)
                    has_check_intent = has_action and has_element

                    if 'index' in params and browser_context:
                        try:
                            idx = params.get('index')
                            dom_el = await browser_context.get_dom_element_by_index(idx)
                            if dom_el:
                                element = await browser_context.get_locate_element(dom_el)
                                if element:
                                    elem_type = await element.get_attribute('type')
                                    role = await element.get_attribute('role')
                                    is_radio_checkbox = elem_type in ['checkbox', 'radio'] or role in ['checkbox', 'radio']

                                    if is_radio_checkbox and (has_check_intent or True):
                                        logger.info(f"🔄 Auto-converting click_element_by_index to safe_check_and_click for {elem_type or role} at index {idx}")
                                        action_name = "safe_check_and_click"
                                        executed_action_name = action_name
                        except Exception as e:
                            logger.debug(f"Element inspection for auto-conversion failed: {e}")

                # ===== COMPREHENSIVE ELEMENT CAPTURE =====
                # Capture detailed element data BEFORE action
                if browser_context and 'index' in params:
                    try:
                        page = await browser_context.get_current_page()
                        idx = params.get('index')
                        dom_el = await browser_context.get_dom_element_by_index(idx)
                        
                        if dom_el:
                            element = await browser_context.get_locate_element(dom_el)
                            if element:
                                # Capture comprehensive element data
                                element_data = await self.element_capturer.capture_element_data(
                                    page,
                                    element,
                                    index=idx,
                                    action_context=action_name
                                )
                                
                                # Save to JSON
                                filename = f"element_{action_name}_idx{idx}_{element_data['elementHash']}.json"
                                await self.element_capturer.save_element_data(element_data, filename)
                                
                                logger.info(f"✓ Captured comprehensive element data: {element_data['recommendedSelector']['selector'][:60]}")
                                
                                # Store in result for later use
                                params['_element_data'] = element_data
                    except Exception as e:
                        logger.debug(f"Comprehensive element capture failed (non-blocking): {e}")
                # ===== END COMPREHENSIVE CAPTURE =====

                if action_name.startswith("mcp"):
                    logger.debug(f"Invoke MCP tool: {action_name}")
                    mcp_tool = self.registry.registry.actions.get(action_name).function
                    result = await mcp_tool.ainvoke(params)
                else:
                    try:
                        result = await self.registry.execute_action(
                            action_name,
                            params,
                            browser=browser_context,
                            page_extraction_llm=page_extraction_llm,
                            sensitive_data=sensitive_data,
                            available_file_paths=self.available_file_paths,
                            context=context,
                        )
                    except IndexError as ie:
                        logger.warning(f"Index error: {ie}")
                        result = ActionResult(error="Element index out of range. Try text-based selection.")

                # ===== ENHANCED ERROR MESSAGES FOR BETTER LLM GUIDANCE =====
                # Add context-aware suggestions when actions fail to help LLM choose better alternatives
                # Only suggest alternatives after MULTIPLE failures to allow normal scrolling/retries
                if isinstance(result, ActionResult) and result.error:
                    original_error = result.error
                    
                    # Create tracking key for this specific action+parameter combo
                    tracking_key = None
                    if action_name == "scroll_to_text" and "text" in params:
                        tracking_key = ("scroll_to_text", params["text"])
                    elif action_name == "click_element_by_index" and "index" in params:
                        tracking_key = ("click_element_by_index", params["index"])
                    elif action_name == "input_text" and "index" in params:
                        tracking_key = ("input_text", params["index"])
                    
                    # Track consecutive failures
                    if tracking_key:
                        self.action_failure_tracker[tracking_key] = self.action_failure_tracker.get(tracking_key, 0) + 1
                        failure_count = self.action_failure_tracker[tracking_key]
                    else:
                        failure_count = 1
                    
                    # Only add hints after MULTIPLE failures (2+ attempts)
                    # This allows normal scrolling/retries before suggesting alternatives
                    
                    # Detect scroll_to_text failures (after 2 failures)
                    if action_name == "scroll_to_text" and ("not found" in original_error.lower() or "not visible" in original_error.lower()) and failure_count >= 2:
                        try:
                            # Check if page has table elements
                            page = await browser_context.get_current_page()
                            has_tables = await page.evaluate("() => document.querySelectorAll('table').length > 0")
                            
                            hint = f"{original_error}\n💡 HINT: Failed {failure_count} times."
                            
                            if has_tables:
                                hint += " This page contains tables. Text within table cells may not be searchable with scroll_to_text. "
                                hint += "Consider using 'extract_content' to get the full table data."
                            else:
                                hint += " If the text is hard to find or the page layout is complex, try using 'scroll_page' with direction='down' and a large amount (e.g. 800) to move through the page manually."
                                
                            result.error = hint
                            logger.info(f"Enhanced scroll_to_text error with fallback hint (after {failure_count} failures)")
                        except Exception as e:
                            logger.debug(f"Could not check for tables: {e}")
                    
                    # Detect repeated click failures - suggest text-based or extraction approaches (after 2 failures)
                    elif action_name == "click_element_by_index" and ("out of range" in original_error.lower() or "unable to locate" in original_error.lower()) and failure_count >= 2:
                        result.error = (
                            f"{original_error}\n"
                            f"💡 HINT: Failed {failure_count} times. Element index may be stale or incorrect. Try 'click_element_by_text' if you know the button/link text, "
                            f"or use 'extract_content' to get current page structure and find the correct element."
                        )
                    
                    # Detect input_text failures - suggest wait or extraction (after 2 failures)
                    elif action_name == "input_text" and ("not found" in original_error.lower() or "timeout" in original_error.lower()) and failure_count >= 2:
                        result.error = (
                            f"{original_error}\n"
                            f"💡 HINT: Failed {failure_count} times. Input field may not be loaded yet. Try 'wait_for_element' with appropriate timeout, "
                            f"or use 'extract_content' to verify the form structure is present before attempting input."
                        )
                else:
                    # Action succeeded - reset failure tracker for this action
                    if action_name == "scroll_to_text" and "text" in params:
                        tracking_key = ("scroll_to_text", params["text"])
                        if tracking_key in self.action_failure_tracker:
                            del self.action_failure_tracker[tracking_key]
                    elif action_name == "click_element_by_index" and "index" in params:
                        tracking_key = ("click_element_by_index", params["index"])
                        if tracking_key in self.action_failure_tracker:
                            del self.action_failure_tracker[tracking_key]
                    elif action_name == "input_text" and "index" in params:
                        tracking_key = ("input_text", params["index"])
                        if tracking_key in self.action_failure_tracker:
                            del self.action_failure_tracker[tracking_key]
                # ===== END ENHANCED ERROR MESSAGES =====

                if isinstance(result, str):
                    result_obj = ActionResult(extracted_content=result)
                elif isinstance(result, ActionResult):
                    result_obj = result
                elif result is None:
                    result_obj = ActionResult()
                else:
                    raise ValueError(f'Invalid result type: {type(result)}')

                break  # Only one action expected per call

            if result_obj is None:
                result_obj = ActionResult()

            # PERF: Post-action snapshots removed. Pre-action snapshot of the NEXT step is sufficient.
            # This saves significant time (1-2s) per action in sequential turns.
            
            return result_obj
        except Exception as e:
            raise e

    async def setup_mcp_client(self, mcp_server_config: Optional[Dict[str, Any]] = None):
        self.mcp_server_config = mcp_server_config
        if self.mcp_server_config:
            self.mcp_client = await setup_mcp_client_and_tools(self.mcp_server_config)
            self.register_mcp_tools()

    def register_mcp_tools(self):
        if self.mcp_client:
            for server_name in self.mcp_client.server_name_to_tools:
                for tool in self.mcp_client.server_name_to_tools[server_name]:
                    tool_name = f"mcp.{server_name}.{tool.name}"
                    self.registry.registry.actions[tool_name] = RegisteredAction(
                        name=tool_name,
                        description=tool.description,
                        function=tool,
                        param_model=create_tool_param_model(tool),
                    )
                    logger.info(f"Registered MCP tool: {tool_name}")

    async def close_mcp_client(self):
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)