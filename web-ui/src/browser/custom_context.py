import asyncio
import json
import logging
import os

from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import BrowserContext as PlaywrightBrowserContext
from typing import Optional
from browser_use.browser.context import BrowserContextState
from src.utils.dom_snapshot import capture_dom_snapshot

# Define IN_DOCKER if not available (fallback)
try:
    from browser_use.browser.browser import IN_DOCKER
except ImportError:
    IN_DOCKER = False

logger = logging.getLogger(__name__)


class CustomBrowserContext(BrowserContext):
    def __init__(
            self,
            browser: 'Browser',
            config: BrowserContextConfig | None = None,
            state: Optional[BrowserContextState] = None,
    ):
        super(CustomBrowserContext, self).__init__(browser=browser, config=config, state=state)
        self._dom_hooks_attached: bool = False

    @property
    def page(self):
        """Access the current page from the browser context"""
        return self.get_current_page()

    async def _initialize_session(self):
        session = await super()._initialize_session()
        try:
            await self._attach_dom_snapshot_hooks()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to attach DOM snapshot hooks: {exc}")
        return session

    async def _attach_dom_snapshot_hooks(self):
        if self._dom_hooks_attached:
            return

        if not self.session or not self.session.context:
            return

        context = self.session.context

        # Listen for newly opened pages (tabs, popups, redirects spawning new windows)
        context.on('page', lambda page: asyncio.create_task(self._handle_new_page(page)))

        # Wire existing pages
        for page in context.pages:
            await self._wire_page_navigation(page, reason="context_initialized")

        self._dom_hooks_attached = True

    async def _handle_new_page(self, page):
        try:
            await self._wire_page_navigation(page, reason="new_page_opened")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to wire DOM snapshot hooks for new page: {exc}")

    async def _wire_page_navigation(self, page, reason: str = "navigation"):
        if hasattr(page, "_dom_snapshot_hooks_installed"):
            return

        setattr(page, "_dom_snapshot_hooks_installed", True)

        async def handle_navigation(frame):
            try:
                frame_page = getattr(frame, "page", None) or page
                if frame_page and frame == frame_page.main_frame:
                    await capture_dom_snapshot(frame_page, reason="framenavigated")
            except Exception as exc:  # pragma: no cover
                logger.debug(f"DOM snapshot navigation hook skipped: {exc}")

        async def handle_load():
            try:
                await capture_dom_snapshot(page, reason="page_load")
            except Exception as exc:  # pragma: no cover
                logger.debug(f"DOM snapshot load hook skipped: {exc}")

        page.on('framenavigated', lambda frame: asyncio.create_task(handle_navigation(frame)))
        page.on('load', lambda: asyncio.create_task(handle_load()))

        try:
            await page.wait_for_load_state('load')
        except Exception:
            pass

        await capture_dom_snapshot(page, reason=reason)
