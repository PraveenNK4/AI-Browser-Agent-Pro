"""
POC: Chrome-frame screenshot in headless mode.

Opens Google, searches for a term, then wraps the page screenshot
inside a pixel-accurate Chrome window frame (tabs, address bar, lock icon)
and saves the composite as a PNG — all running fully headless.
"""
import asyncio
import base64
import os
import ssl
import socket
import datetime
from urllib.parse import urlparse
from playwright.async_api import async_playwright


async def capture_browser_frame(page, *, extra_tabs=None) -> bytes:
    """Wrap page.screenshot() inside a realistic Chrome window frame.

    Works in both headed AND headless mode — no OS dependencies.

    Args:
        page:       Playwright page object (already navigated).
        extra_tabs: Optional list of tab names to show in the tab bar
                    (the active tab is always the current page's title).

    Returns:
        PNG bytes of the composite "full browser window" image.
    """

    # ── Collect live data from the page ──────────────────────────────────────
    url        = page.url
    parsed     = urlparse(url)
    hostname   = parsed.hostname or ""
    title      = await page.title() or hostname
    is_https   = parsed.scheme == "https"

    # Security state via CDP (works headless)
    is_secure = False
    try:
        cdp = await page.context.new_cdp_session(page)
        await cdp.send("Security.enable")
        sec_result = await cdp.send("Security.getSecurityState")
        sec_state = sec_result.get("visibleSecurityState", sec_result)
        is_secure = sec_state.get("securityState", "") == "secure"
        await cdp.detach()
    except Exception:
        is_secure = is_https  # fallback: assume https = secure

    # Take the actual page screenshot
    viewport_png = await page.screenshot(type="png", full_page=False)
    viewport_b64 = base64.b64encode(viewport_png).decode()

    # ── Build Chrome frame HTML ──────────────────────────────────────────────

    # Detect OS for window controls style
    import platform
    is_windows = platform.system() == "Windows"

    if is_windows:
        window_controls_html = '''
        <div class="win-controls">
            <div class="win-btn minimize">&#x2500;</div>
            <div class="win-btn maximize">&#x25A1;</div>
            <div class="win-btn close">&#x2715;</div>
        </div>'''
        title_bar_style = "justify-content: flex-end;"
    else:
        window_controls_html = '''
        <div class="traffic-lights">
            <div class="dot red"></div>
            <div class="dot yellow"></div>
            <div class="dot green"></div>
        </div>'''
        title_bar_style = ""

    # Lock icon SVG
    if is_secure:
        lock_svg = '''<svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="2" y="6.5" width="10" height="7" rx="1.5" fill="#5f6368"/>
            <path d="M5 6.5V4.5a2 2 0 014 0v2" stroke="#5f6368" stroke-width="1.5"
                  stroke-linecap="round" fill="none"/>
        </svg>'''
    else:
        lock_svg = '''<svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="7" cy="7" r="6" stroke="#ea4335" stroke-width="1.5" fill="none"/>
            <text x="5.5" y="11" font-size="9" fill="#ea4335" font-weight="bold">!</text>
        </svg>'''

    # Display URL (hide scheme for https, show for http)
    display_url = url
    if is_https:
        display_url = url.replace("https://", "")

    # Tab bar: active tab + optional background tabs
    bg_tabs_html = ""
    if extra_tabs:
        for tab_name in extra_tabs:
            bg_tabs_html += f'''
            <div class="tab bg-tab">
                <div class="tab-favicon">
                    <div style="width:12px;height:12px;border-radius:50%;background:#dadce0"></div>
                </div>
                <span class="tab-title">{tab_name}</span>
                <span class="tab-close">&#x2715;</span>
            </div>'''

    # Truncate title for tab display
    tab_title = title[:40] + "…" if len(title) > 40 else title

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #f0f0f0;
    padding: 16px;
}}
.browser-window {{
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,.25), 0 2px 8px rgba(0,0,0,.15);
    background: #fff;
    max-width: 1280px;
}}

/* ── Title bar (window controls) ─────────────────────────────── */
.title-bar {{
    background: #dee1e6;
    height: 32px;
    display: flex;
    align-items: center;
    padding: 0 12px;
    gap: 8px;
    {title_bar_style}
}}
.traffic-lights {{
    display: flex;
    gap: 6px;
}}
.win-controls {{
    display: flex;
    gap: 0;
}}
.win-btn {{
    width: 46px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    color: #5f6368;
    cursor: default;
}}
.win-btn:hover {{ background: #d0d3d8; }}
.win-btn.close:hover {{ background: #e81123; color: #fff; }}
.dot {{
    width: 12px; height: 12px; border-radius: 50%;
}}
.dot.red    {{ background: #ff5f57; }}
.dot.yellow {{ background: #febc2e; }}
.dot.green  {{ background: #28c840; }}

/* ── Tab bar ─────────────────────────────────────────────────── */
.tab-bar {{
    background: #dee1e6;
    display: flex;
    align-items: flex-end;
    padding: 0 8px;
    height: 34px;
    gap: 0;
}}
.tab {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    font-size: 12px;
    color: #5f6368;
    border-radius: 8px 8px 0 0;
    max-width: 220px;
    min-width: 60px;
    height: 30px;
    cursor: default;
    position: relative;
}}
.tab.active-tab {{
    background: #fff;
    color: #202124;
    box-shadow: 0 -1px 3px rgba(0,0,0,.08);
}}
.tab.bg-tab {{
    background: #d3d6db;
}}
.tab.bg-tab:hover {{
    background: #c8cbcf;
}}
.tab-favicon {{
    flex-shrink: 0;
    width: 16px;
    height: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.tab-favicon img {{
    width: 16px;
    height: 16px;
}}
.tab-title {{
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11.5px;
}}
.tab-close {{
    font-size: 10px;
    color: #80868b;
    flex-shrink: 0;
    width: 16px;
    height: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
}}
.new-tab-btn {{
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #5f6368;
    font-size: 16px;
    border-radius: 50%;
    margin-left: 4px;
    margin-bottom: 4px;
    cursor: default;
}}

/* ── Address / toolbar bar ───────────────────────────────────── */
.toolbar {{
    background: #fff;
    display: flex;
    align-items: center;
    padding: 6px 10px;
    gap: 6px;
    border-bottom: 1px solid #e8eaed;
}}
.nav-btn {{
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #5f6368;
    font-size: 16px;
    border-radius: 50%;
    cursor: default;
    flex-shrink: 0;
}}
.nav-btn.disabled {{
    color: #bdc1c6;
}}
.omnibox {{
    flex: 1;
    display: flex;
    align-items: center;
    gap: 8px;
    background: #f1f3f4;
    border-radius: 24px;
    padding: 6px 14px;
    font-size: 13px;
    color: #202124;
    min-width: 0;
}}
.omnibox .lock {{
    flex-shrink: 0;
    display: flex;
    align-items: center;
}}
.omnibox .url {{
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: #5f6368;
}}
.omnibox .url .hostname {{
    color: #202124;
}}
.toolbar-right {{
    display: flex;
    align-items: center;
    gap: 4px;
}}
.toolbar-icon {{
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #5f6368;
    font-size: 16px;
    border-radius: 50%;
}}
.avatar {{
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4285f4, #34a853);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 12px;
    font-weight: 600;
}}

/* ── Bookmarks bar ───────────────────────────────────────────── */
.bookmarks {{
    background: #fff;
    padding: 4px 10px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid #e8eaed;
    font-size: 12px;
    color: #5f6368;
}}
.bookmark-item {{
    display: flex;
    align-items: center;
    gap: 4px;
}}
.bookmark-dot {{
    width: 10px;
    height: 10px;
    border-radius: 2px;
    background: #dadce0;
}}

/* ── Viewport ────────────────────────────────────────────────── */
.viewport {{
    line-height: 0;
}}
.viewport img {{
    width: 100%;
    display: block;
}}

/* ── Status bar ──────────────────────────────────────────────── */
.status-bar {{
    background: #f8f9fa;
    padding: 3px 10px;
    font-size: 10px;
    color: #80868b;
    border-top: 1px solid #e8eaed;
    display: flex;
    justify-content: space-between;
}}
</style>
</head>
<body>
<div class="browser-window">

    <!-- Title bar with window controls -->
    <div class="title-bar">
        {window_controls_html}
    </div>

    <!-- Tab bar -->
    <div class="tab-bar">
        {bg_tabs_html}
        <div class="tab active-tab">
            <div class="tab-favicon">
                <img src="https://www.google.com/favicon.ico"
                     onerror="this.style.display='none'" />
            </div>
            <span class="tab-title">{tab_title}</span>
            <span class="tab-close">&#x2715;</span>
        </div>
        <div class="new-tab-btn">+</div>
    </div>

    <!-- Toolbar / address bar -->
    <div class="toolbar">
        <div class="nav-btn disabled">&#8592;</div>
        <div class="nav-btn disabled">&#8594;</div>
        <div class="nav-btn">&#x21BB;</div>
        <div class="nav-btn">&#x2302;</div>

        <div class="omnibox">
            <div class="lock">{lock_svg}</div>
            <div class="url">
                <span class="hostname">{hostname}</span>{parsed.path if parsed.path != '/' else ''}{('?' + parsed.query) if parsed.query else ''}
            </div>
        </div>

        <div class="toolbar-right">
            <div class="toolbar-icon">&#x2606;</div>
            <div class="toolbar-icon">&#x22EE;</div>
            <div class="avatar">P</div>
        </div>
    </div>

    <!-- Bookmarks bar -->
    <div class="bookmarks">
        <div class="bookmark-item"><div class="bookmark-dot"></div> Bookmarks</div>
    </div>

    <!-- Page viewport (the real screenshot) -->
    <div class="viewport">
        <img src="data:image/png;base64,{viewport_b64}" />
    </div>

    <!-- Status bar -->
    <div class="status-bar">
        <span>{url}</span>
        <span>Captured {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
    </div>

</div>
</body></html>'''

    # Render the composite in a new page
    frame_page = await page.context.new_page()
    try:
        await frame_page.set_viewport_size({"width": 1300, "height": 900})
        await frame_page.set_content(html, wait_until="networkidle")
        composite = await frame_page.locator(".browser-window").screenshot(type="png")
    finally:
        await frame_page.close()

    return composite


async def main():
    async with async_playwright() as p:
        # Launch HEADLESS
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page    = await context.new_page()

        # Step 1: Go to Google
        print("[*] Navigating to Google...")
        await page.goto("https://www.google.com", timeout=60000)
        await page.wait_for_load_state("networkidle")

        # Capture frame screenshot of Google homepage
        frame1 = await capture_browser_frame(page)
        out_dir = os.path.dirname(os.path.abspath(__file__))
        path1   = os.path.join(out_dir, "frame_google_home.png")
        with open(path1, "wb") as f:
            f.write(frame1)
        print(f"[+] Frame screenshot 1 saved: {path1}")

        # Step 2: Search for something
        print("[*] Searching for 'OpenText'...")
        search = page.locator('textarea[name="q"], input[name="q"]').first
        await search.fill("OpenText")
        await search.press("Enter")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(2000)

        # Capture frame screenshot of search results
        frame2 = await capture_browser_frame(page)
        path2   = os.path.join(out_dir, "frame_google_search.png")
        with open(path2, "wb") as f:
            f.write(frame2)
        print(f"[+] Frame screenshot 2 saved: {path2}")

        await browser.close()
        print("\n✅ Done! Check the PNG files.")


if __name__ == "__main__":
    asyncio.run(main())
