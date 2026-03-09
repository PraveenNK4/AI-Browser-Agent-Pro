"""
Chrome Frame Overlay — wraps a raw headless screenshot in a realistic Chrome UI.

When running headless, Playwright captures only page content (no tabs, address bar,
lock icon). This module adds a pixel-perfect Chrome 2025 frame around those
screenshots so they look identical to headed captures.

Usage:
    from src.utils.chrome_frame import apply_chrome_frame

    # Inside an async context with a Playwright browser already running:
    framed_b64 = await apply_chrome_frame(
        browser=browser,          # Playwright Browser object
        screenshot_b64=raw_b64,   # base64-encoded PNG/JPEG
        url="https://example.com",
        title="Example Domain",
    )
"""
import base64
import datetime
import io
import time as _t
from urllib.parse import urlparse

# ─── Timezone shorthand ──────────────────────────────────────────────────────
def _short_tz() -> str:
    """Return short timezone abbreviation (IST, PST, EST, etc.)."""
    tzn = _t.tzname[_t.daylight] if _t.daylight else _t.tzname[0]
    return "".join(w[0] for w in tzn.split()) if len(tzn) > 5 else tzn


# ─── HTML builder ─────────────────────────────────────────────────────────────
def build_chrome_frame_html(
    page_png_b64: str,
    url: str,
    title: str,
    favicon_data_uri: str = "",
    is_secure: bool = True,
    page_width: int = 1280,
    page_height: int = 900,
    timestamp: str = "",
) -> str:
    """Build a complete HTML document that looks like Chrome wrapping a page screenshot."""
    hostname  = urlparse(url).hostname or url
    lock_col  = "#188038" if is_secure else "#c5221f"
    ts_label  = timestamp or datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S {_short_tz()}")
    tab_title = title[:30] + "…" if len(title) > 32 else title
    path_part = urlparse(url).path or "/"

    # Lock / warning SVG
    if is_secure:
        lock_svg = f'<svg viewBox="0 0 16 16" width="14" height="14" fill="{lock_col}" style="flex-shrink:0"><path d="M8 1a3.5 3.5 0 0 0-3.5 3.5V6H4a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1h-.5V4.5A3.5 3.5 0 0 0 8 1zm2 5H6V4.5a2 2 0 1 1 4 0V6z"/></svg>'
    else:
        lock_svg = f'<svg viewBox="0 0 16 16" width="14" height="14" fill="{lock_col}" style="flex-shrink:0"><path d="M8 1a3.5 3.5 0 0 0-3.5 3.5V6H4a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1h-.5V4.5a3.5 3.5 0 0 0-1-2.45V6H6V4.5a2 2 0 1 1 4 0V6z"/></svg>'

    # Favicon
    if favicon_data_uri:
        favicon_html = f'<img src="{favicon_data_uri}" width="16" height="16" style="flex-shrink:0;border-radius:2px" onerror="this.style.display=\'none\'">'
    else:
        favicon_html = '<svg viewBox="0 0 16 16" width="16" height="16" fill="#5f6368" style="flex-shrink:0"><circle cx="8" cy="8" r="7" fill="none" stroke="#5f6368" stroke-width="1.2"/><ellipse cx="8" cy="8" rx="3" ry="7" fill="none" stroke="#5f6368" stroke-width="1.2"/><line x1="1" y1="8" x2="15" y2="8" stroke="#5f6368" stroke-width="1.2"/></svg>'

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<style>
  html, body {{ overflow: hidden; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0;
       font-family: 'Segoe UI', -apple-system, system-ui, sans-serif; }}
  body {{ background: #e8eaed; display: inline-flex; flex-direction: column; }}
  .window {{ display: flex; flex-direction: column; width: {page_width + 2}px;
             box-shadow: 0 2px 16px rgba(0,0,0,.25); border-radius: 8px 8px 0 0;
             overflow: hidden; border: 1px solid #b0b0b0; }}

  /* ── Tab bar ── */
  .tabbar {{ background: #dee1e6; display: flex; align-items: flex-end;
             height: 40px; padding-left: 10px; flex-shrink: 0; }}
  .tab {{ background: #fff; height: 34px; padding: 0 12px;
          min-width: 180px; max-width: 240px;
          display: flex; align-items: center; gap: 8px;
          border-radius: 8px 8px 0 0; font-size: 12px; color: #202124;
          flex-shrink: 0; position: relative; }}
  .tab::before {{ content: ''; position: absolute; bottom: 0; left: -8px;
                  width: 8px; height: 8px; background: transparent;
                  border-bottom-right-radius: 8px;
                  box-shadow: 2px 2px 0 2px #fff; }}
  .tab::after {{ content: ''; position: absolute; bottom: 0; right: -8px;
                 width: 8px; height: 8px; background: transparent;
                 border-bottom-left-radius: 8px;
                 box-shadow: -2px 2px 0 2px #fff; }}
  .tab-title {{ flex: 1; overflow: hidden; white-space: nowrap;
                text-overflow: ellipsis; font-size: 12px; }}
  .tab-x {{ color: #5f6368; font-size: 16px; flex-shrink: 0; width: 18px; height: 18px;
             display: flex; align-items: center; justify-content: center;
             border-radius: 50%; line-height: 1; }}
  .newtab {{ background: none; border: none; width: 28px; height: 28px;
             margin: 0 2px; align-self: center; cursor: pointer;
             display: flex; align-items: center; justify-content: center; }}

  /* ── Window controls (Windows 10/11 style) ── */
  .wc {{ margin-left: auto; display: flex; height: 40px; align-self: flex-start; flex-shrink: 0; }}
  .wb {{ width: 46px; height: 32px; border: none; background: none;
         cursor: pointer; display: flex; align-items: center; justify-content: center; }}
  .wb svg {{ fill: #3c4043; width: 10px; height: 10px; }}
  .wb:hover {{ background: rgba(0,0,0,.08); }}
  .wb.x:hover {{ background: #c42b1c; }}
  .wb.x:hover svg {{ fill: #fff; }}

  /* ── Toolbar ── */
  .toolbar {{ background: #fff; height: 40px; display: flex; align-items: center;
              padding: 0 8px; gap: 4px; border-bottom: 1px solid #dadce0; flex-shrink: 0; }}
  .nav-btn {{ width: 30px; height: 30px; border-radius: 50%; border: none;
              background: none; cursor: pointer;
              display: flex; align-items: center; justify-content: center; }}
  .nav-btn svg {{ fill: #5f6368; width: 16px; height: 16px; }}
  .nav-btn.off svg {{ fill: #c4c7c5; }}

  .omni {{ flex: 1; height: 32px; background: #f1f3f4; border-radius: 24px;
            display: flex; align-items: center; gap: 8px; padding: 0 14px; margin: 0 4px; }}
  .lock-icon {{ flex-shrink: 0; display: flex; align-items: center; }}
  .addr {{ font-size: 14px; color: #202124; flex: 1;
           white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .addr-host {{ color: #202124; }}
  .addr-rest {{ color: #5f6368; }}

  .tb-btn {{ width: 30px; height: 30px; border-radius: 50%; border: none;
             background: none; cursor: pointer;
             display: flex; align-items: center; justify-content: center; }}
  .tb-btn svg {{ fill: #5f6368; width: 18px; height: 18px; }}

  .page img {{ display: block; }}

  .status {{ background: #f8f9fa; height: 22px; display: flex; align-items: center;
             padding: 0 10px; font-size: 11px; color: #80868b;
             border-top: 1px solid #e8eaed; flex-shrink: 0; justify-content: space-between; }}
</style>
</head>
<body>
<div class="window">

  <div class="tabbar">
    <div class="tab">
      {favicon_html}
      <span class="tab-title">{tab_title}</span>
      <span class="tab-x">×</span>
    </div>
    <button class="newtab">
      <svg viewBox="0 0 16 16" width="16" height="16"><path d="M8 1v14M1 8h14" stroke="#5f6368" stroke-width="1.5" fill="none" stroke-linecap="round"/></svg>
    </button>
    <div class="wc">
      <button class="wb" title="Minimize">
        <svg viewBox="0 0 10 1"><rect width="10" height="1" fill="#3c4043"/></svg>
      </button>
      <button class="wb" title="Maximize">
        <svg viewBox="0 0 10 10"><rect x=".5" y=".5" width="9" height="9" fill="none" stroke="#3c4043" stroke-width="1"/></svg>
      </button>
      <button class="wb x" title="Close">
        <svg viewBox="0 0 10 10"><path d="M0 0L10 10M10 0L0 10" stroke="#3c4043" stroke-width="1.2"/></svg>
      </button>
    </div>
  </div>

  <div class="toolbar">
    <button class="nav-btn off" title="Back">
      <svg viewBox="0 0 24 24"><path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/></svg>
    </button>
    <button class="nav-btn off" title="Forward">
      <svg viewBox="0 0 24 24"><path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8z"/></svg>
    </button>
    <button class="nav-btn" title="Reload">
      <svg viewBox="0 0 24 24"><path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>
    </button>

    <div class="omni">
      <span class="lock-icon">{lock_svg}</span>
      <div class="addr"><span class="addr-host">{hostname}</span><span class="addr-rest">{path_part}</span></div>
    </div>

    <button class="tb-btn" title="Bookmark">
      <svg viewBox="0 0 24 24"><path d="M17 3H7c-1.1 0-2 .9-2 2v16l7-3 7 3V5c0-1.1-.9-2-2-2zm0 15l-5-2.18L7 18V5h10v13z"/></svg>
    </button>
    <button class="tb-btn" title="Extensions">
      <svg viewBox="0 0 24 24"><path d="M20.5 11H19V7c0-1.1-.9-2-2-2h-4V3.5a2.5 2.5 0 00-5 0V5H4c-1.1 0-2 .9-2 2v3.8h1.5c1.5 0 2.7 1.2 2.7 2.7s-1.2 2.7-2.7 2.7H2V20c0 1.1.9 2 2 2h3.8v-1.5c0-1.5 1.2-2.7 2.7-2.7s2.7 1.2 2.7 2.7V22H17c1.1 0 2-.9 2-2v-4h1.5a2.5 2.5 0 000-5z"/></svg>
    </button>
    <button class="tb-btn" title="Chrome menu">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="5" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="12" cy="19" r="2"/></svg>
    </button>
  </div>

  <div class="page">
    <img src="data:image/png;base64,{page_png_b64}" width="{page_width}" height="{page_height}">
  </div>

  <div class="status">
    <span>{url}</span>
    <span>Captured {ts_label}</span>
  </div>

</div>
</body>
</html>"""


# ─── Main API ─────────────────────────────────────────────────────────────────
async def apply_chrome_frame(
    browser,
    screenshot_b64: str,
    url: str,
    title: str = "",
    favicon_b64: str = "",
    is_secure: bool = True,
    page_width: int = 1280,
    page_height: int = 900,
) -> str:
    """
    Wrap a headless screenshot in a Chrome frame and return as base64 PNG.

    Args:
        browser:        Playwright Browser (or BrowserContext) object — reuses same instance
        screenshot_b64: base64-encoded raw page screenshot (PNG or JPEG)
        url:            current page URL (shown in address bar)
        title:          page title (shown in tab)
        favicon_b64:    optional base64 favicon data URI
        is_secure:      True for HTTPS lock icon, False for warning
        page_width:     width of the raw screenshot
        page_height:    height of the raw screenshot

    Returns:
        base64-encoded PNG of the framed screenshot
    """
    html = build_chrome_frame_html(
        page_png_b64=screenshot_b64,
        url=url,
        title=title or url,
        favicon_data_uri=favicon_b64,
        is_secure=is_secure,
        page_width=page_width,
        page_height=page_height,
    )

    # Use an existing context or create a temporary one
    context = None
    own_context = False
    try:
        # If browser is a BrowserContext, use it directly
        if hasattr(browser, 'new_page'):
            context = browser
        else:
            # It's a Browser object — create a temporary context
            context = await browser.new_context()
            own_context = True

        page = await context.new_page()
        await page.set_content(html, wait_until="load")
        await page.wait_for_timeout(300)

        # Screenshot just the .window element for tight cropping
        window_el = page.locator(".window")
        screenshot_bytes = await window_el.screenshot(type="png")

        await page.close()
        if own_context:
            await context.close()

        return base64.b64encode(screenshot_bytes).decode("utf-8")

    except Exception as e:
        print(f"[chrome_frame] Frame rendering failed: {e}")
        # Fallback: return original screenshot unchanged
        return screenshot_b64


async def get_page_metadata(page) -> dict:
    """
    Extract metadata from a Playwright page for framing.

    Returns dict with: url, title, is_secure, width, height, favicon_b64
    """
    url = page.url or ""
    title = await page.title() or url

    # Detect HTTPS
    is_secure = url.startswith("https://")

    # Get viewport size
    vp = page.viewport_size or {}
    width = vp.get("width", 1280)
    height = vp.get("height", 900)

    # Try to get favicon
    favicon_b64 = ""
    try:
        favicon_b64 = await page.evaluate("""() => {
            const link = document.querySelector('link[rel*="icon"]');
            if (!link) return '';
            return link.href || '';
        }""")
        # If it's a URL (not already base64), fetch it
        if favicon_b64 and not favicon_b64.startswith("data:"):
            try:
                resp = await page.context.request.get(favicon_b64)
                if resp.ok:
                    body = await resp.body()
                    content_type = resp.headers.get("content-type", "image/x-icon")
                    favicon_b64 = f"data:{content_type};base64,{base64.b64encode(body).decode()}"
                else:
                    favicon_b64 = ""
            except Exception:
                favicon_b64 = ""
    except Exception:
        favicon_b64 = ""

    return {
        "url": url,
        "title": title,
        "is_secure": is_secure,
        "width": width,
        "height": height,
        "favicon_b64": favicon_b64,
    }
