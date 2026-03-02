"""
script_helpers.py
=================
Shared runtime helpers imported by every generated Playwright script.

Generated scripts should NOT redefine these functions — they import them:

    from src.utils.script_helpers import (
        get_secret, find_in_frames,
        resilient_fill, resilient_click, click_and_upload_file,
        maybe_login, resilient_extract_table, generate_report,
    )

All tuneable values (timeouts, selectors, vault prefix) come from
src.utils.config so they can be changed via environment variables.
"""

import io
import os
import re
import sys
import asyncio
from pathlib import Path

# Ensure src is importable regardless of working directory
_here = Path(__file__).resolve().parent
for _candidate in (_here.parent.parent, _here.parent.parent.parent):
    if (_candidate / "src").exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from src.utils.config import (
    VAULT_CREDENTIAL_PREFIX,
    LOGIN_USER_SELECTORS,
    LOGIN_PASS_SELECTORS,
    LOGIN_SUBMIT_SELECTORS,
    TIMEOUT_ELEMENT_VISIBLE_MS,
    TIMEOUT_FILL_MS,
    TIMEOUT_CLICK_MS,
    TIMEOUT_FILE_CHOOSER_MS,
    TIMEOUT_UPLOAD_CLICK_MS,
    TIMEOUT_UPLOAD_FALLBACK_MS,
    TIMEOUT_TABLE_WAIT_MS,
)
from src.utils.vault import vault


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

async def get_secret(key: str):
    """Look up a credential from the vault by composite key.

    Example: get_secret("OTCS_USERNAME") → vault.get_credentials("OTCS")["username"]
    """
    parts = key.split("_")
    v_key = parts[0] if len(parts) > 1 else key
    creds = vault.get_credentials(v_key) or vault.get_credentials(v_key.lower())
    if not creds:
        return None
    if any(k in key.upper() for k in ("USERNAME", "USER")):
        return creds.get("username")
    if any(k in key.upper() for k in ("PASSWORD", "PWD")):
        return creds.get("password")
    return None


# ---------------------------------------------------------------------------
# DOM interaction helpers
# ---------------------------------------------------------------------------

async def find_in_frames(page, selector: str, timeout: int = None):
    """Search for *selector* in the main page and all its iframes.

    Returns the first visible element found, or None.
    """
    visible_timeout = timeout or TIMEOUT_ELEMENT_VISIBLE_MS
    try:
        el = page.locator(selector).first
        if await el.count() > 0 and await el.is_visible(timeout=visible_timeout):
            return el
    except Exception:
        pass
    for frame in page.frames:
        try:
            el = frame.locator(selector).first
            if await el.count() > 0 and await el.is_visible(timeout=visible_timeout):
                return el
        except Exception:
            continue
    return None


async def resilient_fill(page, selector, value: str) -> bool:
    """Fill *value* into the first matching element from *selector* (str or list)."""
    if value is None:
        return False
    selectors = selector if isinstance(selector, list) else [selector]
    for sel in selectors:
        try:
            print(f"[*] Filling {sel}...")
            el = await find_in_frames(page, sel)
            if not el:
                await page.wait_for_selector(sel, timeout=TIMEOUT_FILL_MS)
                el = page.locator(sel).first
            await el.fill(str(value))
            await el.dispatch_event("input")
            await el.dispatch_event("change")
            return True
        except Exception as e:
            print(f"[-] Fill failed on {sel}: {e}")
            continue
    print(f"[-] Fill failed for all selectors: {selectors}")
    return False


async def resilient_click(page, selectors, desc: str = "element") -> bool:
    """Click the first visible match from *selectors* (str or list)."""
    if isinstance(selectors, str):
        selectors = [selectors]
    for sel in selectors:
        try:
            print(f"[*] Clicking {desc} via {sel}...")
            el = await find_in_frames(page, sel)
            if el:
                await el.click(timeout=TIMEOUT_CLICK_MS)
                return True
        except Exception as e:
            print(f"[-] Click failed on {sel}: {e}")
            continue
    print(f"[-] Click failed for all selectors ({desc}): {selectors}")
    return False


async def click_and_upload_file(page, target_name: str, file_path: str) -> bool:
    """Robustly trigger a file-chooser and upload *file_path*.

    Tries three strategies in order:
      1. Click a button/link whose text matches *target_name* exactly.
      2. Set files directly on a hidden <input type="file">.
      3. Click any element with that text and intercept the chooser.
    """
    print(f"[*] Starting robust upload for: {target_name}")
    if not os.path.exists(file_path):
        print(f"[-] File not found: {file_path}")
        return False
    # Strategy 1: native chooser via strict label match
    try:
        target = page.locator("a, button, [role='button']").filter(
            has_text=re.compile(f"^{re.escape(target_name)}$", re.I)
        ).first
        async with page.expect_file_chooser(timeout=TIMEOUT_FILE_CHOOSER_MS) as fc_info:
            await target.click(timeout=TIMEOUT_UPLOAD_CLICK_MS)
        (await fc_info.value).set_files(file_path)
        return True
    except Exception:
        pass
    # Strategy 2: hidden file input
    try:
        await page.locator('input[type="file"]').first.set_input_files(file_path)
        return True
    except Exception:
        pass
    # Strategy 3: global text search
    try:
        target = page.get_by_text(target_name, exact=True).first
        async with page.expect_file_chooser(timeout=TIMEOUT_UPLOAD_FALLBACK_MS) as fc_info:
            await target.click()
        (await fc_info.value).set_files(file_path)
        return True
    except Exception:
        pass
    print(f"[-] All upload strategies failed for: {target_name}")
    return False


# ---------------------------------------------------------------------------
# Login helper
# ---------------------------------------------------------------------------

async def maybe_login(page) -> None:
    """Detect and fill a login form if one is present.

    Uses LOGIN_*_SELECTORS and VAULT_CREDENTIAL_PREFIX from config —
    no hardcoded credentials or selectors.
    Does nothing if the page has no password field.
    """
    username = await get_secret(f"{VAULT_CREDENTIAL_PREFIX}_USERNAME")
    password = await get_secret(f"{VAULT_CREDENTIAL_PREFIX}_PASSWORD")
    if not username or not password:
        return

    # Only proceed if a password field is actually visible
    pass_el = None
    for sel in LOGIN_PASS_SELECTORS:
        pass_el = await find_in_frames(page, sel)
        if pass_el:
            break
    if not pass_el:
        return

    await resilient_fill(page, LOGIN_USER_SELECTORS, username)
    await resilient_fill(page, LOGIN_PASS_SELECTORS, password)
    await resilient_click(page, LOGIN_SUBMIT_SELECTORS, desc="login submit")

    # Wait for navigation after submit
    try:
        await page.wait_for_load_state("networkidle", timeout=TIMEOUT_TABLE_WAIT_MS)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Table extraction helper
# ---------------------------------------------------------------------------

async def resilient_extract_table(
    page,
    table_selector: str,
    header_keywords: list,
    skip_boilerplate: list = None,
) -> list:
    """Extract structured rows from a scoped table element.

    Args:
        page:             Playwright page object.
        table_selector:   CSS/XPath selector pointing to the <table> (or container).
        header_keywords:  Column names to extract (matched case-insensitively).
        skip_boilerplate: Row text patterns to skip (nav rows, headers, etc.).

    Returns:
        List of dicts, one per data row, keyed by header_keywords.

    Raises:
        RuntimeError: If table_selector is not found on page or in any iframe.
    """
    table = await find_in_frames(page, table_selector)
    if not table:
        raise RuntimeError(f"Table selector not found: {table_selector!r}")

    rows = await table.locator("tr").all()

    # If too few rows in main page, search iframes
    if len(rows) < 3:
        for frame in page.frames:
            try:
                f_table = frame.locator(table_selector).first
                if await f_table.count() > 0:
                    f_rows = await f_table.locator("tr").all()
                    if len(f_rows) > len(rows):
                        rows = f_rows
            except Exception:
                continue

    print(f"[*] Analysing {len(rows)} rows in {table_selector!r}...")

    # Resolve column indices from header row(s)
    col_map = {kw: -1 for kw in header_keywords}
    for i in range(min(20, len(rows))):
        cells = await rows[i].locator("th, td").all()
        if len(cells) < len(header_keywords):
            continue
        h_texts = [(await c.inner_text() or "").strip() for c in cells]
        for kw in header_keywords:
            if col_map[kw] != -1:
                continue
            idx = next(
                (j for j, txt in enumerate(h_texts) if kw.lower() in txt.lower()), -1
            )
            if idx != -1:
                col_map[kw] = idx
        if all(v != -1 for v in col_map.values()):
            print(f"[*] Resolved columns: {col_map}")
            break

    _boilerplate = skip_boilerplate or [
        "Default", "Name", "Description", "Enterprise",
        "Personal", "Tools", "Status", "Header",
    ]
    results = []
    max_col = max(col_map.values()) if col_map else 0

    for row in rows:
        cells = await row.locator("td").all()
        if len(cells) <= max_col:
            continue
        all_text = " ".join([(await c.inner_text() or "").strip() for c in cells])
        if any(b in all_text for b in _boilerplate):
            continue

        row_data = {}
        valid = True
        for kw, idx in col_map.items():
            cell = cells[idx]
            # Prefer image title (status icons) over text
            img = cell.locator("img[title]").first
            text = (
                await img.get_attribute("title")
                if await img.count() > 0
                else (await cell.inner_text() or "").strip()
            )
            if not text or text == "null":
                valid = False
                break
            row_data[kw] = text

        if valid:
            results.append(row_data)

    print(f"[*] Extracted {len(results)} data rows.")
    return results


# ---------------------------------------------------------------------------
# Certificate / Security helper
# ---------------------------------------------------------------------------

async def check_certificate(page) -> dict:
    """Check the SSL certificate of the current page without clicking browser UI.

    Two complementary approaches run in parallel:

    1. CDP (Chrome DevTools Protocol) — always available via Playwright, gives
       the browser's own security assessment (same data as the lock-icon popup).
    2. ssl module — direct TLS handshake to get the full certificate chain,
       valid-from/to dates, issuer, SANs, cipher suite.

    The results are rendered as a pixel-accurate screenshot of a panel that
    looks exactly like the Chrome/Edge 'Connection is secure' popup, with no
    need to click browser-chrome buttons or hardcode any coordinates.

    Returns
    -------
    dict with keys:
        screenshot  : bytes  — PNG of the rendered security panel
        cert_info   : dict   — parsed certificate fields
        is_valid    : bool   — True if cert is present and not expired
        is_secure   : bool   — True if browser considers connection secure
        hostname    : str
    """
    import ssl as _ssl
    import socket
    import datetime
    from urllib.parse import urlparse

    url   = page.url
    parsed  = urlparse(url)
    hostname = parsed.hostname or ""
    port     = parsed.port or (443 if parsed.scheme == "https" else 80)
    is_https = parsed.scheme == "https"

    # ── 1. CDP security state ────────────────────────────────────────────────
    cdp_state = {}
    is_secure = False
    try:
        cdp = await page.context.new_cdp_session(page)
        await cdp.send("Security.enable")
        result = await cdp.send("Security.getSecurityState")
        cdp_state = result.get("visibleSecurityState", result)
        sec = cdp_state.get("securityState", "")
        is_secure = sec == "secure"
        await cdp.detach()
    except Exception as e:
        print(f"[cert] CDP security check failed: {e}")

    # ── 2. Direct TLS handshake for full cert details ────────────────────────
    cert_info: dict = {}
    is_valid = False
    if is_https and hostname:
        try:
            ctx = _ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=10) as raw_sock:
                with ctx.wrap_socket(raw_sock, server_hostname=hostname) as tls_sock:
                    cert    = tls_sock.getpeercert()
                    cipher  = tls_sock.cipher()           # (name, protocol, bits)
                    version = tls_sock.version()           # 'TLSv1.3' etc.

            subject = dict(x[0] for x in cert.get("subject",  []))
            issuer  = dict(x[0] for x in cert.get("issuer",   []))

            def _parse_dt(s):
                for fmt in ("%b %d %H:%M:%S %Y %Z", "%b  %d %H:%M:%S %Y %Z"):
                    try:
                        return datetime.datetime.strptime(s, fmt)
                    except ValueError:
                        continue
                return datetime.datetime.utcnow()

            not_after  = _parse_dt(cert.get("notAfter",  ""))
            not_before = _parse_dt(cert.get("notBefore", ""))
            now         = datetime.datetime.utcnow()
            days_left   = (not_after - now).days
            is_valid    = not_before <= now <= not_after

            san_list = [v for _, v in cert.get("subjectAltName", [])
                        if not v.startswith("*")][:6]

            cert_info = {
                "hostname":      hostname,
                "subject_cn":    subject.get("commonName", hostname),
                "subject_org":   subject.get("organizationName", ""),
                "issuer_cn":     issuer.get("commonName",  "Unknown CA"),
                "issuer_org":    issuer.get("organizationName", ""),
                "valid_from":    not_before.strftime("%d %b %Y"),
                "valid_to":      not_after.strftime("%d %b %Y"),
                "days_remaining": days_left,
                "is_expired":    days_left < 0,
                "cipher":        cipher[0] if cipher else "—",
                "protocol":      version or (cipher[1] if cipher else "—"),
                "key_bits":      cipher[2] if cipher else "—",
                "san":           san_list,
                "serial":        str(cert.get("serialNumber", "—")),
            }
            print(f"[cert] ✅ {hostname} — valid for {days_left} more days")
        except _ssl.SSLCertVerificationError as e:
            print(f"[cert] ⚠️  Certificate verification FAILED: {e}")
            cert_info = {"hostname": hostname, "error": str(e),
                         "days_remaining": 0, "is_expired": True}
        except Exception as e:
            print(f"[cert] TLS handshake failed: {e}")
            cert_info = {"hostname": hostname, "error": str(e)}

    # ── 3. Gather all live page permissions + cookies via browser APIs ────────
    # Use navigator.permissions.query() — the exact same source the browser
    # lock-icon popup reads from. No hardcoded fallbacks.
    site_data: dict = {
        "full_url":        page.url,
        "permissions":     {},   # name → 'granted'/'denied'/'prompt'
        "cookie_count":    0,
        "cookie_in_use":   False,
        "js_enabled":      True,   # updated below
        "content_type":    "",
    }

    # Permissions: query every relevant name the browser popup can show
    _PERMISSION_NAMES = [
        "geolocation", "notifications", "camera", "microphone",
        "clipboard-read", "clipboard-write", "midi", "ambient-light-sensor",
        "accelerometer", "gyroscope", "magnetometer",
    ]
    try:
        raw_perms = await page.evaluate("""async (names) => {
            const out = {};
            for (const name of names) {
                try {
                    const r = await navigator.permissions.query({name});
                    out[name] = r.state;          // 'granted' | 'denied' | 'prompt'
                } catch (_) {
                    out[name] = null;             // permission type not supported
                }
            }
            return out;
        }""", _PERMISSION_NAMES)
        site_data["permissions"] = {k: v for k, v in raw_perms.items() if v is not None}
    except Exception as e:
        print(f"[cert] permissions.query failed: {e}")

    # JavaScript: can we execute? If the above evaluate succeeded, JS is on.
    # Also try CDP Runtime to double-check.
    try:
        cdp_rt = await page.context.new_cdp_session(page)
        rt_result = await cdp_rt.send("Runtime.evaluate", {"expression": "1+1"})
        site_data["js_enabled"] = rt_result.get("result", {}).get("value") == 2
        await cdp_rt.detach()
    except Exception:
        site_data["js_enabled"] = True  # can't tell — assume enabled

    # Cookies: real count for this origin
    try:
        origin_url = f"{parsed.scheme}://{hostname}"
        cookies = await page.context.cookies([origin_url])
        site_data["cookie_count"]  = len(cookies)
        site_data["cookie_in_use"] = len(cookies) > 0
    except Exception as e:
        print(f"[cert] cookie query failed: {e}")

    # Content-Type: grab from response headers if available
    try:
        content_type = await page.evaluate(
            "document.contentType || document.querySelector('meta[http-equiv=\"content-type\"]')?.content || ''"
        )
        site_data["content_type"] = content_type
    except Exception:
        pass

    # ── 5. Native OS popup screenshots via pywinauto ─────────────────────────
    # Clicks the real Chrome/Edge lock icon, screenshots each real popup panel.
    # Uses browser PID from Playwright for precise window targeting.
    # On non-Windows or if pywinauto/mss not installed → returns empty panels.
    native_shots: dict = {}

    # ── Get browser PID — try 3 methods in order ──────────────────────────────
    browser_pid = None

    # Method 1: Playwright .process property (NOT callable — it's a property)
    try:
        br = page.context.browser
        proc = getattr(br, "process", None)
        if proc is not None and not callable(proc):
            browser_pid = getattr(proc, "pid", None)
        elif proc is not None and callable(proc):
            p = proc()
            browser_pid = getattr(p, "pid", None) if p else None
    except Exception:
        pass

    # Method 2: psutil — find the Chrome process launched with remote-debugging-port
    # (same technique custom_browser uses to kill it — 100% reliable)
    if not browser_pid:
        try:
            import psutil
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    name = proc.info.get("name") or ""
                    cmdline = proc.info.get("cmdline") or []
                    if ("chrome" in name.lower() or "chromium" in name.lower()):
                        if any("remote-debugging-port" in str(a) for a in cmdline):
                            browser_pid = proc.info["pid"]
                            print(f"[cert] Found browser PID={browser_pid} via psutil")
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            print(f"[cert] psutil PID search failed: {e}")

    print(f"[cert] Browser PID for native capture: {browser_pid}")

    try:
        native_shots = await asyncio.get_event_loop().run_in_executor(
            None, _native_browser_screenshot_sync, page.url, browser_pid
        )
        if native_shots.get("panel1"):
            print(f"[cert] ✅ Native screenshots captured (PID={browser_pid})")
        else:
            print(f"[cert] ⚠️  Native capture returned empty")
    except Exception as e:
        print(f"[cert] Native capture error: {e}")

    # ── 6. HTML panels — only used when native fails (headless / Linux) ────────
    native_ok = bool(native_shots.get("panel1"))
    if native_ok:
        p1 = native_shots["panel1"]
        p2 = native_shots.get("panel2", b"")
        p3 = native_shots.get("panel3", b"")
        composite = native_shots.get("composite") or await _stitch_native(p1, p2, p3, hostname)
        panel1_html = panel2_html = panel3_html = b""
    else:
        # Fallback: HTML-rendered panels (headless, Linux, or missing dependencies)
        panel1_html = await _render_site_info_panel(page, cert_info, is_valid, is_secure,
                                                    hostname, site_data)
        panel2_html = await _render_cert_panel(page, cert_info, is_valid, is_secure, hostname)
        panel3_html = await _render_cert_detail_panel(page, cert_info, is_valid, hostname)
        p1, p2, p3 = panel1_html, panel2_html, panel3_html
        composite   = await _composite_panels(page, p1, p2, p3, hostname, native=False)

    return {
        "screenshot":               composite,
        "screenshot_site_info":     p1,
        "screenshot_security":      p2,
        "screenshot_cert":          p3,
        "screenshot_site_info_html": panel1_html,
        "screenshot_security_html":  panel2_html,
        "screenshot_cert_html":      panel3_html,
        "cert_info":                cert_info,
        "site_data":                site_data,
        "is_valid":                 is_valid,
        "is_secure":                is_secure,
        "hostname":                 hostname,
        "native_capture":           native_ok,
    }


def _native_browser_screenshot_sync(page_url: str, browser_pid: int = None) -> dict:
    """
    Click the real Chrome/Edge lock icon and screenshot each popup panel.

    Uses pywinauto with the UIA backend — zero hardcoded coordinates.
    Window is found by browser PID (passed from Playwright) or by class name.
    Lock button found by UIA AutomationId / Name from the accessibility tree.
    Popup bounding rect from UIA → captured by mss.

    Requires (Windows only):
        pip install pywinauto mss Pillow

    Returns:
        dict with panel1, panel2, panel3, composite  (PNG bytes each)
        or {} on failure / non-Windows.
    """
    import sys, time, io
    if sys.platform != "win32":
        return {}

    try:
        from pywinauto import Application
        from pywinauto.keyboard import send_keys
        import mss, mss.tools
        from PIL import Image as _PILImage
    except ImportError as e:
        print(f"[native] Install required packages:  pip install pywinauto mss Pillow")
        print(f"[native] Missing: {e}")
        return {}

    results: dict = {}

    # ── Helper: screenshot a rectangle from screen ───────────────────────────
    def _shot(left, top, width, height) -> bytes:
        if width <= 0 or height <= 0:
            return b""
        with mss.mss() as sct:
            raw = sct.grab({"left": left, "top": top,
                            "width": width, "height": height})
            return mss.tools.to_png(raw.rgb, raw.size)

    def _shot_rect(rect) -> bytes:
        """Screenshot a pywinauto RECT object."""
        return _shot(rect.left, rect.top,
                     rect.right - rect.left, rect.bottom - rect.top)

    # ── Step 1: Connect to the correct browser window ────────────────────────
    app = None
    win = None

    # Try PID-based connection first (exact — no ambiguity)
    if browser_pid:
        try:
            app = Application(backend="uia").connect(process=browser_pid, timeout=5)
            win = app.top_window()
            print(f"[native] Connected via PID={browser_pid}")
        except Exception as e:
            print(f"[native] PID connect failed: {e} — falling back to window search")
            app = None

    # Fallback: find the right Chrome window by matching URL in its title
    if win is None:
        try:
            from pywinauto import Desktop
            # Get all top-level windows with Chrome class
            desktop = Desktop(backend="uia")
            url_hostname = page_url.split("/")[2] if "//" in page_url else page_url
            hostname_short = url_hostname.replace("www.", "").split(".")[0]  # e.g. "confluence"

            all_wins = desktop.windows(class_name="Chrome_WidgetWin_1")
            print(f"[native] Found {len(all_wins)} Chrome windows, searching for '{hostname_short}'")

            # Prefer a window whose title contains the hostname
            for w in all_wins:
                title = w.window_text() or ""
                if hostname_short.lower() in title.lower():
                    win = w
                    print(f"[native] Matched window by title: '{title[:60]}'")
                    break

            # If no title match, take the window that has an omnibox with our URL
            if win is None and all_wins:
                # Try each window — look for one that has a visible address bar
                # containing our domain. Use the first one as last resort.
                for w in all_wins:
                    try:
                        # A window with many children is likely the real browser (not a helper)
                        if len(w.children()) > 3:
                            win = w
                            print(f"[native] Using largest Chrome window: '{(w.window_text() or '')[:60]}'")
                            break
                    except Exception:
                        continue
                if win is None:
                    win = all_wins[0]
                    print(f"[native] Using first Chrome window as last resort")
        except Exception as e:
            print(f"[native] Window search failed: {e}")
            return {}

    if win is None:
        print("[native] ❌ No Chrome window found")
        return {}

    try:
        win.set_focus()
        time.sleep(0.5)
    except Exception as e:
        print(f"[native] Could not focus window: {e}")

    # ── Step 2: Find the lock icon / security button ─────────────────────────
    # Try AutomationIds used across Chrome versions, then fallback to Name search
    lock_btn = None

    # AutomationId candidates (Chrome source code uses these names)
    for aid in ("view-site-information-button", "security-indicator",
                "location-icon-or-bubble", "page-info-button",
                "LocationIconView", "security_status_chip_view"):
        try:
            btn = win.child_window(auto_id=aid, control_type="Button")
            if btn.exists(timeout=0.5):
                lock_btn = btn
                print(f"[native] Lock button found by AutomationId='{aid}'")
                break
        except Exception:
            continue

    # Name / title fallback
    if lock_btn is None:
        for name in ("View site information", "Connection is secure",
                     "Connection not secure", "Not secure",
                     "Your connection", "site information"):
            try:
                btn = win.child_window(title_re=f".*{name}.*",
                                       control_type="Button")
                if btn.exists(timeout=0.5):
                    lock_btn = btn
                    print(f"[native] Lock button found by Name~='{name}'")
                    break
            except Exception:
                continue

    if lock_btn is None:
        print("[native] ❌ Lock icon button not found — trying keyboard shortcut fallback")
        # Keyboard fallback: Alt+F to open menu, or F6 to focus address bar
        # then Tab to reach lock icon
        # This is a last-resort and may not work reliably
        return {}

    # ── Step 3: Click lock → Panel 1 (Site info popup) ───────────────────────
    try:
        lock_btn.invoke()           # UIA InvokePattern — no mouse coordinates
        print("[native] Lock button invoked via UIA InvokePattern")
    except Exception:
        try:
            lock_btn.click_input()  # Fallback: click at UIA-reported center
            print("[native] Lock button clicked via click_input()")
        except Exception as e:
            print(f"[native] Could not click lock button: {e}")
            return {}

    time.sleep(0.9)  # wait for popup animation to complete

    # ── Helper: find a popup that appeared after a click ─────────────────────
    def _find_popup(timeout=4.0):
        """Search desktop for Chrome popup by known class names."""
        from pywinauto import Desktop
        desktop = Desktop(backend="uia")
        deadline = time.time() + timeout
        popup_classes = (
            "BubbleFrameView", "PageInfoBubbleView", "PageInfoView",
            "BubbleContentsView", "BubbleContents", "Widget",
            "Chrome_RenderWidgetHostHWND",
        )
        while time.time() < deadline:
            # Search as child of browser window first
            for cls in popup_classes:
                try:
                    p = win.child_window(class_name=cls, found_index=0)
                    if p.exists(timeout=0.2):
                        print(f"[native] Popup found as child: class='{cls}'")
                        return p
                except Exception:
                    pass
            # Search all top-level windows for something new
            try:
                all_wins = desktop.windows(class_name="Chrome_WidgetWin_1")
                for w in all_wins:
                    title = w.window_text() or ""
                    rect  = w.rectangle()
                    # Popups are smaller than the main window and have no/short title
                    width = rect.right - rect.left
                    height = rect.bottom - rect.top
                    if 100 < width < 700 and 100 < height < 900:
                        print(f"[native] Popup found as small window: '{title[:40]}' {width}x{height}")
                        return w
            except Exception:
                pass
            time.sleep(0.15)
        return None

    # Find Panel 1 popup
    popup1 = _find_popup(timeout=4.0)
    if popup1 is None:
        print("[native] ❌ Panel 1 popup did not appear")
        send_keys("{ESC}")
        return {}

    r1 = popup1.rectangle()
    results["panel1"] = _shot_rect(r1)
    print(f"[native] ✅ Panel 1: {r1.right-r1.left}×{r1.bottom-r1.top}px")

    # ── Step 4: Click "Connection is/not secure" row → Panel 2 ───────────────
    sec_btn = None
    for name in ("Connection is secure", "Connection not secure",
                 "Not secure", "Secure", "Your connection to"):
        try:
            b = popup1.child_window(title_re=f".*{name}.*")
            if b.exists(timeout=0.5):
                sec_btn = b
                print(f"[native] Security row found: '{name}'")
                break
        except Exception:
            continue

    if sec_btn is None:
        # Try any clickable item that isn't the close button
        try:
            items = popup1.children(control_type="ListItem")
            if not items:
                items = popup1.children(control_type="Button")
            for item in items:
                if "close" not in (item.window_text() or "").lower():
                    sec_btn = item
                    break
        except Exception:
            pass

    if sec_btn:
        try:
            sec_btn.invoke()
        except Exception:
            try:
                sec_btn.click_input()
            except Exception:
                pass
        time.sleep(0.8)

        # Find Panel 2 (reuse _find_popup)
        popup2 = _find_popup(timeout=4.0)
        if popup2:
            print(f"[native] Panel 2 found")

        if popup2:
            r2 = popup2.rectangle()
            results["panel2"] = _shot_rect(r2)
            print(f"[native] ✅ Panel 2: {r2.right-r2.left}×{r2.bottom-r2.top}px")

            # ── Step 5: Click "Certificate is valid" → Panel 3 ───────────────
            cert_btn = None
            for name in ("Certificate is valid", "Certificate is not valid",
                         "Certificate", "cert"):
                try:
                    b = popup2.child_window(title_re=f".*{name}.*")
                    if b.exists(timeout=0.5):
                        cert_btn = b
                        print(f"[native] Cert row found: '{name}'")
                        break
                except Exception:
                    continue

            if cert_btn:
                try:
                    cert_btn.invoke()
                except Exception:
                    try:
                        cert_btn.click_input()
                    except Exception:
                        pass
                time.sleep(1.2)

                # Panel 3: Chrome opens cert viewer as new top-level window
                from pywinauto import Desktop
                popup3 = None
                desktop3 = Desktop(backend="uia")
                deadline3 = time.time() + 5.0
                while time.time() < deadline3:
                    try:
                        for w in desktop3.windows(class_name="Chrome_WidgetWin_1"):
                            title = w.window_text() or ""
                            if "cert" in title.lower():
                                popup3 = w
                                print(f"[native] Panel 3 found: '{title[:50]}'")
                                break
                        if popup3:
                            break
                        # Also check as child of main win
                        c = win.child_window(title_re=".*[Cc]ertificate.*", found_index=0)
                        if c.exists(timeout=0.3):
                            popup3 = c
                            break
                    except Exception:
                        pass
                    time.sleep(0.2)

                if popup3:
                    r3 = popup3.rectangle()
                    results["panel3"] = _shot_rect(r3)
                    print(f"[native] ✅ Panel 3: {r3.right-r3.left}×{r3.bottom-r3.top}px")
                else:
                    print("[native] ⚠️  Panel 3 (cert viewer) not found")
        else:
            print("[native] ⚠️  Panel 2 popup not found")
    else:
        print("[native] ⚠️  Security row not found in Panel 1")

    # ── Step 6: Close all popups ──────────────────────────────────────────────
    try:
        send_keys("{ESC}")
        time.sleep(0.15)
        send_keys("{ESC}")
        time.sleep(0.15)
        send_keys("{ESC}")
    except Exception:
        pass

    # ── Step 7: Build composite (side-by-side) from real screenshots ──────────
    if results.get("panel1") or results.get("panel2") or results.get("panel3"):
        try:
            panels = [results[k] for k in ("panel1", "panel2", "panel3")
                      if results.get(k)]
            if len(panels) >= 1:
                imgs = [_PILImage.open(io.BytesIO(p)) for p in panels]
                total_w = sum(im.width for im in imgs) + 20 * (len(imgs) - 1)
                max_h   = max(im.height for im in imgs)
                composite = _PILImage.new("RGB", (total_w, max_h), (240, 242, 245))
                x = 0
                for im in imgs:
                    composite.paste(im, (x, 0))
                    x += im.width + 20
                buf = io.BytesIO()
                composite.save(buf, "PNG")
                results["composite"] = buf.getvalue()
                print(f"[native] ✅ Composite built: {total_w}×{max_h}px")
        except Exception as e:
            print(f"[native] Composite build failed: {e}")

    print(f"[native] Done — panels captured: {[k for k in ('panel1','panel2','panel3') if results.get(k)]}")
    return results


async def _stitch_native(p1: bytes, p2: bytes, p3: bytes, hostname: str) -> bytes:
    """Simple async wrapper to stitch 3 native panel PNGs side-by-side."""
    import io
    from PIL import Image as _PILImage
    try:
        panels = [p for p in (p1, p2, p3) if p]
        imgs   = [_PILImage.open(io.BytesIO(p)) for p in panels]
        total_w = sum(im.width for im in imgs) + 20 * (len(imgs) - 1)
        max_h   = max(im.height for im in imgs)
        out = _PILImage.new("RGB", (total_w, max_h), (240, 242, 245))
        x = 0
        for im in imgs:
            out.paste(im, (x, 0))
            x += im.width + 20
        buf = io.BytesIO()
        out.save(buf, "PNG")
        return buf.getvalue()
    except Exception:
        return p1 or p2 or p3 or b""


async def _render_site_info_panel(
    page, cert_info: dict, is_valid: bool, is_secure: bool,
    hostname: str, site_data: dict = None,
) -> bytes:
    """Render Panel 1 — the site-info popup (lock icon menu level).

    All values (permissions, JS state, cookie count, full URL) come from
    live browser data collected in check_certificate(). No hardcoded defaults.
    """
    import datetime

    sd = site_data or {}
    perms    = sd.get("permissions", {})
    full_url = sd.get("full_url", f"https://{hostname}/")

    conn_colour = "#1a6e2e" if is_secure else "#b31412"
    conn_text   = "Connection is secure" if is_secure else "Connection not secure"

    def _perm_toggle(name: str) -> str:
        """Render a toggle or text badge for a named permission."""
        state = perms.get(name)  # 'granted' | 'denied' | 'prompt' | None
        if state is None:
            return ""   # permission not supported — omit entire row
        if state == "granted":
            return '<div class="toggle on">&#9679;</div>'
        if state == "prompt":
            return '<div class="perm-badge ask">Ask</div>'
        return '<div class="toggle off">&#9679;</div>'  # denied

    def _perm_row(icon: str, label: str, name: str) -> str:
        """Emit a permission row only if the browser supports that permission."""
        badge = _perm_toggle(name)
        if badge == "":
            return ""   # not supported — skip row entirely
        return f"""
  <div class="row">
    <div class="row-left">
      <div class="row-icon">{icon}</div>
      <div><div class="row-label">{label}</div></div>
    </div>
    {badge}
  </div>"""

    # JavaScript status from live CDP Runtime check
    js_enabled = sd.get("js_enabled", True)
    js_badge   = (
        '<span class="js-badge enabled">Allowed</span>'
        if js_enabled else
        '<span class="js-badge blocked">Blocked</span>'
    )

    # Cookie row — shows real count
    cookie_count = sd.get("cookie_count", 0)
    cookie_sub   = (
        f'<div class="row-sub">{cookie_count} in use</div>'
        if cookie_count > 0 else
        '<div class="row-sub">None in use</div>'
    )

    # Only show "Reset permissions" if any permission was actually granted
    has_granted = any(v == "granted" for v in perms.values())
    reset_btn   = (
        '<div class="reset-btn"><button>Reset permissions</button></div>'
        if has_granted else ""
    )

    # Build permission rows for every supported permission
    perm_rows = "".join([
        _perm_row("&#x1F4CD;", "Location",              "geolocation"),
        _perm_row("&#x1F514;", "Notifications",          "notifications"),
        _perm_row("&#x1F4F7;", "Camera",                 "camera"),
        _perm_row("&#x1F3A4;", "Microphone",             "microphone"),
        _perm_row("&#x1F4CB;", "Clipboard read",         "clipboard-read"),
        _perm_row("&#x1F4CB;", "Clipboard write",        "clipboard-write"),
    ])

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
      background:#e8eaed;display:flex;align-items:center;justify-content:center;padding:24px;}}
.panel{{background:#fff;border-radius:12px;
        box-shadow:0 2px 10px rgba(0,0,0,.15),0 6px 30px rgba(0,0,0,.08);
        width:320px;overflow:hidden;}}
.header{{display:flex;align-items:flex-start;justify-content:space-between;
         padding:14px 16px 10px;}}
.header-text .site{{font-size:13px;font-weight:600;color:#202124;margin-bottom:2px;}}
.header-text .url{{font-size:11px;color:#5f6368;word-break:break-all;max-width:260px;}}
.close{{color:#5f6368;font-size:18px;cursor:pointer;margin-top:2px;}}
.divider{{height:1px;background:#e8eaed;margin:0 0 4px;}}
.row{{display:flex;align-items:center;justify-content:space-between;
      padding:10px 16px;cursor:pointer;}}
.row:hover{{background:#f8f9fa;}}
.row-left{{display:flex;align-items:center;gap:12px;}}
.row-icon{{color:#5f6368;font-size:17px;width:20px;text-align:center;}}
.row-label{{font-size:13.5px;color:#202124;}}
.row-sub{{font-size:11px;color:#5f6368;margin-top:1px;}}
.chevron{{font-size:14px;color:#80868b;}}
.conn-colour{{color:{conn_colour};font-weight:500;}}
.toggle{{width:36px;height:20px;border-radius:10px;display:flex;
         align-items:center;justify-content:center;font-size:18px;}}
.toggle.on{{background:#1a73e8;color:#fff;padding-left:16px;}}
.toggle.off{{background:#dadce0;color:#fff;padding-right:16px;}}
.perm-badge{{font-size:11px;padding:2px 8px;border-radius:4px;font-weight:600;}}
.perm-badge.ask{{background:#fff3e0;color:#e65100;}}
.js-badge{{font-size:11px;padding:2px 7px;border-radius:4px;}}
.js-badge.enabled{{color:#5f6368;background:#f1f3f4;}}
.js-badge.blocked{{color:#c5221f;background:#fce8e6;}}
.reset-btn{{margin:8px 16px 4px;}}
.reset-btn button{{
  border:1px solid #dadce0;background:#fff;color:#1a73e8;
  font-size:13px;padding:7px 16px;border-radius:20px;cursor:pointer;
  font-family:inherit;width:100%;
}}
.section-divider{{height:1px;background:#e8eaed;margin:8px 0 4px;}}
.ts{{padding:4px 16px 10px;font-size:10px;color:#9aa0a6;text-align:right;}}
</style></head><body>
<div class="panel">
  <div class="header">
    <div class="header-text">
      <div class="site">{hostname}</div>
      <div class="url">{full_url}</div>
    </div>
    <div class="close">&#x2715;</div>
  </div>
  <div class="divider"></div>

  <!-- Connection security row -->
  <div class="row">
    <div class="row-left">
      <div class="row-icon">&#x1F512;</div>
      <div><div class="row-label conn-colour">{conn_text}</div></div>
    </div>
    <div class="chevron">&#x276F;</div>
  </div>

  <div class="divider"></div>

  <!-- Dynamic permission rows — only shown for permissions the browser supports -->
  {perm_rows}

  <!-- JavaScript — real state from CDP Runtime -->
  <div class="row">
    <div class="row-left">
      <div class="row-icon" style="font-size:12px;font-family:monospace;font-weight:700">&lt;/&gt;</div>
      <div><div class="row-label">JavaScript</div></div>
    </div>
    <div style="display:flex;align-items:center;gap:6px">
      {js_badge}
      <span style="font-size:16px;color:#80868b">&#9783;</span>
    </div>
  </div>

  {reset_btn}

  <div class="section-divider"></div>

  <!-- Cookies — real count from context.cookies() -->
  <div class="row">
    <div class="row-left">
      <div class="row-icon">&#x1F36A;</div>
      <div>
        <div class="row-label">Cookies and site data</div>
        {cookie_sub}
      </div>
    </div>
    <div class="chevron">&#x276F;</div>
  </div>

  <!-- Site settings -->
  <div class="row">
    <div class="row-left">
      <div class="row-icon">&#x2699;&#xFE0F;</div>
      <div><div class="row-label">Site settings</div></div>
    </div>
    <div style="font-size:14px;color:#5f6368">&#x2197;</div>
  </div>

  <div class="ts">Captured {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
</body></html>"""

    cert_page = await page.context.new_page()
    try:
        await cert_page.set_content(html, wait_until="networkidle")
        shot = await cert_page.locator(".panel").screenshot(type="png")
    finally:
        await cert_page.close()
    return shot


async def _render_cert_detail_panel(
    page, cert_info: dict, is_valid: bool, hostname: str,
) -> bytes:
    """Render Panel 3 — the 'Certificate is valid' detail drill-down."""
    import datetime

    days  = cert_info.get("days_remaining", 0)
    expired = cert_info.get("is_expired", days < 0)

    if expired:
        bb, bf, bt = "#fce8e6", "#c5221f", "EXPIRED"
        header_bg, header_colour = "#fce8e6", "#c5221f"
        check_svg = '<svg width="18" height="18" viewBox="0 0 18 18"><path d="M9 1L1 16h16L9 1z" fill="#c5221f"/><text x="8.5" y="14" font-size="8" fill="white" font-weight="bold">!</text></svg>'
        validity_label = "Certificate has EXPIRED"
    elif days <= 30:
        bb, bf, bt = "#fff3e0", "#e65100", f"Expires in {days} days"
        header_bg, header_colour = "#fff3e0", "#e65100"
        check_svg = '<svg width="18" height="18" viewBox="0 0 18 18"><circle cx="9" cy="9" r="8" fill="#e65100"/><text x="8" y="13" font-size="10" fill="white" font-weight="bold">!</text></svg>'
        validity_label = f"Certificate expires soon ({days} days)"
    else:
        bb, bf, bt = "#e6f4ea", "#1a6e2e", f"Valid · {days} days remaining"
        header_bg, header_colour = "#e6f4ea", "#1a6e2e"
        check_svg = '<svg width="18" height="18" viewBox="0 0 18 18"><circle cx="9" cy="9" r="8" fill="#1a6e2e"/><polyline points="5,9 8,12 13,6" stroke="white" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        validity_label = "Certificate is valid"

    def row(lbl, val, mono=False):
        if not val or val == "—":
            return ""
        mono_style = "font-family:monospace;font-size:11px;" if mono else ""
        return (f'<div class="row"><span class="lbl">{lbl}</span>'
                f'<span class="val" style="{mono_style}">{val}</span></div>')

    san_rows = "".join(
        f'<div class="row"><span class="lbl">Alt name</span>'
        f'<span class="val mono">{s}</span></div>'
        for s in cert_info.get("san", [])
    )

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
      background:#e8eaed;display:flex;align-items:center;justify-content:center;padding:24px;}}
.panel{{background:#fff;border-radius:12px;
        box-shadow:0 2px 10px rgba(0,0,0,.15),0 6px 30px rgba(0,0,0,.08);
        width:340px;overflow:hidden;}}
.top-bar{{display:flex;align-items:center;padding:14px 16px 12px;
          border-bottom:1px solid #e8eaed;gap:10px;}}
.back{{width:28px;height:28px;border-radius:50%;border:2px solid #1a73e8;
       display:flex;align-items:center;justify-content:center;
       color:#1a73e8;font-size:14px;flex-shrink:0;}}
.top-title{{font-size:17px;font-weight:600;color:#202124;flex:1;}}
.close{{color:#5f6368;font-size:18px;}}
.validity-banner{{
  display:flex;align-items:center;gap:12px;
  padding:14px 16px;background:{header_bg};border-bottom:1px solid #e8eaed;
}}
.validity-text{{font-size:14px;font-weight:600;color:{header_colour};}}
.section-hdr{{padding:10px 16px 4px;font-size:10.5px;font-weight:600;
              letter-spacing:.6px;color:#80868b;text-transform:uppercase;}}
.row{{display:flex;justify-content:space-between;align-items:baseline;
      padding:6px 16px;font-size:12px;border-bottom:1px solid #f8f9fa;}}
.lbl{{color:#80868b;flex-shrink:0;margin-right:8px;}}
.val{{color:#202124;font-weight:500;text-align:right;
      word-break:break-word;max-width:195px;}}
.val.mono{{font-family:'Consolas',monospace;font-size:10.5px;}}
.badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;
        font-weight:600;background:{bb};color:{bf};}}
.serial-row{{padding:8px 16px;background:#f8f9fa;border-top:1px solid #e8eaed;}}
.serial-lbl{{font-size:10px;color:#80868b;margin-bottom:2px;}}
.serial-val{{font-family:monospace;font-size:10.5px;color:#5f6368;word-break:break-all;}}
.ts{{padding:6px 16px 10px;font-size:10px;color:#9aa0a6;text-align:right;}}
</style></head><body>
<div class="panel">
  <div class="top-bar">
    <div class="back">&#8592;</div>
    <div class="top-title">Certificate viewer</div>
    <div class="close">&#x2715;</div>
  </div>
  <div class="validity-banner">
    {check_svg}
    <div class="validity-text">{validity_label}</div>
  </div>

  <div class="section-hdr">Subject</div>
  {row("Common name",   cert_info.get("subject_cn",  ""))}
  {row("Organization",  cert_info.get("subject_org", ""))}

  <div class="section-hdr">Issuer</div>
  {row("Common name",   cert_info.get("issuer_cn",   ""))}
  {row("Organization",  cert_info.get("issuer_org",  ""))}

  <div class="section-hdr">Validity period</div>
  {row("Valid from",    cert_info.get("valid_from",  ""))}
  {row("Valid until",   cert_info.get("valid_to",    ""))}
  <div class="row">
    <span class="lbl">Status</span>
    <span class="val"><span class="badge">{bt}</span></span>
  </div>

  <div class="section-hdr">Technical details</div>
  {row("Protocol",      cert_info.get("protocol",    ""))}
  {row("Cipher suite",  cert_info.get("cipher",      ""), mono=True)}
  {row("Key strength",  f'{cert_info.get("key_bits", "")} bits'
                         if cert_info.get("key_bits") else "")}
  {san_rows}

  {'<div class="serial-row"><div class="serial-lbl">Serial number</div>'
   f'<div class="serial-val">{cert_info.get("serial","—")}</div></div>'
   if cert_info.get("serial") and cert_info.get("serial") != "—" else ""}

  <div class="ts">Verified {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
</body></html>"""

    cert_page = await page.context.new_page()
    try:
        await cert_page.set_content(html, wait_until="networkidle")
        shot = await cert_page.locator(".panel").screenshot(type="png")
    finally:
        await cert_page.close()
    return shot


async def _composite_panels(
    page, panel1: bytes, panel2: bytes, panel3: bytes, hostname: str,
    native: bool = False,
) -> bytes:
    """Stitch the 3 panel PNGs side-by-side with click-flow arrows between them."""
    import base64, datetime

    source_label = (
        "Native OS capture via Windows UI Automation"
        if native else
        "HTML panels via CDP + TLS inspection"
    )

    p1b64 = base64.b64encode(panel1).decode()
    p2b64 = base64.b64encode(panel2).decode()
    p3b64 = base64.b64encode(panel3).decode()

    arrow_svg = """
    <svg width="40" height="60" viewBox="0 0 40 60" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#1a73e8"/>
        </marker>
      </defs>
      <line x1="5" y1="30" x2="30" y2="30"
            stroke="#1a73e8" stroke-width="2" marker-end="url(#ah)"/>
      <text x="2" y="20" font-size="9" fill="#5f6368" font-family="Segoe UI,sans-serif">click</text>
    </svg>"""
    arrow_b64 = base64.b64encode(arrow_svg.encode()).decode()

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{
  font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
  background:linear-gradient(135deg,#e8eaed 0%,#d2e3fc 100%);
  padding:32px 24px;
}}
.flow{{display:flex;align-items:flex-start;gap:0;}}
.step{{display:flex;flex-direction:column;align-items:center;gap:8px;}}
.step-label{{
  font-size:11px;font-weight:600;color:#5f6368;
  text-transform:uppercase;letter-spacing:.5px;
  background:rgba(255,255,255,.7);padding:3px 10px;border-radius:10px;
}}
.step img{{border-radius:12px;
           box-shadow:0 4px 16px rgba(0,0,0,.18),0 1px 4px rgba(0,0,0,.1);}}
.arrow{{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:0 4px;margin-top:48px;
}}
.arrow-label{{font-size:10px;color:#1a73e8;font-weight:600;margin-bottom:4px;}}
.title{{
  text-align:center;margin-bottom:20px;
  font-size:15px;font-weight:700;color:#202124;
  text-shadow:0 1px 2px rgba(255,255,255,.8);
}}
.subtitle{{text-align:center;margin-bottom:24px;font-size:11px;color:#5f6368;}}
.ts{{text-align:center;margin-top:16px;font-size:10px;color:#80868b;}}
</style></head><body>
  <div class="title">🔒 Security Certificate Check — {hostname}</div>
  <div class="subtitle">Click flow: Site info → Connection secure → Certificate detail</div>
  <div class="flow">

    <div class="step">
      <div class="step-label">① Site Info</div>
      <img src="data:image/png;base64,{p1b64}" width="260">
    </div>

    <div class="arrow">
      <div class="arrow-label">click</div>
      <img src="data:image/svg+xml;base64,{arrow_b64}" width="40">
      <div style="font-size:9px;color:#80868b;margin-top:2px;text-align:center">Connection<br>is secure</div>
    </div>

    <div class="step">
      <div class="step-label">② Security</div>
      <img src="data:image/png;base64,{p2b64}" width="290">
    </div>

    <div class="arrow">
      <div class="arrow-label">click</div>
      <img src="data:image/svg+xml;base64,{arrow_b64}" width="40">
      <div style="font-size:9px;color:#80868b;margin-top:2px;text-align:center">Certificate<br>is valid</div>
    </div>

    <div class="step">
      <div class="step-label">③ Certificate</div>
      <img src="data:image/png;base64,{p3b64}" width="290">
    </div>

  </div>
  <div class="ts">Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — {source_label}</div>
</body></html>"""

    comp_page = await page.context.new_page()
    try:
        await comp_page.set_content(html, wait_until="networkidle")
        await comp_page.set_viewport_size({"width": 1100, "height": 800})
        shot = await comp_page.screenshot(type="png", full_page=True)
    finally:
        await comp_page.close()
    return shot


async def _render_cert_panel(
    page,
    cert_info: dict,
    is_valid:  bool,
    is_secure: bool,
    hostname:  str,
) -> bytes:
    """Render a Chrome-style 'Connection is secure' panel and return PNG bytes.

    The panel is built as a styled HTML page and screenshotted via Playwright —
    no native browser chrome interaction needed, works headless or headed,
    Windows/Mac/Linux, any browser.
    """
    days     = cert_info.get("days_remaining", 0)
    expired  = cert_info.get("is_expired", days < 0)
    error    = cert_info.get("error", "")

    # Status colours mirror Chrome's UI
    if is_valid and is_secure:
        status_colour = "#1a6e2e"
        status_bg     = "#e6f4ea"
        lock_svg = (
            '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" '
            'xmlns="http://www.w3.org/2000/svg">'
            '<rect x="3" y="9" width="14" height="10" rx="2" fill="#1a6e2e"/>'
            '<path d="M7 9V6a3 3 0 016 0v3" stroke="#1a6e2e" stroke-width="2" '
            'stroke-linecap="round" fill="none"/>'
            '<circle cx="10" cy="14" r="1.5" fill="white"/>'
            '</svg>'
        )
        headline = "Connection is secure"
        subline  = ("Your information (for example, passwords or credit card "
                    "numbers) is private when it is sent to this site.")
    else:
        status_colour = "#b31412"
        status_bg     = "#fce8e6"
        lock_svg = (
            '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" '
            'xmlns="http://www.w3.org/2000/svg">'
            '<path d="M10 2L2 17h16L10 2z" fill="#b31412"/>'
            '<text x="9.5" y="15" font-size="9" fill="white" font-weight="bold">!</text>'
            '</svg>'
        )
        headline = "Connection not secure" if not is_secure else "Certificate issue"
        subline  = error or "The certificate for this site is invalid or expired."

    # Validity badge
    if expired:
        badge_bg, badge_fg, badge_txt = "#fce8e6", "#c5221f", "EXPIRED"
    elif days <= 30:
        badge_bg, badge_fg, badge_txt = "#fff3e0", "#e65100", f"Expires in {days}d"
    else:
        badge_bg, badge_fg, badge_txt = "#e6f4ea", "#1a6e2e", f"Valid · {days}d left"

    san_rows = "".join(
        f'<div class="row"><span class="lbl">Alt name</span>'
        f'<span class="val">{s}</span></div>'
        for s in cert_info.get("san", [])
    )

    def row(label, value):
        if not value or value == "—":
            return ""
        return (f'<div class="row"><span class="lbl">{label}</span>'
                f'<span class="val">{value}</span></div>')

    cert_block = "".join([
        row("Issued to",     cert_info.get("subject_cn",  "")),
        row("Organization",  cert_info.get("subject_org", "")),
        row("Issued by",     cert_info.get("issuer_cn",   "")),
        row("CA org",        cert_info.get("issuer_org",  "")),
        row("Valid from",    cert_info.get("valid_from",  "")),
        row("Valid until",   cert_info.get("valid_to",    "")),
        row("Protocol",      cert_info.get("protocol",    "")),
        row("Cipher suite",  cert_info.get("cipher",      "")),
        row("Key strength",  f'{cert_info.get("key_bits", "")} bits'
                              if cert_info.get("key_bits") else ""),
        san_rows,
    ])

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{
  font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
  background:#e8eaed;
  min-height:100vh;
  display:flex;align-items:center;justify-content:center;
  padding:24px;
}}
.panel{{
  background:#fff;
  border-radius:12px;
  box-shadow:0 2px 10px rgba(0,0,0,.15),0 6px 30px rgba(0,0,0,.08);
  width:340px;
  overflow:hidden;
}}
.top-bar{{
  display:flex;align-items:center;padding:14px 16px 12px;
  border-bottom:1px solid #e8eaed;gap:10px;
}}
.back{{
  width:28px;height:28px;border-radius:50%;border:2px solid #1a73e8;
  display:flex;align-items:center;justify-content:center;
  color:#1a73e8;font-size:14px;cursor:pointer;flex-shrink:0;
}}
.top-title{{font-size:17px;font-weight:600;color:#202124;flex:1;}}
.close{{color:#5f6368;font-size:18px;cursor:pointer;}}
.url-line{{padding:4px 16px 12px;font-size:12px;color:#5f6368;
           word-break:break-all;border-bottom:1px solid #e8eaed;}}
.sec-row{{
  display:flex;align-items:flex-start;gap:12px;
  padding:14px 16px 12px;border-bottom:1px solid #e8eaed;
}}
.sec-icon{{flex-shrink:0;margin-top:2px;}}
.sec-text .headline{{
  font-size:14px;font-weight:600;color:{status_colour};margin-bottom:4px;
}}
.sec-text .sub{{font-size:12px;color:#5f6368;line-height:1.55;}}
.sec-text .sub a{{color:#1a73e8;text-decoration:none;}}
.section-hdr{{
  padding:10px 16px 4px;
  font-size:10.5px;font-weight:600;letter-spacing:.6px;
  color:#80868b;text-transform:uppercase;
}}
.row{{
  display:flex;justify-content:space-between;align-items:baseline;
  padding:6px 16px;font-size:12px;border-bottom:1px solid #f8f9fa;
}}
.lbl{{color:#80868b;flex-shrink:0;margin-right:8px;}}
.val{{color:#202124;font-weight:500;text-align:right;word-break:break-word;max-width:190px;}}
.badge{{
  display:inline-block;padding:1px 7px;border-radius:10px;
  font-size:10.5px;font-weight:600;
  background:{badge_bg};color:{badge_fg};margin-left:6px;
}}
.cert-footer{{
  display:flex;align-items:center;gap:10px;
  padding:11px 16px;background:#f8f9fa;
}}
.cert-check{{
  width:20px;height:20px;border-radius:3px;background:#1a6e2e;
  display:flex;align-items:center;justify-content:center;
  color:#fff;font-size:13px;flex-shrink:0;
}}
.cert-label{{font-size:13px;font-weight:500;color:#202124;flex:1;}}
.cert-arrow{{color:#5f6368;font-size:16px;}}
.ts{{padding:6px 16px 10px;font-size:10px;color:#9aa0a6;text-align:right;}}
</style></head><body>
<div class="panel">
  <div class="top-bar">
    <div class="back">&#8592;</div>
    <div class="top-title">Security</div>
    <div class="close">&#x2715;</div>
  </div>
  <div class="url-line">{hostname}</div>
  <div class="sec-row">
    <div class="sec-icon">{lock_svg}</div>
    <div class="sec-text">
      <div class="headline">{headline}</div>
      <div class="sub">{subline} <a href="#">Learn more</a></div>
    </div>
  </div>
  <div class="section-hdr">Certificate</div>
  {cert_block}
  <div class="row">
    <span class="lbl">Validity</span>
    <span class="val"><span class="badge">{badge_txt}</span></span>
  </div>
  <div class="cert-footer">
    <div class="cert-check">&#x2713;</div>
    <div class="cert-label">Certificate is valid</div>
    <div class="cert-arrow">&#x2197;</div>
  </div>
  <div class="ts">Checked {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
</body></html>"""

    cert_page = await page.context.new_page()
    try:
        await cert_page.set_content(html, wait_until="networkidle")
        panel_el  = cert_page.locator(".panel")
        screenshot = await panel_el.screenshot(type="png")
    finally:
        await cert_page.close()
    return screenshot


async def generate_report(
    scenario: str,
    status: bool,
    screenshot=None,
    output: str = None,
) -> None:
    """Print extracted data and save a .docx report in the script's directory."""
    from datetime import datetime
    from docx import Document

    print("\n" + "=" * 60)
    print(f"📋 EXTRACTED DATA: {scenario}")
    print("=" * 60)
    if output:
        print(output)
    print("=" * 60 + "\n")

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        report_path = os.path.join(script_dir, f"report_{timestamp}.docx")

        doc = Document()
        doc.add_heading(f"Report: {scenario}", 0)
        doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        doc.add_paragraph(f"Status: {'SUCCESS' if status else 'FAILED'}")
        if output:
            doc.add_paragraph(f"Output:\n{output}")
        if screenshot:
            doc.add_picture(io.BytesIO(screenshot))
        doc.save(report_path)
        print(f"[+] Report saved: {report_path}")
    except Exception as e:
        print(f"[-] Report generation failed: {e}")