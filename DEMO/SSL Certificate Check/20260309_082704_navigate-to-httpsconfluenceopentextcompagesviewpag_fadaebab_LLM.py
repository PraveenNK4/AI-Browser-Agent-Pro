import argparse
import datetime
import io
import os
import sys
import time
from urllib.parse import urlparse


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_text(ctrl) -> str:
    try:
        t = getattr(ctrl, "window_text", None)
        if not t: return ""
        val = t() if callable(t) else t
        return str(val or "").strip()
    except Exception:
        return ""


def _safe_aid(ctrl) -> str:
    try:
        a = getattr(ctrl, "automation_id", None)
        if a is None and hasattr(ctrl, "element_info"):
            a = getattr(ctrl.element_info, "automation_id", None)
        if not a: return ""
        val = a() if callable(a) else a
        return str(val or "").strip()
    except Exception:
        return ""


def _shot(left, top, width, height) -> bytes:
    import mss, mss.tools
    if width <= 0 or height <= 0:
        return b""
    with mss.mss() as sct:
        raw = sct.grab({"left": left, "top": top, "width": width, "height": height})
        return mss.tools.to_png(raw.rgb, raw.size)


def _shot_full_screen() -> bytes:
    """Take a screenshot of the FULL PRIMARY MONITOR (including taskbar)."""
    import mss, mss.tools
    with mss.mss() as sct:
        # Index 1 is the primary monitor, 0 is 'all monitors'
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        return mss.tools.to_png(raw.rgb, raw.size)


# ─────────────────────────────────────────────────────────────────────────────
# Core: click lock icon → capture 3 native Chrome panels
# ─────────────────────────────────────────────────────────────────────────────

def _force_foreground(win):
    """Bring window to the true OS foreground — stronger than pywinauto set_focus()."""
    import ctypes
    try:
        hwnd = win.handle
        # If minimized (Iconic), restore it. Otherwise just Show it so we don't un-maximize.
        if ctypes.windll.user32.IsIconic(hwnd):
            ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        else:
            ctypes.windll.user32.ShowWindow(hwnd, 5)  # SW_SHOW
        ctypes.windll.user32.SetForegroundWindow(hwnd)
        time.sleep(0.4)
    except Exception as e:
        print(f"[native] SetForegroundWindow: {e}")
    try:
        win.set_focus()
        time.sleep(0.3)
    except Exception:
        pass


def native_capture(page_url: str, chrome_pid: int = None) -> dict:
    """
    Find the Chrome window showing page_url, click the lock icon,
    and grab screenshots of the 3 native panels.
    """
    if sys.platform != "win32":
        return {}

    try:
        from pywinauto import Application, Desktop
        from pywinauto.keyboard import send_keys
    except ImportError:
        return {}

    results: dict = {}
    hostname = urlparse(page_url).hostname or page_url
    win = None

    if chrome_pid:
        try:
            app = Application(backend="uia").connect(process=chrome_pid, timeout=6)
            win = app.top_window()
        except Exception:
            win = None

    if win is None:
        desktop = Desktop(backend="uia")
        all_wins = [
            w for w in desktop.windows(class_name="Chrome_WidgetWin_1")
            if (w.rectangle().right - w.rectangle().left) > 200
        ]
        title_match = None
        for w in all_wins:
            omnibox_url = ""
            for aid in ("omnibox", "LocationBar", "address-and-search-bar"):
                try:
                    ob = w.child_window(auto_id=aid, control_type="Edit", found_index=0)
                    if ob.exists(timeout=0.5):
                        omnibox_url = ob.get_value() or ""
                        if omnibox_url: break
                except Exception: pass
            if not omnibox_url:
                try:
                    for e in w.descendants(control_type="Edit")[:8]:
                        try:
                            val = e.get_value() or ""
                            if val.startswith("http") or hostname in val:
                                omnibox_url = val
                                break
                        except Exception: pass
                except Exception: pass
            if omnibox_url and hostname in omnibox_url:
                win = w
                break
            if hostname in (_safe_text(w) or "").lower() and title_match is None:
                title_match = w
        if win is None:
            win = title_match or (all_wins[0] if all_wins else None)

    if win is None: return {}
    _force_foreground(win)

    lock_btn = None
    for aid in ("view-site-information-button", "security-indicator", "location-icon-or-bubble", 
                "page-info-button", "LocationIconView", "security_status_chip_view", 
                "PageInfoChip", "omnibox-icon"):
        try:
            btn = win.child_window(auto_id=aid)
            if btn.exists(timeout=0.5):
                lock_btn = btn
                break
        except Exception: continue

    if lock_btn is None:
        try:
            for btn in win.descendants(control_type="Button"):
                name = _safe_text(btn).lower()
                aid  = _safe_aid(btn).lower()
                if any(kw in name for kw in ("view site information", "connection is secure", "not secure", "certificate")):
                    lock_btn = btn
                    break
                if any(kw in aid for kw in ("security", "site-info", "location-icon", "page-info", "lock", "omnibox-icon")):
                    lock_btn = btn
                    break
        except Exception: pass

    if lock_btn is None: return {}

    try:
        lock_btn.click_input()
    except Exception:
        try: lock_btn.invoke()
        except Exception: return {}

    time.sleep(1.2)

    def _find_popup(timeout=4.5):
        deadline = time.time() + timeout
        main_rect = win.rectangle()
        main_w, main_h = main_rect.right - main_rect.left, main_rect.bottom - main_rect.top
        popup_classes = ("BubbleFrameView", "PageInfoBubbleView", "PageInfoView", "BubbleContentsView", "BubbleContents")
        while time.time() < deadline:
            for cls in popup_classes:
                try:
                    p = win.child_window(class_name=cls, found_index=0)
                    if p.exists(timeout=0.25): return p
                except Exception: pass
            try:
                for w in Desktop(backend="uia").windows(class_name="Chrome_WidgetWin_1"):
                    try:
                        r = w.rectangle()
                        pw, ph = r.right - r.left, r.bottom - r.top
                        if (pw < main_w - 50 and ph < main_h - 50 and 80 < pw < 800 and 80 < ph < 1000):
                            return w
                    except Exception: continue
            except Exception: pass
            time.sleep(0.15)
        return None

    popup1 = _find_popup(timeout=4.0)
    if popup1 is None:
        send_keys("{ESC}")
        return {}

    results["panel1"] = _shot_full_screen()
    sec_btn = None
    try:
        items = popup1.descendants(control_type="Button") + popup1.descendants(control_type="ListItem")
        for item in items:
            name = _safe_text(item).lower()
            if any(kw in name for kw in ("connection is secure", "connection not secure", "not secure", "your connection")):
                sec_btn = item
                break
    except Exception: pass

    if sec_btn:
        try: sec_btn.invoke()
        except Exception:
            try: sec_btn.click_input()
            except Exception: pass
        time.sleep(0.9)
        popup2 = _find_popup(timeout=4.0)
        if popup2:
            results["panel2"] = _shot_full_screen()
            cert_btn = None
            try:
                items = popup2.descendants(control_type="Button") + popup2.descendants(control_type="ListItem")
                for item in items:
                    name = _safe_text(item).lower()
                    if "certificate" in name or "cert" in name:
                        cert_btn = item
                        break
            except Exception: pass
            if cert_btn:
                try: cert_btn.invoke()
                except Exception:
                    try: cert_btn.click_input()
                    except Exception: pass
                time.sleep(1.4)
                popup3 = None
                deadline3 = time.time() + 5.0
                while time.time() < deadline3:
                    try:
                        for w in Desktop(backend="uia").windows(class_name="Chrome_WidgetWin_1"):
                            if "certificate viewer" in _safe_text(w).lower() and w.process_id() == win.process_id():
                                popup3 = w
                                break
                        if popup3: break
                        c = win.child_window(title_re=".*[Cc]ertificate.*", found_index=0)
                        if c.exists(timeout=0.3):
                            popup3 = c
                            break
                    except Exception: pass
                    time.sleep(0.2)
                if popup3:
                    results["panel3"] = _shot_full_screen()

    for _ in range(3):
        try:
            send_keys("{ESC}")
            time.sleep(0.15)
        except Exception: pass

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if sys.platform != "win32":
        print("[ERROR] This script requires Windows (UIAutomation is Win32 only).")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Native Chrome SSL certificate screenshot via UIAutomation (Windows)"
    )
    parser.add_argument("--url", action="append", dest="urls", metavar="URL",
                        help="HTTPS URL to inspect (repeatable)")
    args = parser.parse_args()

    urls = args.urls or ["https://confluence.opentext.com"]

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir     = os.path.join(_script_dir, "screenshots")
    os.makedirs(out_dir, exist_ok=True)

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    opts.add_argument("--start-maximized")
    # NOT headless — must be headed for native UI to respond to UIAutomation clicks
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")

    print("[*] Launching Chrome (headed — required for native UI capture)...")
    driver = webdriver.Chrome(options=opts)
    driver.maximize_window()   # dynamically fills the screen at any resolution

    # Get the Chrome browser PID from Selenium's ChromeDriver process
    # chromedriver PID → its child chrome.exe is the actual browser
    chrome_pid = None
    try:
        import psutil
        cd_pid = driver.service.process.pid
        parent = psutil.Process(cd_pid)
        for child in parent.children(recursive=True):
            if "chrome" in child.name().lower():
                chrome_pid = child.pid
                break
        if chrome_pid:
            print(f"[*] Chrome browser PID: {chrome_pid}")
        else:
            print("[*] Could not determine Chrome PID — will scan all Chrome windows")
    except Exception as e:
        print(f"[*] PID detection: {e}")
    try:
        for url in urls:
            driver.get(url)
            time.sleep(3)
            driver.execute_script("window.focus();")
            time.sleep(0.5)
            results = native_capture(url, chrome_pid=chrome_pid)
            if not results: continue

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            hostname = urlparse(url).hostname or url
            slug     = hostname.replace(".", "_").replace("-", "_")
            for key in ("panel1", "panel2", "panel3"):
                if results.get(key):
                    p = os.path.join(out_dir, f"cert_{key}_{slug}_{ts}.png")
                    with open(p, "wb") as f:
                        f.write(results[key])
    finally:
        driver.quit()

    print(f"\n[+] Done — screenshots saved to: {out_dir}")


if __name__ == "__main__":
    main()
