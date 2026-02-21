import gradio as gr
import asyncio

# Monkey patch to fix pending_message_lock issue
import gradio.queueing

original_init = gradio.queueing.Queue.__init__

def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if not hasattr(self, 'pending_message_lock') or self.pending_message_lock is None:
        self.pending_message_lock = asyncio.Lock()
    if not hasattr(self, 'delete_lock') or self.delete_lock is None:
        self.delete_lock = asyncio.Lock()

gradio.queueing.Queue.__init__ = patched_init

from src.webui.webui_manager import WebuiManager
from src.webui.components.browser_use_agent_tab import create_browser_use_agent_tab
# from src.webui.components.deep_research_agent_tab import create_deep_research_agent_tab

theme_map = {
    "Default": gr.themes.Default(),
    "Soft": gr.themes.Soft(),
    "Monochrome": gr.themes.Monochrome(),
    "Glass": gr.themes.Glass(),
    "Origin": gr.themes.Origin(),
    "Citrus": gr.themes.Citrus(),
    "Ocean": gr.themes.Ocean(),
    "Base": gr.themes.Base()
}


def create_ui(theme_name="Ocean"):
    css = """
    .gradio-container {
        width: 70vw !important; 
        max-width: 70% !important; 
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 10px !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 20px;
    }
    .tab-header-text {
        text-align: center;
    }
    .theme-section {
        margin-bottom: 10px;
        padding: 15px;
        border-radius: 10px;
    }
    """

    # dark mode in default
    js_func = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """

    ui_manager = WebuiManager()

    with gr.Blocks(
            title="AI Browser Agent Pro", theme=theme_map[theme_name], css=css, js=js_func,
    ) as demo:
        # Initialize queue with explicit settings
        gr.set_static_paths(paths=[])
        with gr.Row():
            gr.Markdown(
                """
                # AI Browser Agent Pro
                """,
                elem_classes=["header-text"],
            )

        with gr.Tabs() as tabs:
            with gr.TabItem("Run Agent"):
                create_browser_use_agent_tab(ui_manager)

    return demo
