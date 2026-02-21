
import asyncio
import logging
import os
import gradio as gr
from typing import Dict, Any, List

from src.webui.webui_manager import WebuiManager
from src.agent.multi_process_runner import MultiProcessRunner
from src.browser.custom_browser import CustomBrowser
from src.browser.context import BrowserContextConfig
from src.utils import llm_provider
from browser_use.browser.browser import BrowserConfig

logger = logging.getLogger(__name__)

def create_multi_process_automation_tab(webui_manager: WebuiManager):
    
    with gr.Tab("Multi-Process Automation") as tab:
        gr.Markdown("## Multi-Process Orchestrator\nExecute complex workflows by splitting them into sequential processes.")
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Multi-Process Goal / Prompt",
                    placeholder="Describe your workflow. Use 'Process 1:', 'Process 2:' or '---' to split steps.",
                    lines=10,
                    interactive=True
                )
                
                with gr.Row():
                   run_button = gr.Button("🚀 Run Multi-Process", variant="primary")
                   stop_button = gr.Button("🛑 Stop", variant="stop")

            with gr.Column(scale=1):
                # Simple LLM Config for Multi-Process
                provider_dropdown = gr.Dropdown(
                    choices=[p.value for p in llm_provider.LLMProvider],
                    value="ollama",
                    label="LLM Provider",
                    interactive=True
                )
                model_dropdown = gr.Dropdown(
                    value="qwen2.5:14b",
                    label="Model Name",
                    interactive=True,
                    allow_custom_value=True
                )
                temperature_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature"
                )
                
                gr.Markdown("### Configuration")
                max_steps_input = gr.Number(value=30, label="Max Steps per Process", precision=0)
                use_vision_checkbox = gr.Checkbox(label="Use Vision", value=True)

        with gr.Row():
             status_markdown = gr.Markdown("### Status: Ready")
        
        with gr.Row():
            output_log = gr.Textbox(label="Execution Log", lines=15, interactive=False)
            
        with gr.Row():
             results_markdown = gr.Markdown("### Results will appear here...")

    async def run_multi_process(prompt, provider, model, temp, max_steps, use_vision):
        if not prompt:
            yield "Please enter a prompt.", "❌ No prompt provided."
            return

        try:
            # Initialize LLM
            llm = llm_provider.get_llm_model(
                provider=provider,
                model_name=model,
                temperature=temp,
                base_url=None,
                api_key=None,
                num_ctx=32000 # Default high context for complex tasks
            )
            
            # Initialize Browser
            browser = CustomBrowser(config=BrowserConfig(headless=False))
            # We don't create context here, the runner does it or we pass a shared one?
            # MultiProcessRunner takes browser and browser_context.
            # Let's create a shared context config.
            context_config = BrowserContextConfig(
                viewport_expansion=0,
            )
            context = await browser.new_context(config=context_config)
            
            runner = MultiProcessRunner(
                browser=browser,
                browser_context=context,
                llm=llm,
                use_vision=use_vision,
                max_steps=int(max_steps),
                save_agent_history_path="./tmp/agent_history"
            )
            
            logs = []
            
            async def on_process_start(name, idx, total):
                msg = f"▶️ Starting Process {idx}/{total}: {name}"
                logs.append(msg)
                yield "\n".join(logs), f"Running: {name}..."
            
            async def on_process_complete(result):
                icon = "✅" if result.status == "PASS" else "❌"
                msg = f"{icon} Process Completed: {result.name} ({result.status})"
                logs.append(msg)
                yield "\n".join(logs), f"Completed: {result.name}"

            # We need to hook up logging or callbacks to stream updates
            # For now, we just run it and await result
            
            yield "Starting Multi-Process Runner...", "Initializing..."
            
            results, duration = await runner.run(
                prompt, 
                on_process_start=on_process_start,
                on_process_complete=on_process_complete
            )
            
            summary = runner.get_summary_markdown(duration)
            yield "\n".join(logs) + "\n\nDONE.", summary
            
            await context.close()
            await browser.close()
            
        except Exception as e:
            logger.error(f"Multi-process run failed: {e}", exc_info=True)
            yield f"Error: {str(e)}", f"❌ Error: {str(e)}"

    run_button.click(
        fn=run_multi_process,
        inputs=[prompt_input, provider_dropdown, model_dropdown, temperature_slider, max_steps_input, use_vision_checkbox],
        outputs=[output_log, results_markdown]
    )
    
    return tab
