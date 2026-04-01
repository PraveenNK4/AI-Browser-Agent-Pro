# Web-UI File Architecture Documentation

This document explains every relevant core code file found inside the `web-ui/` directory. (It excludes `.png`, `.docx`, `.txt` outputs, and `tmp/` contents to focus solely on the source code and logic).

---

## 1. Root Configuration & Entry Files (`web-ui/`)
- **`webui.py`**: The primary entry point for launching the main Web Interface. It utilizes Gradio to render the interface, processes command-line arguments (like Port, IP, and Theme), and most importantly, initializes a global `RedactingFilter` that obscures vault secrets from console logs.
- **`.env` and `.gitignore`**: Standard environment configuration settings (for API keys, local URLs) and Git tracking instructions.
- **`requirements.txt`**: Python dependencies required to run the web-ui.
- **Root Utility/Debug Scripts (`test_*.py`, `debug_*.py`, `vault_test.py`, `setup_bitnet.py`, `regenerate_script.py`)**: A large collection of root-level standalone scripts used for isolated testing. For example, `regenerate_script.py` attempts to rebuild a broken Python automation file, while `test_ollama_max_accuracy.py` tests the accuracy of local LLMs. 

---

## 2. Agent Logic (`web-ui/src/agent/`)
- **`agent_memory.py`**: Manages the contextual state and memory window for the AI. It ensures that the LLM remembers previous steps taken without hallucinating older page states.
- **`multi_process_runner.py`**: A robust processing layer designed to execute multiple independent automation agents across different tabs/processes simultaneously without blocking the main web UI server thread.

---

## 3. Browser Initialization (`web-ui/src/browser/`)
- **`custom_browser.py`**: Intercepts the default Playwright browser initialization to inject custom Chromium arguments, anti-bot evasions, and headless config optimizations natively required across secure environments (like Citrix).
- **`custom_context.py`**: Controls specific browser contexts (session data, cookies, local storage maps) allowing the agent to persist logins across sequential tasks.

---

## 4. The Action Hub (`web-ui/src/controller/`)
- **`custom_controller.py`**: Extremely critical file. It extends standard browser automation tools specifically tuned for AI execution to drastically reduce hallucination. Notable features:
  - `smart_login`: Automatically hijacks tasks where the AI tries to guess login fields, securely auto-populating credentials via logic.
  - `upload_file`: Replaces manual typing with actual OS-level File Chooser triggers.
  - `validate_value` & `retrieve_value_by_element`: Allows the LLM to inspect visual DOM badges, colors, and complex SVG statuses correctly instead of purely relying on textual scraping.

---

## 5. Workflow Orchestration (`web-ui/src/orchestrator/`)
*This module handles complex, multi-stage task lists rather than single conversational instructions.*
- **`process_orchestrator.py`**: The main brain for handling master "playbooks". If an agent needs to perform 10 distinct, disparate steps across 3 websites, this orchestrator manages that lifecycle.
- **`report_generator.py`**: Compiles the successes, failures, and screenshots tracked during an orchestration run into structured output files.
- **`process_models.py` & `process_parser.py`**: Utilizes Pydantic schemas to validate and decompose user intent into machine-readable JSON workflows.
- **`cli.py`**: Exposes the Orchestrator engine to command-line prompts (so it can be run headlessly in CI/CD without the Web UI).

---

## 6. Utilities & Translations (`web-ui/src/utils/`)
*This directory handles DOM interactions, model networking, and log translations.*
- **`dom_snapshot.py`**: Triggers just milliseconds before any element interaction. It logs exactly what the visual hierarchy looked like (preventing the LLM from clicking something obstructed by a sudden pop-up).
- **`comprehensive_element_capture.py`**: Extracts high-fidelity metadata from targeted elements (color maps, true bounding boxes, class names) to provide backup finding instructions to auto-generated fallback scripts.
- **`enrich_agent_history.py` & `strip_and_enrich_history.py`**: Analyzes generated agent execution logs and merges them with raw element data. This translates vague "I clicked index 5" into exact representations like "I clicked `#submit_btn`".
- **`llm_script_generator.py` & `generate_playwright_script.py`**: Directly converts agent execution traces into raw, standalone Playwright/Python code for use outside of the AI wrapper.
- **`llm_dom_verifier.py` & `ollama_max_accuracy_verifier.py`**: Uses LLMs dynamically to check if a specific action *actually* completed successfully (like reading a success banner to determine if saving worked).
- **`llm_provider.py` & `mcp_client.py`**: Configures API connectors for external endpoint services (OpenAI, Anthropic Claude, specific local Ollama nodes, and Model Context Protocol integrations).
- **`table_extraction_compiler.py`**: Built specifically and explicitly to target HTML `<table>` structures. Since tabular data is hard for regular LLMs to read, this extracts it flawlessly into JSON/DataFrames.
- **`vault.py` & `config.py`**: Safely loads `.env` states and integrates secure vault calls directly into utility execution.

---

## 7. User Interface Engine (`web-ui/src/webui/`)
- **`interface.py`**: Defines the main window styling and layout logic.
- **`webui_manager.py`**: Handles states, listeners, and linking Gradio UI buttons to background Python agents.
- **`components/browser_use_agent_tab.py`**: The individual GUI module handling standard "conversational" AI agent configurations.
- **`components/multi_process_automation_tab.py`**: The separate GUI tab managing bulk, parallel tasks and multi-agent coordination metrics.

---

## 8. Secure Backend Processing (`web-ui/backend/`)
- **`credential_vault.py`**: System logic to establish Windows DPAPI-encrypted vaults or local keyed `.vault` files. It converts raw passwords into `.vault_key` managed objects so they are strictly never read out loud or rendered physically in memory by accident.

---

## 9. Developer Scripts (`web-ui/scripts/`)
- Contains over a dozen isolated diagnostic programs (`check_element.py`, `find_elements.py`, `validate_dom_capture.py`, `verify_enrichment.py`). These scripts allow developers to rapidly test DOM extractions or verify element interactions without mounting the entire LLM + Gradio server framework.

---

## 10. Usage Demonstrations (`web-ui/examples/`)
- **`comprehensive_element_capture_examples.py` & `dom_capture_examples.py`**: Educational scripts that act as reference manuals on how to import and safely engage the `src/utils` libraries programmatically if one were to build entirely new toolings independent from `webui.py`.
