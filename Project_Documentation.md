# Dynamic Scrapper: Project Documentation

This document provides a comprehensive overview of the **Dynamic Scrapper / AI Browser Agent** project. It outlines how the project works, the architecture, and detailed explanations of every core file related to the system's execution (ignoring tmp, examples, and generated reports).

---

## 1. Project Overview & How It Works

This project is an advanced, AI-driven Browser Automation Agent built on top of a framework (such as `browser-use` and `playwright`). It acts as a bridge between Large Language Models (LLMs) and standard web interfaces.

### Core Mechanisms:
1. **Web UI Interface:** The agent exposes a server UI (via `webui.py`), allowing users to input goals or load tasks.
2. **Custom Controller Actions:** Once an LLM decides on a required action (e.g., "Click the login button"), the system passes this instruction to `custom_controller.py`.
3. **Smart DOM Handling:** Instead of blindly interacting with the page, the agent takes a snapshot of the DOM **before** the action executes. It evaluates element visibility, bounding boxes, and actual attributes to ensure the action is physically possible (preventing standard scraper errors).
4. **Smart Operations:** Operations like `input_text` dynamically search for credential inputs to trigger a `smart_login` fallback natively, preventing the AI from getting confused if a login prompt suddenly appears.
5. **Enrichment:** Once the action executes, the system logs the element data. This makes the session replayable and highly deterministic.

---

## 2. Main Logic & Executable Files (`web-ui/`)

The `web-ui/` directory houses the actual AI application layer and execution engine. 

### Server Entry Files
#### `web-ui/webui.py`
This is the **Entry Point** of the application. 
- **Purpose:** It initializes the web-facing GUI (using `gradio` or similar components). 
- **Features:** It sets up CLI arguments (allowing you to bind to specific IPs/Ports and set themes). Crucially, it defines a rigid `RedactingFilter` for Python's logging system. Any passwords, API keys, or inputs passed to the browser are immediately replaced with `******` before writing to the console or logs, guaranteeing operational security.

### Core Agent Operations (`web-ui/src/controller/`)
#### `web-ui/src/controller/custom_controller.py`
This is arguably the most important file in the project. It overrides standard browser automation to create an "AI-proof" execution layer.
- **`smart_login`:** Checks if an LLM is trying to type into a login form. If it detects username/password parameters, it hijacks the process and intelligently attempts to map those to the correct inputs, clicking the login button automatically.
- **`upload_file`:** Intercepts LLM attempts to "type" into file inputs and properly triggers Playwright's specific file-upload/file-chooser mechanics. It even supports directory path expansion.
- **`retrieve_value_by_element` & `validate_value`:** Allows the LLM to safely read data from the DOM, parsing complex SVGs or checking status colors (like detecting "✓ RUNNING" or "✗ ERROR" based on CSS classes).
- **`click_element`:** Adds smart navigation. If it detects the LLM is trying to click a menu trigger (like a dropdown), it intelligently looks for the actual link inside the menu and clicks that instead.

### Utilities & Core Architecture (`web-ui/src/utils/`)
#### `web-ui/src/utils/dom_snapshot.py`
- **Purpose:** Responsible for creating the `pre_action` and `post_action` mappings. It captures a JSON representation of the website’s current loaded DOM so the AI understands exactly where elements sit on the screen.

#### `web-ui/src/utils/comprehensive_element_capture.py`
- **Purpose:** Attached to the custom controller, this script parses the precise technical specs of an element (bounding boxes, CSS classes) before an action is performed, acting as a fallback for the LLM if an index shifts.

#### `web-ui/src/utils/enrich_agent_history.py`
- **Purpose:** Post-processing script. After an agent finishes its job, this file marries the raw textual instructions from the LLM with the underlying DOM structures. This generates a "Contextual Log" to tell developers *precisely* what the LLM clicked and why.

### Agent Environment Management (`web-ui/src/agent/`)
#### `web-ui/src/agent/multi_process_runner.py`
- **Purpose:** Manages parallel execution. This file sets up pools and workers to ensure that multiple browser tasks or multiple agents can run simultaneously without blocking each other or crashing the main server.

### Front-End Infrastructure (`web-ui/src/webui/`)
#### `web-ui/src/webui/interface.py` & `webui/webui_manager.py`
- **Purpose:** They define the actual frontend user experience of the tool (buttons, forms, themes). `interface.py` manages the layout logic, while `webui_manager.py` bridges the gap between those buttons and the core `Agent` Python classes.

---

## 5. Summary of Design Considerations

When building this project, several key things were considered regarding standard AI unreliability:

1. **AI "Hallucinations":** The LLM will often try to "type" an actual file path into a file-upload box, which crashes traditional Selenium scripts. This tool specifically intercepts that behavior and automatically executes a File Uploader context.
2. **Shifting DOM Indexes:** Websites load dynamically. The `comprehensive_element_capture` ensures that if an element index changes during a slow load, the agent tracks its internal DOM tree location.
3. **Security:** By default, LLMs log out everything they see. The `webui.py` custom logging filter ensures that vault credentials dynamically injected into the workspace are completely redacted before ever hitting the console.
4. **Smart Recoveries:** Built-in macros (like `ask_for_assistant` or `smart_login`) guarantee that if the AI hits a CAPTCHA or a sudden credential expiration, it either asks a human seamlessly or re-authenticates automatically without breaking the pipeline.
