import os

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2.5:14b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
USE_VISION = os.getenv("USE_VISION", "false").lower() == "true"

# Agent Configuration
MAX_STEPS = int(os.getenv("MAX_STEPS", "25"))
MAX_ACTIONS = int(os.getenv("MAX_ACTIONS", "3"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "128000"))
TOOL_CALLING_METHOD = os.getenv("TOOL_CALLING_METHOD", None)

# Browser Configuration
BROWSER_HEADLESS = os.getenv("HEADLESS", "false").lower() == "true"
BROWSER_WIDTH = int(os.getenv("WINDOW_WIDTH", "1280"))
BROWSER_HEIGHT = int(os.getenv("WINDOW_HEIGHT", "1100"))
KEEP_BROWSER_OPEN = os.getenv("KEEP_BROWSER_OPEN", "true").lower() == "true"
USE_OWN_BROWSER = os.getenv("USE_OWN_BROWSER", "true").lower() == "true"
BROWSER_BINARY_PATH = os.getenv("BROWSER_PATH")
BROWSER_USER_DATA_DIR = os.getenv("BROWSER_USER_DATA")

# Storage Paths
SAVE_AGENT_HISTORY_PATH = os.getenv("AGENT_HISTORY_PATH", "./tmp/agent_history")
SAVE_TRACE_PATH = os.getenv("TRACE_PATH")
SAVE_RECORDING_PATH = os.getenv("RECORDING_PATH")
SAVE_DOWNLOAD_PATH = os.getenv("DOWNLOADS_PATH", "./tmp/downloads")