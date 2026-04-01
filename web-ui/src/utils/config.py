import os

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2.5:14b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
USE_VISION = os.getenv("USE_VISION", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Script Generator – dedicated model (separate from the agent task model)
# Use a small, fast coder model here so code generation doesn't block the UI.
# The agent task continues to use LLM_MODEL_NAME / LLM_PROVIDER.
# ---------------------------------------------------------------------------
SCRIPT_GEN_MODEL    = os.getenv("SCRIPT_GEN_MODEL",    "qwen2.5-coder:7b")
SCRIPT_GEN_PROVIDER = os.getenv("SCRIPT_GEN_PROVIDER", os.getenv("LLM_PROVIDER", "ollama"))

# Script Generator – LLM settings
# ---------------------------------------------------------------------------
# Temperature used when generating Playwright scripts (lower = more deterministic)
SCRIPT_GEN_TEMPERATURE = float(os.getenv("SCRIPT_GEN_TEMPERATURE", "0.1"))
# Context window size (tokens) passed to the generation model.
# 32768 accommodates the system prompt (~1500 tok) + large agent histories + headroom.
SCRIPT_GEN_NUM_CTX = int(os.getenv("SCRIPT_GEN_NUM_CTX", "32768"))
# num_predict token limit for generation (-1 = unlimited).
# 2000 was far too small — a full script with reporting needs 3000–6000 tokens.
SCRIPT_GEN_NUM_PREDICT = int(os.getenv("SCRIPT_GEN_NUM_PREDICT", "-1"))

# ---------------------------------------------------------------------------
# Script Generator – OOM fallback chain
# Comma-separated list of models to try in order when CUDA OOM is detected.
# The primary model (LLM_MODEL_NAME) is automatically prepended at runtime.
# ---------------------------------------------------------------------------
_oom_models_raw = os.getenv("OOM_FALLBACK_MODELS", "qwen2.5:7b,llama3.1:8b")
OOM_FALLBACK_MODELS: list[str] = [m.strip() for m in _oom_models_raw.split(",") if m.strip()]

_oom_ctx_raw = os.getenv("OOM_FALLBACK_CTX", "16000,8000")
OOM_FALLBACK_CTX: list[int] = [int(c.strip()) for c in _oom_ctx_raw.split(",") if c.strip()]

# num_predict used on OOM retry attempts (smaller to fit reduced ctx)
OOM_RETRY_NUM_PREDICT = int(os.getenv("OOM_RETRY_NUM_PREDICT", "2048"))

# ---------------------------------------------------------------------------
# Vault / credential settings
# ---------------------------------------------------------------------------
# Vault key prefix used by the target application's credentials.
# E.g. "PREFIX" means vault.get_credentials("PREFIX") -> {username, password}.
VAULT_CREDENTIAL_PREFIX = os.getenv("VAULT_CREDENTIAL_PREFIX", "APP")

# ---------------------------------------------------------------------------
# Login selectors (used by maybe_login helper injected into generated scripts)
# Comma-separated CSS selector lists
# ---------------------------------------------------------------------------
def _sel_list(env_var: str, default: str) -> list[str]:
    raw = os.getenv(env_var, default)
    return [s.strip() for s in raw.split(",") if s.strip()]

LOGIN_USER_SELECTORS: list[str] = _sel_list(
    "LOGIN_USER_SELECTORS",
    "input[name='username'],input#username,input[name='user'],input[type='text'],#login_username",
)
LOGIN_PASS_SELECTORS: list[str] = _sel_list(
    "LOGIN_PASS_SELECTORS",
    "input[name='password'],input#password,input[type='password'],#login_password",
)
LOGIN_SUBMIT_SELECTORS: list[str] = _sel_list(
    "LOGIN_SUBMIT_SELECTORS",
    "button[type='submit'],#login_button,button:has-text('Sign in'),button:has-text('Sign In'),input[type='submit']",
)

# ---------------------------------------------------------------------------
# Timeout settings (milliseconds) used inside generated Playwright scripts
# ---------------------------------------------------------------------------
TIMEOUT_FIND_IN_FRAMES_MS   = int(os.getenv("TIMEOUT_FIND_IN_FRAMES_MS",   "5000"))
TIMEOUT_ELEMENT_VISIBLE_MS  = int(os.getenv("TIMEOUT_ELEMENT_VISIBLE_MS",  "1000"))
TIMEOUT_FILL_MS             = int(os.getenv("TIMEOUT_FILL_MS",             "3000"))
TIMEOUT_CLICK_MS            = int(os.getenv("TIMEOUT_CLICK_MS",            "3000"))
TIMEOUT_FILE_CHOOSER_MS     = int(os.getenv("TIMEOUT_FILE_CHOOSER_MS",     "7000"))
TIMEOUT_UPLOAD_CLICK_MS     = int(os.getenv("TIMEOUT_UPLOAD_CLICK_MS",     "5000"))
TIMEOUT_UPLOAD_FALLBACK_MS  = int(os.getenv("TIMEOUT_UPLOAD_FALLBACK_MS",  "5000"))
TIMEOUT_TABLE_WAIT_MS       = int(os.getenv("TIMEOUT_TABLE_WAIT_MS",       "10000"))

# ---------------------------------------------------------------------------
# Ollama / Verifier settings
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL             = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_VERIFIER_MODEL       = os.getenv("OLLAMA_VERIFIER_MODEL", "qwen2.5:14b")
OLLAMA_VERIFIER_TEMPERATURE = float(os.getenv("OLLAMA_VERIFIER_TEMPERATURE", "0.1"))
OLLAMA_VERIFIER_TOP_P       = float(os.getenv("OLLAMA_VERIFIER_TOP_P", "0.95"))
OLLAMA_VERIFIER_TIMEOUT_S   = int(os.getenv("OLLAMA_VERIFIER_TIMEOUT_S", "180"))
OLLAMA_VERIFIER_MIN_CONF    = int(os.getenv("OLLAMA_VERIFIER_MIN_CONF", "90"))
OLLAMA_VERIFIER_STAGE_THRESH= int(os.getenv("OLLAMA_VERIFIER_STAGE_THRESH", "75"))
OLLAMA_AVAILABILITY_TIMEOUT_S = int(os.getenv("OLLAMA_AVAILABILITY_TIMEOUT_S", "5"))

# ---------------------------------------------------------------------------
# Agent run timeouts (seconds)
# ---------------------------------------------------------------------------
# Maximum wall-clock time for a single agent task run
AGENT_RUN_TIMEOUT_S      = float(os.getenv("AGENT_RUN_TIMEOUT_S",   "600"))
# Maximum time to wait for the browser-use response event (WebUI)
WEBUI_RESPONSE_TIMEOUT_S = float(os.getenv("WEBUI_RESPONSE_TIMEOUT_S", "3600"))
# Short timeout for graceful task cancellation
TASK_CANCEL_TIMEOUT_S    = float(os.getenv("TASK_CANCEL_TIMEOUT_S",  "2"))

# ---------------------------------------------------------------------------
# Agent Configuration
# ---------------------------------------------------------------------------
MAX_STEPS = int(os.getenv("MAX_STEPS", "25"))
MAX_ACTIONS = int(os.getenv("MAX_ACTIONS", "3"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "128000"))
TOOL_CALLING_METHOD = os.getenv("TOOL_CALLING_METHOD", None)

# ---------------------------------------------------------------------------
# Browser Configuration
# ---------------------------------------------------------------------------
BROWSER_HEADLESS = os.getenv("HEADLESS", "false").lower() == "true"
BROWSER_WIDTH = int(os.getenv("WINDOW_WIDTH", "1280"))
BROWSER_HEIGHT = int(os.getenv("WINDOW_HEIGHT", "1100"))
KEEP_BROWSER_OPEN = os.getenv("KEEP_BROWSER_OPEN", "true").lower() == "true"
USE_OWN_BROWSER = os.getenv("USE_OWN_BROWSER", "true").lower() == "true"
BROWSER_BINARY_PATH = os.getenv("BROWSER_PATH")
BROWSER_USER_DATA_DIR = os.getenv("BROWSER_USER_DATA")

# ---------------------------------------------------------------------------
# Storage Paths
# ---------------------------------------------------------------------------
SAVE_AGENT_HISTORY_PATH = os.getenv("AGENT_HISTORY_PATH", "./tmp/agent_history")
SAVE_TRACE_PATH = os.getenv("TRACE_PATH")
SAVE_RECORDING_PATH = os.getenv("RECORDING_PATH")
SAVE_DOWNLOAD_PATH = os.getenv("DOWNLOADS_PATH", "./tmp/downloads")