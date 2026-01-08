import requests
import os
from typing import List, Dict, Any


def get_ollama_models() -> List[str]:
    """Fetch installed models from Ollama API."""
    try:
        ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        response = requests.get(f"{ollama_endpoint}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            # Sort models for consistent ordering
            models.sort()
            return models
        else:
            print(f"Failed to fetch Ollama models: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")

    # Fallback to predefined models if Ollama is not available
    return [
        "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5-coder:14b", "qwen2.5-coder:32b",
        "qwen3-vl:30b", "llama2:7b", "deepseek-r1:14b", "deepseek-r1:32b"
    ]


def get_model_names() -> Dict[str, List[str]]:
    """Get model names for all providers, with dynamic Ollama models."""
    return {
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
        "openai": [
            "gpt-4o", "gpt-4", "gpt-3.5-turbo", "o3-mini",
            "google/gemini-2.0-flash-exp:free",
            "qwen/qwen3-4b:free",
            "moonshotai/kimi-k2",
            "venice/uncensored:free",
            "google/gemma-3-12b:free",
            "mistralai/mistral-7b-instruct:free",
            "qwen/qwen2.5-vl-7b-instruct:free",
            "nvidia/nemotron-3-nano-30b-a3b:free"
        ],
        "deepseek": ["deepseek-chat", "deepseek-reasoner"],
        "google": ["gemini-2.5-flash-lite", "gemini-2.5-flash-tts", "gemini-2.5-flash", "gemini-3-flash",
                   "gemini-robotics-er-1.5-preview", "gemma-3-12b", "gemma-3-1b", "gemma-3-27b",
                   "gemma-3-2b", "gemma-3-4b", "gemini-2.5-flash-native-audio-dialog"],
        "ollama": get_ollama_models(),
        "azure_openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
        "mistral": ["pixtral-large-latest", "mistral-large-latest", "mistral-small-latest", "ministral-8b-latest"],
        "alibaba": ["qwen-plus", "qwen-max", "qwen-vl-max", "qwen-vl-plus", "qwen-turbo", "qwen-long"],
        "moonshot": ["moonshot-v1-32k-vision-preview", "moonshot-v1-8k-vision-preview"],
        "unbound": ["gemini-2.0-flash", "gpt-4o-mini", "gpt-4o", "gpt-4.5-preview"],
        "grok": [
            "grok-3",
            "grok-3-fast",
            "grok-3-mini",
            "grok-3-mini-fast",
            "grok-2-vision",
            "grok-2-image",
            "grok-2",
        ],
        "siliconflow": [
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-V2.5",
            "deepseek-ai/deepseek-vl2",
            "Qwen/Qwen2.5-72B-Instruct-128K",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen2-1.5B-Instruct",
            "Qwen/QwQ-32B-Preview",
            "Qwen/Qwen2-VL-72B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "TeleAI/TeleChat2",
            "THUDM/glm-4-9b-chat",
            "Vendor-A/Qwen/Qwen2.5-72B-Instruct",
            "internlm/internlm2_5-7b-chat",
            "internlm/internlm2_5-20b-chat",
            "Pro/Qwen/Qwen2.5-7B-Instruct",
            "Pro/Qwen/Qwen2-7B-Instruct",
            "Pro/Qwen/Qwen2-1.5B-Instruct",
            "Pro/THUDM/chatglm3-6b",
            "Pro/THUDM/glm-4-9b-chat",
        ],
        "ibm": ["ibm/granite-vision-3.1-2b-preview", "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
                "meta-llama/llama-3-2-90b-vision-instruct"],
        "modelscope":[
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/QwQ-32B-Preview",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-V3",
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-14B",
            "Qwen/Qwen3-30B-A3B",
            "Qwen/Qwen3-32B",
            "Qwen/Qwen3-235B-A22B",
        ],
    }


PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "azure_openai": "Azure OpenAI",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google",
    "alibaba": "Alibaba",
    "moonshot": "MoonShot",
    "unbound": "Unbound AI",
    "ibm": "IBM",
    "grok": "Grok",
}

# Predefined model names for common providers
model_names = get_model_names()
