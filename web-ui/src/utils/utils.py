import base64
import re
import os
import time
from pathlib import Path
from typing import Dict, Optional
import requests
import json
import gradio as gr
import uuid
from typing import Dict, Optional, Any


def encode_image(img_path):
    if not img_path:
        return None
    with open(img_path, "rb") as fin:
        image_data = base64.b64encode(fin.read()).decode("utf-8")
    return image_data


def get_latest_files(directory: str, file_types: list = ['.webm', '.zip']) -> Dict[str, Optional[str]]:
    """Get the latest recording and trace files"""
    latest_files: Dict[str, Optional[str]] = {ext: None for ext in file_types}

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return latest_files

    for file_type in file_types:
        try:
            matches = list(Path(directory).rglob(f"*{file_type}"))
            if matches:
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                # Only return files that are complete (not being written)
                if time.time() - latest.stat().st_mtime > 1.0:
                    latest_files[file_type] = str(latest)
        except Exception as e:
            print(f"Error getting latest {file_type} file: {e}")

    return latest_files


def slugify(value: str) -> str:
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = str(value)
    # Remove non-word characters (everything except numbers and letters)
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    # Replace all runs of whitespace with a single dash
    return re.sub(r'[-\s]+', '-', value)


def generate_task_title(llm: Any, task: str) -> str:
    """
    Generate a concise, 3-5 word title for the task using the provided LLM.
    Falls back to slugify if the LLM call fails.
    """
    if not llm or not task:
        return slugify(task[:50])

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        
        system_msg = SystemMessage(content=(
            "You are a helpful assistant that summarizes user prompts into concise, 3-5 word folder names. "
            "Output ONLY the short title (e.g., 'audit_confluence_security', 'login_search_report'). "
            "No punctuation, no quotes, separate words with underscores."
        ))
        human_msg = HumanMessage(content=f"Summarize this task into a 3-5 word slug: {task}")
        
        # Call the LLM (synchronous interface for simple title gen)
        response = llm.invoke([system_msg, human_msg])
        title = response.content.strip().lower()
        
        # Clean up the output to ensure it's a valid slug
        title = re.sub(r'[^a-z0-9_]+', '_', title).strip('_')
        # Limit to 6 words maximum
        words = title.split('_')
        if len(words) > 6:
            title = '_'.join(words[:6])
            
        if not title:
            return slugify(task[:50])
            
        return title
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to generate intelligent title: {e}")
        return slugify(task[:50])
