#!/usr/bin/env python3
import subprocess
import sys
import os

def run_command(command):
    print(f"[*] Running: {command}")
    try:
        # Use utf-8 encoding to avoid 'charmap' errors on Windows
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()
        return process.returncode == 0
    except Exception as e:
        print(f"[-] Error: {e}")
        return False

def setup_bitnet():
    print("="*60)
    print("🚀 DeepSeek-R1 / BitNet Setup Utility")
    print("="*60)

    # 1. Check if Ollama is installed
    if not run_command("ollama --version"):
        print("[-] Ollama not found. Please install it from https://ollama.com")
        sys.exit(1)

    # 2. Pull High-Tier Reasoning Model
    # User requested DeepSeek-R1 (Distill-Llama-70B)
    target_model = "deepseek-r1:70b"
    alias_name = "bitnet-b1.58-7b" # Keep alias for agent compatibility
    
    print(f"[*] Checking for {target_model}...")
    
    if run_command(f'ollama list | findstr "{target_model}"'):
        print(f"[+] {target_model} already exists.")
    else:
        print(f"[*] Pulling high-tier reasoning model: {target_model}")
        print("[!] WARNING: This is a 40GB+ download and requires significant VRAM.")
        if run_command(f"ollama pull {target_model}"):
            print(f"[*] Creating compatibility alias: {alias_name} from {target_model}")
            run_command(f"ollama cp {target_model} {alias_name}")
        else:
            print(f"[-] Critical error: Could not pull {target_model}.")

    print("\n[+] Setup Phase Complete!")
    print(f"Using: {target_model} (Aliased as {alias_name})")
    print("="*60)

if __name__ == "__main__":
    setup_bitnet()
