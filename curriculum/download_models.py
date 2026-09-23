#!/usr/bin/env python3
"""
download_models.py - Downloads GGUF models with retry and resume support.
Run: .venv\Scripts\python download_models.py
"""

import pathlib
import sys
import os
import time

MODELS_DIR = pathlib.Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Set longer timeout for large file downloads
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Run: .venv\\Scripts\\pip install huggingface_hub")
    sys.exit(1)

# Gemma first (smaller, more stable CDN), then Phi-3
DOWNLOADS = [
    {
        "name"     : "Gemma-2-2B-IT-Q4_K_M  (~1.6 GB)",
        "repo_id"  : "bartowski/gemma-2-2b-it-GGUF",
        "filename" : "gemma-2-2b-it-Q4_K_M.gguf",
        "save_as"  : "gemma-2-2b-it-q4_k_m.gguf",
    },
    {
        "name"     : "Phi-3-Mini-4K-Q4  (~2.2 GB)",
        "repo_id"  : "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename" : "Phi-3-mini-4k-instruct-q4.gguf",
        "save_as"  : "Phi-3-mini-4k-instruct-q4.gguf",
    },
]

MAX_RETRIES = 5

def download_model(model: dict) -> bool:
    dest = MODELS_DIR / model["save_as"]
    if dest.exists() and dest.stat().st_size > 100_000_000:
        print(f"  [SKIP] {model['name']} already present ({dest.stat().st_size // 1_000_000} MB)")
        return True

    print(f"\n  Downloading {model['name']} ...")
    print(f"  Repo: {model['repo_id']}")
    print(f"  File: {model['filename']}")
    print(f"  Dest: {dest}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Attempt {attempt}/{MAX_RETRIES} ...")
            path = hf_hub_download(
                repo_id   = model["repo_id"],
                filename  = model["filename"],
                local_dir = str(MODELS_DIR),
                resume_download = True,
            )
            downloaded = pathlib.Path(path)
            if downloaded.resolve() != dest.resolve():
                downloaded.rename(dest)
            size_mb = dest.stat().st_size // 1_000_000
            print(f"  Saved: {dest}  ({size_mb} MB)")
            return True
        except Exception as e:
            print(f"  [WARN] Attempt {attempt} failed: {type(e).__name__}: {str(e)[:120]}")
            if attempt < MAX_RETRIES:
                wait = 10 * attempt
                print(f"  Retrying in {wait}s ...")
                time.sleep(wait)

    print(f"  [ERROR] All {MAX_RETRIES} attempts failed for {model['name']}")
    print("  Try downloading manually in your browser:")
    if "gemma" in model["save_as"]:
        print("  https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf")
    else:
        print("  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf")
    print(f"  Save to: {MODELS_DIR}")
    return False

def main():
    print("\n=== GGUF Model Downloader ===")
    print(f"Models directory: {MODELS_DIR}")
    print("Timeout: 300s per request. Large files may take 10-30 minutes.\n")

    results = []
    for model in DOWNLOADS:
        ok = download_model(model)
        results.append((model["name"], ok))

    print("\n=== Summary ===")
    for name, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {name}")

    ok_count = sum(1 for _, ok in results if ok)
    if ok_count >= 1:
        print(f"\nAt least one model available. Run:")
        print("  .venv\\Scripts\\python benchmark_models.py")
    else:
        print("\nNo models downloaded. Please download manually (links above).")

if __name__ == "__main__":
    main()
