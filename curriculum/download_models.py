#!/usr/bin/env python3
"""
download_models.py
==================
Downloads GGUF models from HuggingFace into curriculum/models/.
Uses the huggingface_hub library (pip install huggingface_hub).

Run:
  .venv\Scripts\python download_models.py
"""

import pathlib
import sys
import os

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("huggingface_hub not found.")
    print("Run: .venv\\Scripts\\pip install huggingface_hub")
    sys.exit(1)

MODELS_DIR = pathlib.Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

DOWNLOADS = [
    {
        "name"    : "Gemma-2-2B-IT-Q4_K_M",
        "repo_id" : "bartowski/gemma-2-2b-it-GGUF",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "save_as" : "gemma-2-2b-it-q4_k_m.gguf",
        "size_hint": "~1.6 GB",
    },
    {
        "name"    : "Phi-3-Mini-4K-Q4",
        "repo_id" : "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "save_as" : "Phi-3-mini-4k-instruct-q4.gguf",
        "size_hint": "~2.2 GB",
    },
]

def download(model: dict) -> None:
    dest = MODELS_DIR / model["save_as"]
    if dest.exists():
        print(f"  [SKIP] {model['name']} already exists: {dest}")
        return

    print(f"\n  Downloading {model['name']}  ({model['size_hint']}) ...")
    print(f"  From: {model['repo_id']} / {model['filename']}")
    path = hf_hub_download(
        repo_id   = model["repo_id"],
        filename  = model["filename"],
        local_dir = str(MODELS_DIR),
    )
    # Rename to expected filename if needed
    downloaded = pathlib.Path(path)
    if downloaded.name != model["save_as"]:
        downloaded.rename(dest)
    print(f"  Saved: {dest}")

def main():
    print("\n=== Model Downloader ===")
    print(f"Models directory: {MODELS_DIR}\n")

    # Download both models
    for model in DOWNLOADS:
        try:
            download(model)
        except Exception as e:
            print(f"  [ERROR] Failed to download {model['name']}: {e}")
            print("  Check your internet connection and try again.")

    print("\nDone. Now run:")
    print("  .venv\\Scripts\\python benchmark_models.py")
    print()

if __name__ == "__main__":
    main()
