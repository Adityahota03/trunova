#!/usr/bin/env python3
"""
download_models.py - Downloads GGUF models with retry, resume, and fallback support.
Usage:
  python download_models.py
"""

import os
import sys
import time
import pathlib
import urllib.request

MODELS_DIR = pathlib.Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Set longer timeout for large file downloads
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

DOWNLOADS = [
    {
        "name"     : "Gemma-2-2B-IT-Q4_K_M  (~1.6 GB)",
        "repo_id"  : "bartowski/gemma-2-2b-it-GGUF",
        "filename" : "gemma-2-2b-it-Q4_K_M.gguf",
        "save_as"  : "gemma-2-2b-it-q4_k_m.gguf",
        "url"      : "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
    },
    {
        "name"     : "Phi-3-Mini-4K-Q4  (~2.2 GB)",
        "repo_id"  : "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename" : "Phi-3-mini-4k-instruct-q4.gguf",
        "save_as"  : "Phi-3-mini-4k-instruct-q4.gguf",
        "url"      : "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
    },
]

MAX_RETRIES = 5


def download_via_hf(model: dict, dest: pathlib.Path) -> bool:
    """Download using huggingface_hub if available."""
    try:
        from huggingface_hub import hf_hub_download
        print(f"  Fetching via huggingface_hub ...")
        path = hf_hub_download(
            repo_id   = model["repo_id"],
            filename  = model["filename"],
            local_dir = str(MODELS_DIR),
        )
        downloaded = pathlib.Path(path)
        if downloaded.resolve() != dest.resolve():
            os.replace(downloaded, dest)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  [INFO] huggingface_hub notice: {e}")
        return False


def download_via_urllib(model: dict, dest: pathlib.Path) -> bool:
    """Direct HTTP stream download with progress output and resume capability."""
    url = model["url"]
    temp_dest = dest.with_suffix(".part")
    initial_size = temp_dest.stat().st_size if temp_dest.exists() else 0

    headers = {"User-Agent": "Trunova-Downloader/1.0"}
    if initial_size > 0:
        headers["Range"] = f"bytes={initial_size}-"

    req = urllib.request.Request(url, headers=headers)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Connecting to download mirror (attempt {attempt}/{MAX_RETRIES}) ...")
            with urllib.request.urlopen(req, timeout=300) as response:
                content_len = response.headers.get("Content-Length")
                total_size = int(content_len) + initial_size if content_len else None

                mode = "ab" if initial_size > 0 else "wb"
                downloaded = initial_size
                chunk_size = 1024 * 1024  # 1 MB chunks

                with open(temp_dest, mode) as out_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            pct = (downloaded / total_size) * 100
                            print(f"\r  Progress: {downloaded // 1_000_000} MB / {total_size // 1_000_000} MB ({pct:.1f}%)", end="", flush=True)
                        else:
                            print(f"\r  Downloaded: {downloaded // 1_000_000} MB", end="", flush=True)

                print()
                os.replace(temp_dest, dest)
                return True
        except Exception as e:
            print(f"\n  [WARN] Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                wait = 5 * attempt
                print(f"  Retrying in {wait}s ...")
                time.sleep(wait)

    return False


def download_model(model: dict) -> bool:
    dest = MODELS_DIR / model["save_as"]
    alt_dest = MODELS_DIR / model["filename"]

    if dest.exists() and dest.stat().st_size > 100_000_000:
        print(f"  [SKIP] {model['name']} already present ({dest.stat().st_size // 1_000_000} MB)")
        return True
    if alt_dest.exists() and alt_dest.stat().st_size > 100_000_000:
        print(f"  [SKIP] {model['name']} already present ({alt_dest.stat().st_size // 1_000_000} MB)")
        return True

    print(f"\n  Downloading {model['name']} ...")
    print(f"  Target: {dest}")

    ok = download_via_hf(model, dest)
    if not ok:
        ok = download_via_urllib(model, dest)

    if ok and (dest.exists() or alt_dest.exists()):
        final = dest if dest.exists() else alt_dest
        size_mb = final.stat().st_size // 1_000_000
        print(f"  Saved: {final.name} ({size_mb} MB)")
        return True

    print(f"  [ERROR] All attempts failed for {model['name']}")
    print("  You can also download manually in your browser and save to curriculum/models/:")
    print(f"    {model['url']}")
    return False


def main():
    print("\n=== GGUF Model Downloader ===")
    print(f"Models directory: {MODELS_DIR}")
    print("Large files (~1.6 - 2.2 GB) may take several minutes depending on network speed.\n")

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
        print(f"\nAt least one model is available in {MODELS_DIR}.")
        print("You can run offline RAG and agent tests:")
        print("  python test_rag.py \"What is photosynthesis?\"")
        print("  python agent.py \"What is Newton's first law?\" --class 9 --subject science")
    else:
        print("\nNo models downloaded. Please download manually from the links above.")


if __name__ == "__main__":
    main()
