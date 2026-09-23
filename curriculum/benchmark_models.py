#!/usr/bin/env python3
"""
curriculum/benchmark_models.py
================================
Stage A: Desktop LLM benchmarking.

Compares two GGUF models on:
  - RAM usage (RSS before and after model load)
  - Model load time
  - Tokens per second (inference speed)
  - Answer quality on 5 fixed curriculum questions

Usage:
  python benchmark_models.py

Models must be downloaded and placed in:
  curriculum/models/gemma-2-2b-it-q4_k_m.gguf
  curriculum/models/Phi-3-mini-4k-instruct-q4.gguf

Download from HuggingFace:
  https://huggingface.co/bartowski/gemma-2-2b-it-GGUF
  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf

IMPORTANT: Run with airplane mode / Wi-Fi disabled to prove offline capability.
"""

import os
import sys
import time
import pathlib
import textwrap

try:
    import psutil
except ImportError:
    print("psutil not found. Run: pip install psutil")
    sys.exit(1)

try:
    from tabulate import tabulate
except ImportError:
    print("tabulate not found. Run: pip install tabulate")
    sys.exit(1)

try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python not found. Run: pip install llama-cpp-python")
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

MODELS = {
    "Gemma-2-2B-Q4": MODELS_DIR / "gemma-2-2b-it-q4_k_m.gguf",
    "Phi-3-Mini-Q4": MODELS_DIR / "Phi-3-mini-4k-instruct-q4.gguf",
}

# 5 fixed curriculum questions for quality benchmarking
QUESTIONS = [
    "What is photosynthesis and why is it important?",
    "Explain Newton's first law of motion with an example.",
    "What is the difference between a mixture and a pure substance?",
    "Define HCF and LCM. How are they related?",
    "What causes deficiency diseases? Give two examples.",
]

SYSTEM_PROMPT = (
    "You are a helpful educational assistant for Indian school students. "
    "Answer clearly and concisely in 3-5 sentences. "
    "Focus only on factual, curriculum-appropriate information."
)

N_CTX   = 2048   # context window size
N_GPU_LAYERS = 0  # 0 = CPU-only (safe for all hardware)
MAX_TOKENS = 200  # max tokens per answer


def get_rss_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1024 / 1024


def benchmark_model(name: str, model_path: pathlib.Path) -> dict:
    """Load model, run inference, return metrics dict."""
    print(f"\n{'─'*55}")
    print(f"  Model: {name}")
    print(f"  Path : {model_path}")

    if not model_path.exists():
        print(f"  [SKIP] Model file not found: {model_path}")
        print(f"  Download it from HuggingFace and place it in curriculum/models/")
        return {
            "model": name,
            "ram_load_mb": "N/A",
            "load_time_s": "N/A",
            "avg_tps": "N/A",
            "available": False,
        }

    print(f"{'─'*55}")

    ram_before = get_rss_mb()

    # Load model
    t0 = time.time()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    load_time = time.time() - t0
    ram_after  = get_rss_mb()
    ram_used   = ram_after - ram_before

    print(f"  Load time : {load_time:.2f}s")
    print(f"  RAM used  : {ram_used:.0f} MB (process RSS: {ram_after:.0f} MB)")

    total_tokens = 0
    total_time   = 0.0
    answers      = []

    for i, question in enumerate(QUESTIONS, 1):
        prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\nQuestion: {question} [/INST]"
        t1     = time.time()
        resp   = llm(prompt, max_tokens=MAX_TOKENS, echo=False, stop=["</s>", "[INST]"])
        elapsed = time.time() - t1

        answer      = resp["choices"][0]["text"].strip()
        tokens_used = resp["usage"]["completion_tokens"]

        tps = tokens_used / elapsed if elapsed > 0 else 0
        total_tokens += tokens_used
        total_time   += elapsed
        answers.append((question, answer, f"{tps:.1f} tok/s"))

        print(f"\n  Q{i}: {question}")
        print(f"  A : {textwrap.fill(answer, width=65, subsequent_indent='      ')}")
        print(f"      [{tokens_used} tokens, {elapsed:.1f}s, {tps:.1f} tok/s]")

    avg_tps = total_tokens / total_time if total_time > 0 else 0

    # Free model memory
    del llm

    return {
        "model":       name,
        "ram_load_mb": f"{ram_used:.0f}",
        "load_time_s": f"{load_time:.2f}",
        "avg_tps":     f"{avg_tps:.1f}",
        "available":   True,
        "answers":     answers,
    }


def print_comparison(results: list[dict]) -> None:
    rows = [
        [
            r["model"],
            r["ram_load_mb"],
            r["load_time_s"],
            r["avg_tps"],
            "✓" if r["available"] else "✗ (missing)",
        ]
        for r in results
    ]
    headers = ["Model", "RAM (MB)", "Load (s)", "Avg Tok/s", "Available"]
    print(f"\n{'='*55}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*55}")
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    available = [r for r in results if r["available"]]
    if len(available) == 2:
        print("\n  RECOMMENDATION:")
        # Prefer higher tok/s if quality is similar; prefer lower RAM if on-device
        a, b = available
        tps_a = float(a["avg_tps"]) if a["avg_tps"] != "N/A" else 0
        tps_b = float(b["avg_tps"]) if b["avg_tps"] != "N/A" else 0
        winner = a["model"] if tps_a >= tps_b else b["model"]
        print(f"  → Use '{winner}' for Android deployment (higher tok/s).")
        print(f"  → Verify answer quality above before final decision.")
        print(f"  → Set MODEL_PATH env var to the chosen model path.")
    elif len(available) == 1:
        print(f"\n  Only '{available[0]['model']}' is available. Download the other model to compare.")
    else:
        print("\n  No models found in curriculum/models/. See download instructions above.")
    print()


def main():
    print("\n=== LLM Benchmark (CPU-only, Offline) ===")
    print(f"Models directory: {MODELS_DIR}")
    MODELS_DIR.mkdir(exist_ok=True)

    results = []
    for name, path in MODELS.items():
        result = benchmark_model(name, path)
        results.append(result)

    print_comparison(results)


if __name__ == "__main__":
    main()
