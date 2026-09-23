#!/usr/bin/env python3
"""
curriculum/test_rag.py
=======================
Stage B: Full RAG pipeline prototype.

Flow:
  Question (text)
        ↓
  SQLite FTS5 search  (optional filters: class_level, subject, language)
        ↓
  Top-K curriculum chunks  (with source_ref)
        ↓
  Prompt = system_prompt + curriculum_context + question
        ↓
  llama.cpp (chosen GGUF model)
        ↓
  Answer + source citation + grounding check

Usage:
  python test_rag.py
  python test_rag.py "What is Newton's first law?"
  python test_rag.py "What is photosynthesis?" --class 9 --subject science
  python test_rag.py "Photosynthesis kya hai?" --language hi

Environment variable to select model:
  set MODEL_PATH=models/gemma-2-2b-it-q4_k_m.gguf
  set MODEL_PATH=models/Phi-3-mini-4k-instruct-q4.gguf

PROVE OFFLINE CAPABILITY: Run with airplane mode / Wi-Fi disabled.
"""

import os
import sys
import time
import sqlite3
import pathlib
import argparse
import textwrap

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class Fore:
        GREEN = CYAN = YELLOW = RED = MAGENTA = WHITE = ""
    class Style:
        BRIGHT = RESET_ALL = DIM = ""

try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python not found. Run: pip install llama-cpp-python")
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
DB_PATH    = BASE_DIR / "output" / "curriculum.db"

# Model selection: prefer env var, then try known paths
DEFAULT_MODELS = [
    BASE_DIR / "models" / "gemma-2-2b-it-q4_k_m.gguf",
    BASE_DIR / "models" / "Phi-3-mini-4k-instruct-q4.gguf",
]

N_CTX        = 2048
N_GPU_LAYERS = 0       # 0 = pure CPU
MAX_TOKENS   = 300
TOP_K        = 5       # number of FTS5 results to retrieve
MIN_CONTENT_KEYWORDS = 2  # grounding check threshold


def cprint(color: str, label: str, text: str = "") -> None:
    bright = Style.BRIGHT if HAS_COLOR else ""
    reset  = Style.RESET_ALL if HAS_COLOR else ""
    if text:
        print(f"{bright}{color}[{label}]{reset} {text}")
    else:
        print(f"{bright}{color}{label}{reset}")


# ── Database ─────────────────────────────────────────────────────────────────
def get_db(db_path: pathlib.Path) -> sqlite3.Connection:
    if not db_path.exists():
        cprint(Fore.RED, "ERROR", f"Database not found at {db_path}")
        cprint(Fore.YELLOW, "FIX", "Run:  python build.py")
        sys.exit(1)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def search_fts(
    con: sqlite3.Connection,
    query: str,
    class_level: int | None = None,
    subject: str | None = None,
    language: str = "en",
    top_k: int = TOP_K,
) -> list[dict]:
    """
    FTS5 search with optional filters on class_level, subject, language.
    Returns list of matching curriculum records, ordered by FTS relevance.
    """
    # Build WHERE clause for structured filters on the base table
    filters = ["cc.language = ?"]
    params  = [language]

    if class_level is not None:
        filters.append("cc.class_level = ?")
        params.append(class_level)

    if subject is not None:
        filters.append("cc.subject = ?")
        params.append(subject.lower())

    where = " AND ".join(filters)

    sql = f"""
        SELECT
            cc.id, cc.class_level, cc.subject, cc.language,
            cc.curriculum, cc.chapter, cc.topic, cc.content, cc.source_ref,
            rank
        FROM curriculum_fts
        JOIN curriculum_content cc ON curriculum_fts.rowid = cc.rowid
        WHERE curriculum_fts MATCH ?
          AND {where}
        ORDER BY rank
        LIMIT ?
    """
    params_full = [query] + params + [top_k]
    try:
        rows = con.execute(sql, params_full).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError as e:
        cprint(Fore.YELLOW, "WARN", f"FTS error: {e}. Falling back to LIKE search.")
        return fallback_search(con, query, class_level, subject, language, top_k)


def fallback_search(
    con, query, class_level, subject, language, top_k
) -> list[dict]:
    """LIKE-based fallback when FTS query syntax is invalid."""
    filters = ["language = ?"]
    params  = [language]
    if class_level: filters.append("class_level = ?"); params.append(class_level)
    if subject:     filters.append("subject = ?");     params.append(subject.lower())
    for word in query.split()[:3]:
        filters.append("content LIKE ?")
        params.append(f"%{word}%")
    where = " AND ".join(filters)
    sql   = f"SELECT * FROM curriculum_content WHERE {where} LIMIT ?"
    params.append(top_k)
    rows  = con.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


# ── LLM ──────────────────────────────────────────────────────────────────────
def load_model() -> Llama:
    model_path_env = os.environ.get("MODEL_PATH")
    if model_path_env:
        candidates = [pathlib.Path(model_path_env)]
    else:
        candidates = DEFAULT_MODELS

    for path in candidates:
        if path.exists():
            cprint(Fore.CYAN, "MODEL", f"Loading: {path.name}")
            t0  = time.time()
            llm = Llama(model_path=str(path), n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, verbose=False)
            cprint(Fore.GREEN, "MODEL", f"Loaded in {time.time()-t0:.2f}s")
            return llm

    cprint(Fore.RED, "ERROR", "No GGUF model file found.")
    cprint(Fore.YELLOW, "FIX",
           "Place a GGUF model in curriculum/models/ and set MODEL_PATH env var.\n"
           "  Download: https://huggingface.co/bartowski/gemma-2-2b-it-GGUF\n"
           "  Download: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")
    sys.exit(1)


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: Class {c['class_level']} {c['subject'].title()}, "
            f"{c['chapter']}, Topic: {c['topic']}]\n{c['content']}"
        )
    context = "\n\n".join(context_parts)

    system = (
        "You are a helpful educational assistant for Indian school students (Classes 6-10). "
        "Answer ONLY using the curriculum context provided below. "
        "If the answer is not in the context, say: 'I could not find this in the curriculum.' "
        "Always mention the source chapter/topic at the end."
    )
    return (
        f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
        f"CURRICULUM CONTEXT:\n{context}\n\n"
        f"QUESTION: {question} [/INST]"
    )


def check_grounding(answer: str, chunks: list[dict]) -> tuple[bool, float]:
    """
    Simple keyword-based grounding check.
    Counts how many key words from the retrieved chunks appear in the answer.
    Returns (is_grounded, confidence_score).
    """
    if not chunks:
        return False, 0.0

    # Collect key nouns/terms from chunks (crude but fast, no extra deps)
    stop_words = {"the","a","an","is","are","was","were","of","and","or","in","to","for",
                  "with","by","from","that","this","it","be","been","being","have","has",
                  "had","do","does","did","will","would","could","should","may","might",
                  "on","at","as","not","but","if","so","its","their","our","we","you","they",
                  "he","she","i","my","your","his","her","what","which","who","when","where",
                  "how","why","also","can","all","more","some","any","each","than","then",
                  "into","through","during","before","after","above","below","between"}

    all_words = set()
    for c in chunks:
        words = {w.lower().strip(".,;:()[]") for w in c["content"].split() if len(w) > 4}
        all_words |= words - stop_words

    if not all_words:
        return True, 0.5

    answer_lower = answer.lower()
    matched = sum(1 for w in all_words if w in answer_lower)
    score   = min(matched / max(len(all_words) * 0.15, 1), 1.0)
    return score >= 0.3, round(score, 2)


# ── Main RAG pipeline ─────────────────────────────────────────────────────────
def run_rag(
    question: str,
    llm: Llama,
    con: sqlite3.Connection,
    class_level: int | None = None,
    subject: str | None = None,
    language: str = "en",
) -> None:
    separator = "─" * 55

    print(f"\n{separator}")
    cprint(Fore.MAGENTA, "QUESTION", question)
    if class_level: print(f"  Filter → Class {class_level}")
    if subject:     print(f"  Filter → Subject: {subject}")
    print(f"  Language: {language}")
    print(separator)

    # Step 1: FTS5 Retrieval
    t_ret = time.time()
    cprint(Fore.CYAN, "STATE", "RETRIEVE_LOCAL (FTS5 search...)")
    chunks = search_fts(con, question, class_level, subject, language)
    ret_time = time.time() - t_ret

    if not chunks:
        cprint(Fore.YELLOW, "STATE", "CHECK_CONTEXT → no curriculum context found")
        cprint(Fore.RED,    "ANSWER", "I could not find relevant content in the curriculum for this question.")
        return

    cprint(Fore.GREEN, "STATE",
           f"RETRIEVE_LOCAL → {len(chunks)} chunk(s) found in {ret_time:.3f}s")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] Class {c['class_level']} {c['subject'].title()} | "
              f"{c['chapter']} | {c['topic']}  ({c['source_ref']})")

    # Step 2: Context check
    cprint(Fore.CYAN, "STATE", "CHECK_CONTEXT → sufficient context retrieved ✓")

    # Step 3: Mode decision
    cprint(Fore.CYAN, "STATE", "DECIDE_MODE → OFFLINE (local LLM)")

    # Step 4: Local generation
    prompt = build_prompt(question, chunks)
    cprint(Fore.CYAN, "STATE", "GENERATE_LOCAL (llama.cpp inference...)")
    t_gen = time.time()
    resp  = llm(prompt, max_tokens=MAX_TOKENS, echo=False, stop=["</s>", "[INST]", "\n\n\n"])
    gen_time = time.time() - t_gen

    answer     = resp["choices"][0]["text"].strip()
    tok_count  = resp["usage"]["completion_tokens"]
    tps        = tok_count / gen_time if gen_time > 0 else 0

    cprint(Fore.GREEN, "STATE",
           f"GENERATE_LOCAL → done in {gen_time:.1f}s ({tok_count} tokens, {tps:.1f} tok/s)")

    # Step 5: Grounding check
    grounded, score = check_grounding(answer, chunks)
    cprint(Fore.CYAN, "STATE", f"CHECK_GROUNDING → score={score}")

    if grounded:
        cprint(Fore.GREEN, "STATE", "CHECK_GROUNDING → grounded ✓  → RETURN_ANSWER")
    else:
        cprint(Fore.YELLOW, "STATE",
               "CHECK_GROUNDING → low confidence → ASK_CLARIFICATION")
        answer = ("This answer may not be fully grounded in the curriculum. "
                  "Please ask a more specific question or specify your class and subject.\n\n"
                  + answer)

    # Step 6: Output
    print(f"\n{'='*55}")
    cprint(Fore.GREEN, "ANSWER", "")
    print(textwrap.fill(answer, width=65, subsequent_indent="  "))
    print()
    cprint(Fore.CYAN, "SOURCE", chunks[0]["source_ref"])
    cprint(Fore.YELLOW, "MODE", "OFFLINE  ✓  (no internet required)")
    print(f"{'='*55}\n")


def interactive_mode(llm: Llama, con: sqlite3.Connection) -> None:
    print(f"\n{Fore.MAGENTA if HAS_COLOR else ''}=== Offline RAG — Interactive Mode ==={Style.RESET_ALL if HAS_COLOR else ''}")
    print("Type a question and press Enter. Type 'quit' to exit.")
    print("Optional prefixes:  [class:9]  [subject:science]  [lang:hi]\n")

    while True:
        try:
            raw = input(f"{Fore.CYAN if HAS_COLOR else ''}Question: {Style.RESET_ALL if HAS_COLOR else ''}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not raw or raw.lower() in ("quit", "exit", "q"):
            break

        # Parse optional prefixes
        question = raw
        class_level = subject = None
        language = "en"

        import re
        for match in re.finditer(r'\[(\w+):(\w+)\]', raw):
            key, val = match.group(1), match.group(2)
            if key == "class":    class_level = int(val)
            elif key == "subject": subject = val
            elif key == "lang":   language = val
            question = re.sub(r'\[.*?\]', '', question).strip()

        run_rag(question, llm, con, class_level, subject, language)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline RAG pipeline — Phase 1 prototype"
    )
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--class",   dest="class_level", type=int, help="Filter by class level (e.g. 9)")
    parser.add_argument("--subject", dest="subject",     type=str, help="Filter by subject (e.g. science)")
    parser.add_argument("--language",dest="language",    type=str, default="en", help="Language code: en, hi (default: en)")
    args = parser.parse_args()

    print("\n=== Offline-First RAG Prototype ===")
    print(f"  DB     : {DB_PATH}")

    con = get_db(DB_PATH)
    llm = load_model()

    if args.question:
        run_rag(args.question, llm, con, args.class_level, args.subject, args.language)
    else:
        interactive_mode(llm, con)

    con.close()


if __name__ == "__main__":
    main()
