#!/usr/bin/env python3
"""
curriculum/agent.py
====================
Stage C: Explicit Agent State Machine

States:
  CAPTURE_INPUT → CLASSIFY_INTENT → RETRIEVE_LOCAL → CHECK_CONTEXT
  → DECIDE_MODE → [GENERATE_LOCAL | CALL_BACKEND]
  → CHECK_GROUNDING → [ASK_CLARIFICATION | RETURN_ANSWER]
  → SAVE_ACTION → [QUEUE_ACTION] → END

Each state transition is logged with timing so judges can see the agent "thinking".
This is the Python prototype — the same state machine will be ported to Flutter later.

Usage:
  python agent.py
  python agent.py "What is Newton's first law?" --class 9 --subject science
"""

import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import time
import json
import uuid
import sqlite3
import pathlib
import argparse
import textwrap
import dataclasses
from enum import Enum, auto
from typing import Optional

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    C = True
except ImportError:
    C = False
    class Fore:
        GREEN = CYAN = YELLOW = RED = MAGENTA = WHITE = BLUE = ""
    class Style:
        BRIGHT = RESET_ALL = DIM = ""

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = pathlib.Path(__file__).parent
DB_PATH     = BASE_DIR / "output" / "curriculum.db"
SAVES_PATH  = BASE_DIR / "output" / "saved_sessions.json"
QUEUE_PATH  = BASE_DIR / "output" / "sync_queue.json"
MODELS_DIR  = BASE_DIR / "models"

DEFAULT_MODELS = [
    MODELS_DIR / "gemma-2-2b-it-q4_k_m.gguf",
    MODELS_DIR / "Phi-3-mini-4k-instruct-q4.gguf",
]

N_CTX = 2048; N_GPU_LAYERS = 0; MAX_TOKENS = 300; TOP_K = 5

STOPWORDS = {
    "what", "is", "are", "was", "were", "a", "an", "the", "in", "on", "of", "and", "or",
    "to", "for", "with", "by", "from", "how", "why", "who", "which", "where", "when",
    "can", "explain", "define", "tell", "me", "about", "give", "describe", "does", "do",
    "kya", "hai", "ka", "ki", "ke", "ko", "se", "mein", "kisko", "kaise",
    "क्या", "है", "हैं", "का", "की", "के", "को", "से", "में", "किसे", "कहते", "होता",
    "होती", "होते", "बताइए", "समझाइए", "कहा", "जाता", "दिए", "और"
}


def build_safe_fts_query(user_query: str) -> str:
    """Sanitize user query into safe FTS5 query tokens joined with OR."""
    tokens = [w for w in re.findall(r"[\w\u0900-\u097f]+", user_query) if len(w) > 1]
    keywords = [t for t in tokens if t.lower() not in STOPWORDS]
    if not keywords:
        keywords = tokens
    if not keywords:
        return ""
    return " OR ".join(f'"{kw}"' for kw in keywords)


# ── State machine ─────────────────────────────────────────────────────────────
class AgentState(Enum):
    IDLE              = auto()
    CAPTURE_INPUT     = auto()
    CLASSIFY_INTENT   = auto()
    RETRIEVE_LOCAL    = auto()
    CHECK_CONTEXT     = auto()
    DECIDE_MODE       = auto()
    GENERATE_LOCAL    = auto()
    CALL_BACKEND      = auto()
    CHECK_GROUNDING   = auto()
    ASK_CLARIFICATION = auto()
    RETURN_ANSWER     = auto()
    SAVE_ACTION       = auto()
    QUEUE_ACTION      = auto()
    SYNC_QUEUE        = auto()
    END               = auto()


@dataclasses.dataclass
class AgentContext:
    question:    str = ""
    language:    str = "en"
    class_level: Optional[int] = None
    subject:     Optional[str] = None
    chunks:      list = dataclasses.field(default_factory=list)
    answer:      str = ""
    source_ref:  str = ""
    mode:        str = "offline"      # 'offline' | 'online'
    grounded:    bool = False
    grounding_score: float = 0.0
    session_id:  str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))


class LearningAgent:
    def __init__(self, llm: Llama, con: sqlite3.Connection, online: bool = False):
        self.llm    = llm
        self.con    = con
        self.online = online
        self.state  = AgentState.IDLE
        self._log: list[dict] = []

    # ── State transition helper ───────────────────────────────────────────────
    def _transition(self, new_state: AgentState, note: str = "", duration: float = 0.0):
        old = self.state.name
        self.state = new_state

        status = f"{Fore.GREEN if C else ''}✓{Style.RESET_ALL if C else ''}" if note and "error" not in note.lower() else f"{Fore.YELLOW if C else ''}●{Style.RESET_ALL if C else ''}"
        dur_str = f"[{duration:.3f}s]" if duration > 0 else ""

        bright = Style.BRIGHT if C else ""
        reset  = Style.RESET_ALL if C else ""
        cyan   = Fore.CYAN if C else ""
        color  = Fore.GREEN if C else ""

        print(f"  {status} {bright}{cyan}{new_state.name:<22}{reset} {color}{note:<35}{reset} {Style.DIM if C else ''}{dur_str}{reset}")
        self._log.append({"state": new_state.name, "note": note, "duration_s": round(duration, 3)})

    # ── Individual state handlers ─────────────────────────────────────────────
    def _capture_input(self, ctx: AgentContext, raw: str) -> None:
        t = time.time()
        ctx.question = raw.strip()
        self._transition(AgentState.CAPTURE_INPUT, f"input={ctx.question[:40]!r}", time.time()-t)

    def _classify_intent(self, ctx: AgentContext) -> None:
        t = time.time()
        # Detect language hint
        hi_chars = sum(1 for c in ctx.question if '\u0900' <= c <= '\u097f')
        if hi_chars > 2 and ctx.language == "en":
            ctx.language = "hi"

        # Detect class level from question if not already specified (e.g. via CLI)
        if ctx.class_level is None:
            m = re.search(r'\bclass\s*(\d+)\b|\bकक्षा\s*(\d+)\b', ctx.question, re.IGNORECASE)
            if m:
                ctx.class_level = int(m.group(1) or m.group(2))

        # Detect subject keywords if not already specified (e.g. via CLI)
        if ctx.subject is None:
            subj_map = {
                "photosynthesis|plant|cell|tissue|animal|digestion|organism|nutrition|respiration|heredity|evolution|force|motion|gravity|energy|electricity|atom|molecule|chemical|acid|base|salt|matter|tissue|newton|law|velocity|acceleration|inertia|प्रकाश संश्लेषण|पादप|कोशिका|ऊतक|बल|गति|ऊर्जा|नियम|जड़त्व|गुरुत्वाकर्षण|त्वरण|संवेग|परमाणु|अणु|पदार्थ": "science",
                "triangle|algebra|geometry|polynomial|fraction|equation|circle|area|volume|number|factor|lcm|hcf|prime|coordinate|त्रिभुज|बीजगणित|ज्यामिति|बहुपद|समीकरण|संख्या": "mathematics",
                "grammar|tense|story|poem|novel|chapter|comprehension|vocabulary|writing|व्याकरण": "english",
            }
            q_lower = ctx.question.lower()
            for pattern, subj in subj_map.items():
                if re.search(pattern, q_lower):
                    ctx.subject = subj
                    break

        note = f"class={ctx.class_level or '?'} subj={ctx.subject or '?'} lang={ctx.language}"
        self._transition(AgentState.CLASSIFY_INTENT, note, time.time()-t)

    def _retrieve_local(self, ctx: AgentContext) -> None:
        t = time.time()
        filters = ["cc.language = ?"]
        params: list = [ctx.language]
        if ctx.class_level:
            filters.append("cc.class_level = ?")
            params.append(ctx.class_level)
        if ctx.subject:
            filters.append("cc.subject = ?")
            params.append(ctx.subject.lower())
        where = " AND ".join(filters)

        fts_query = build_safe_fts_query(ctx.question)
        rows = []
        if fts_query:
            sql = f"""
                SELECT cc.id, cc.class_level, cc.subject, cc.language,
                       cc.curriculum, cc.chapter, cc.topic, cc.content,
                       cc.source_ref, rank
                FROM curriculum_fts
                JOIN curriculum_content cc ON curriculum_fts.rowid = cc.rowid
                WHERE curriculum_fts MATCH ?
                  AND {where}
                ORDER BY rank
                LIMIT ?
            """
            try:
                rows = self.con.execute(sql, [fts_query] + params + [TOP_K]).fetchall()
            except sqlite3.OperationalError:
                rows = []

        if rows:
            ctx.chunks = [dict(r) for r in rows]
        else:
            # Fallback LIKE search respecting class, subject, language filters
            tokens = [w for w in re.findall(r"[\w\u0900-\u097f]+", ctx.question) if len(w) > 1]
            keywords = [t for t in tokens if t.lower() not in STOPWORDS] or tokens
            like_clauses = []
            like_params: list = [ctx.language]
            if ctx.class_level:
                like_params.append(ctx.class_level)
            if ctx.subject:
                like_params.append(ctx.subject.lower())

            for kw in keywords[:5]:
                like_clauses.append("(content LIKE ? OR topic LIKE ? OR chapter LIKE ?)")
                like_params.extend([f"%{kw}%", f"%{kw}%", f"%{kw}%"])

            fallback_where = " AND ".join(filters)
            if like_clauses:
                fallback_where += f" AND ({' OR '.join(like_clauses)})"

            fallback_sql = f"SELECT * FROM curriculum_content WHERE {fallback_where} LIMIT ?"
            like_params.append(TOP_K)
            fallback_rows = self.con.execute(fallback_sql, like_params).fetchall()
            ctx.chunks = [dict(r) for r in fallback_rows]

        self._transition(AgentState.RETRIEVE_LOCAL, f"{len(ctx.chunks)} chunk(s) found", time.time()-t)

    def _check_context(self, ctx: AgentContext) -> bool:
        t = time.time()
        sufficient = len(ctx.chunks) > 0
        note = "sufficient ✓" if sufficient else "insufficient — no curriculum match"
        self._transition(AgentState.CHECK_CONTEXT, note, time.time()-t)
        return sufficient

    def _decide_mode(self, ctx: AgentContext) -> str:
        t = time.time()
        ctx.mode = "online" if self.online else "offline"
        self._transition(AgentState.DECIDE_MODE, f"→ {ctx.mode.upper()}", time.time()-t)
        return ctx.mode

    def _generate_local(self, ctx: AgentContext) -> None:
        t = time.time()
        ctx.source_ref = ctx.chunks[0]["source_ref"] if ctx.chunks else ""

        context_text = "\n\n".join(
            f"[Source {i}: Class {c['class_level']} {c['subject'].title()}, "
            f"{c['chapter']}, {c['topic']}]\n{c['content']}"
            for i, c in enumerate(ctx.chunks, 1)
        )
        prompt = (
            "<s>[INST] <<SYS>>\n"
            "You are a helpful educational assistant for Indian school students. "
            "Answer ONLY using the curriculum context provided. "
            "If the answer is not in the context, say so. "
            "Mention the source chapter/topic at the end.\n"
            "<</SYS>>\n\n"
            f"CURRICULUM CONTEXT:\n{context_text}\n\n"
            f"QUESTION: {ctx.question} [/INST]"
        )
        resp       = self.llm(prompt, max_tokens=MAX_TOKENS, echo=False, stop=["</s>","[INST]"])
        ctx.answer = resp["choices"][0]["text"].strip()
        tok        = resp["usage"]["completion_tokens"]
        elapsed    = time.time() - t
        tps        = tok / elapsed if elapsed > 0 else 0
        self._transition(AgentState.GENERATE_LOCAL, f"{tok} tokens  {tps:.1f} tok/s", elapsed)

    def _call_backend(self, ctx: AgentContext) -> None:
        """Stub — Phase 4 will implement real HTTP call to FastAPI."""
        t = time.time()
        ctx.answer    = "[Online LLM] Answer would come from FastAPI backend (Phase 4)."
        ctx.source_ref = ctx.chunks[0]["source_ref"] if ctx.chunks else ""
        self._transition(AgentState.CALL_BACKEND, "stub (Phase 4)", time.time()-t)

    def _check_grounding(self, ctx: AgentContext) -> bool:
        t = time.time()
        if not ctx.chunks or not ctx.answer.strip():
            ctx.grounded, ctx.grounding_score = False, 0.0
        else:
            context_words = set()
            for c in ctx.chunks:
                text = f"{c.get('content', '')} {c.get('topic', '')} {c.get('chapter', '')}"
                context_words |= {
                    w.lower() for w in re.findall(r"[\w\u0900-\u097f]+", text) if len(w) > 3
                } - STOPWORDS

            answer_words = {
                w.lower() for w in re.findall(r"[\w\u0900-\u097f]+", ctx.answer) if len(w) > 3
            } - STOPWORDS

            if not answer_words:
                ctx.grounded, ctx.grounding_score = True, 0.50
            else:
                matched = answer_words.intersection(context_words)
                score = len(matched) / len(answer_words)
                ctx.grounding_score = round(min(score, 1.0), 2)
                ctx.grounded = (ctx.grounding_score >= 0.35) or (len(matched) >= 3 and ctx.grounding_score >= 0.25)

        note = f"score={ctx.grounding_score} → {'grounded ✓' if ctx.grounded else 'low confidence'}"
        self._transition(AgentState.CHECK_GROUNDING, note, time.time()-t)
        return ctx.grounded

    def _ask_clarification(self, ctx: AgentContext) -> None:
        t = time.time()
        ctx.answer = (
            "I couldn't find a well-grounded answer in the curriculum. "
            "Could you please specify your class (e.g. Class 9) and subject (e.g. Science)?"
        )
        self._transition(AgentState.ASK_CLARIFICATION, "clarification requested", time.time()-t)

    def _return_answer(self, ctx: AgentContext) -> None:
        t = time.time()
        self._transition(AgentState.RETURN_ANSWER, "answer ready", time.time()-t)

    def _save_action(self, ctx: AgentContext) -> None:
        t = time.time()
        record = {
            "id":          ctx.session_id,
            "question":    ctx.question,
            "answer":      ctx.answer,
            "source_ref":  ctx.source_ref,
            "class_level": ctx.class_level,
            "subject":     ctx.subject,
            "language":    ctx.language,
            "mode":        ctx.mode,
            "grounded":    ctx.grounded,
            "timestamp":   time.time(),
        }
        saves = []
        if SAVES_PATH.exists():
            try:
                with open(SAVES_PATH, encoding="utf-8") as f:
                    saves = json.load(f)
                if not isinstance(saves, list):
                    saves = []
            except Exception:
                saves = []
        saves.append(record)
        SAVES_PATH.parent.mkdir(exist_ok=True)
        with open(SAVES_PATH, "w", encoding="utf-8") as f:
            json.dump(saves, f, indent=2, ensure_ascii=False)
        self._transition(AgentState.SAVE_ACTION, "saved locally", time.time()-t)

    def _queue_action(self, ctx: AgentContext) -> None:
        t = time.time()
        entry = {
            "id":              str(uuid.uuid4()),
            "idempotency_key": ctx.session_id,
            "action_type":     "learning_event",
            "payload": {
                "question":    ctx.question,
                "answer":      ctx.answer,
                "source_ref":  ctx.source_ref,
                "class_level": ctx.class_level,
                "subject":     ctx.subject,
                "language":    ctx.language,
                "mode":        ctx.mode,
            },
            "status":      "pending",
            "retry_count": 0,
            "created_at":  time.time(),
        }
        queue = []
        if QUEUE_PATH.exists():
            try:
                with open(QUEUE_PATH, encoding="utf-8") as f:
                    queue = json.load(f)
                if not isinstance(queue, list):
                    queue = []
            except Exception:
                queue = []
        queue.append(entry)
        QUEUE_PATH.parent.mkdir(exist_ok=True)
        with open(QUEUE_PATH, "w", encoding="utf-8") as f:
            json.dump(queue, f, indent=2, ensure_ascii=False)
        self._transition(AgentState.QUEUE_ACTION, "queued for sync (FIFO)", time.time()-t)

    # ── Main run loop ─────────────────────────────────────────────────────────
    def run(self, raw_question: str, class_level: int = None, subject: str = None, language: str = "en") -> AgentContext:
        ctx = AgentContext(language=language, class_level=class_level, subject=subject)

        bright = Style.BRIGHT if C else ""
        reset  = Style.RESET_ALL if C else ""
        magenta = Fore.MAGENTA if C else ""

        print(f"\n{bright}{magenta}{'─'*55}{reset}")
        print(f"{bright}{magenta}  LEARNING AGENT — State Machine{reset}")
        print(f"{bright}{magenta}{'─'*55}{reset}")

        self._capture_input(ctx, raw_question)
        self._classify_intent(ctx)
        self._retrieve_local(ctx)

        context_ok = self._check_context(ctx)
        if not context_ok:
            ctx.answer = "I could not find relevant curriculum content for this question."
            self._transition(AgentState.RETURN_ANSWER, "no context — safe fallback")
        else:
            mode = self._decide_mode(ctx)
            if mode == "offline":
                self._generate_local(ctx)
            else:
                self._call_backend(ctx)

            grounded = self._check_grounding(ctx)
            if grounded:
                self._return_answer(ctx)
            else:
                self._ask_clarification(ctx)

        self._save_action(ctx)
        if not self.online:
            self._queue_action(ctx)

        self._transition(AgentState.END, "session complete")

        # Print final answer
        print(f"\n{'='*55}")
        bright2 = Style.BRIGHT if C else ""
        green   = Fore.GREEN if C else ""
        cyan    = Fore.CYAN if C else ""
        yellow  = Fore.YELLOW if C else ""
        print(f"{bright2}{green}ANSWER:{reset}")
        print(textwrap.fill(ctx.answer, width=65, subsequent_indent="  "))
        print(f"\n{cyan}Source : {ctx.source_ref}{reset}")
        print(f"{yellow}Mode   : {ctx.mode.upper()}  ✓{reset}")
        print(f"{'='*55}\n")

        return ctx


# ── Bootstrap ─────────────────────────────────────────────────────────────────
def load_db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found at {DB_PATH}. Run: python build.py")
        sys.exit(1)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def load_llm() -> Llama:
    if Llama is None:
        print("[ERROR] llama-cpp-python not found. Run: pip install llama-cpp-python")
        sys.exit(1)
    model_path_env = os.environ.get("MODEL_PATH")
    candidates = [pathlib.Path(model_path_env)] if model_path_env else DEFAULT_MODELS
    for p in candidates:
        if p.exists():
            llm = Llama(model_path=str(p), n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, verbose=False)
            return llm
    print("[ERROR] No GGUF model found. Set MODEL_PATH env var or place model in curriculum/models/")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Learning Agent — State Machine Prototype")
    parser.add_argument("question",  nargs="?",  help="Question to ask")
    parser.add_argument("--class",   dest="class_level", type=int)
    parser.add_argument("--subject", dest="subject",     type=str)
    parser.add_argument("--lang",    dest="language",    default="en")
    parser.add_argument("--online",  action="store_true", help="Simulate online mode")
    args = parser.parse_args()

    con    = load_db()
    llm    = load_llm()
    agent  = LearningAgent(llm, con, online=args.online)

    if args.question:
        agent.run(args.question, args.class_level, args.subject, args.language)
    else:
        print("\n=== Learning Agent — Interactive Mode ===")
        print("Type a question. Type 'quit' to exit.\n")
        while True:
            try:
                q = input(f"{Fore.CYAN if C else ''}> {Style.RESET_ALL if C else ''}").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not q or q.lower() in ("quit","exit","q"):
                break
            agent.run(q, args.class_level, args.subject, args.language)

    con.close()


if __name__ == "__main__":
    main()
