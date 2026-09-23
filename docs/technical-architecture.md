# Technical Architecture
## Offline-First Agentic Learning Assistant

### 1. Architecture Overview
An offline-first architecture with explicit agent states, local retrieval, local LLM inference, online fallback, and reliable synchronization. Core learning must work fully without internet; online connectivity only adds fallback generation, sync, and cloud persistence.

```
┌─────────────────────────────┐
│      Flutter Android App     │
│  (Voice/Text UI, Agent SM)   │
├───────────────┬───────────────┤
│  Local SQLite  │  llama.cpp    │
│  (Drift, FTS5) │  (offline LLM)│
└───────┬────────┴───────┬───────┘
        │  online/timeout  │
        ▼                  ▼
┌─────────────────────────────┐
│       FastAPI Backend        │
│   (REST, auth, sync, health) │
└───────────────┬───────────────┘
                 ▼
     ┌───────────────────────┐
     │ Supabase PostgreSQL    │
     │ (users, learning_events,│
     │  sync_events)          │
     └───────────────────────┘
```

### 2. Agentic Workflow (States)
```
START → CAPTURE_INPUT → CLASSIFY_INTENT → RETRIEVE_LOCAL → CHECK_CONTEXT
     → DECIDE_MODE → [GENERATE_LOCAL | CALL_BACKEND] → CHECK_GROUNDING
     → [ASK_CLARIFICATION | RETURN_ANSWER] → SAVE_ACTION
     → [QUEUE_ACTION] → SYNC_QUEUE → END
```

**Required agentic capabilities:**
- Stateful workflow
- Intent classification
- Local knowledge retrieval
- Context sufficiency decision
- Offline/online decision
- Local or online generation
- Grounding/confidence check
- Clarification loop
- Local action storage
- FIFO synchronization

See `workflow.md` for the full step-by-step decision logic.

### 3. Offline / Online Logic

**Offline (no network dependency for core learning):**
- View curriculum
- Search curriculum
- Ask questions
- Local LLM inference
- Save learning activity
- Queue synchronization actions

**Online (adds):**
- Backend API calls
- Online LLM fallback
- Synchronization
- Cloud persistence

**Connectivity detection rule:**
Use Android connectivity information only as a hint. Confirm actual availability using a short, configurable request timeout — never trust network status alone. Automatically fall back to the local path when the backend cannot be reached.

### 4. Data Layer

**Local SQLite (via Drift) — minimum tables:**
- `curriculum_content`
- `sessions`
- `saved_learning`
- `sync_queue`

Also stores the FTS5 search index and learner/session state.

**Cloud (Supabase PostgreSQL) — minimum tables:**
- `users`
- `learning_events`
- `sync_events`

### 5. API Design
Minimal REST surface — no endpoints beyond what the MVP requires.

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/auth/login` | Authenticate user (if auth included in MVP) |
| POST | `/api/sync` | Synchronize queued offline actions |
| GET | `/api/health` | Backend health check |

### 6. Reliability Requirements
- Application must remain usable when the internet is completely disabled.
- No crash when the backend is unavailable.
- Offline questions must not attempt endless network requests.
- Queued actions must survive application restart.
- Sync must be retryable.
- Duplicate synchronization must not create duplicate records.
- Local curriculum must remain available after restart.
- The local model must fail gracefully if unavailable.
- The user must always know whether the current answer came from the local or online path.

### 7. Live Demo Flow (proves the architecture)
1. Start with internet enabled.
2. Ask a Class 9 curriculum question using voice.
3. Show agent states.
4. Show retrieved curriculum source.
5. Disable internet.
6. Ask another curriculum question.
7. Demonstrate local SQLite retrieval.
8. Demonstrate llama.cpp local inference.
9. Show answer with curriculum source.
10. Save an action while offline.
11. Show FIFO sync queue.
12. Restore internet.
13. Synchronize queued data to the backend.
14. Show successful synchronization.

### 8. Design Principle
The live demo must prove the architecture rather than only describe it — every architectural claim in the pitch deck should map to a visible step in the demo.
