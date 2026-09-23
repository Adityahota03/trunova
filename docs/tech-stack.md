# Tech Stack
## Offline-First Agentic Learning Assistant

| Layer | Technology | Notes / Reason |
|-------|-----------|-----------------|
| **Frontend** | Flutter (Android) | Polished Android UI, large buttons, voice-first interaction, reliable offline packaging |
| **Local database** | SQLite via Drift | Stores curriculum content, FTS5 search index, learner/session state, saved learning, sync queue |
| **Offline content bundle** | JSON + SQLite | Preloaded essential Class 9 curriculum content shipped as an app asset |
| **Local retrieval** | SQLite FTS5 | Lightweight full-text retrieval without needing a vector database |
| **Offline LLM** | llama.cpp + small quantized model | Packaged or downloaded as an app asset so inference works with no internet |
| **Online LLM** | Small API-based LLM | Used when internet is available and the local model is insufficient |
| **Agent framework** | Explicit custom state machine (preferred), LangGraph (alternative) | States and decision loop must be easy to demo and debug during the hackathon |
| **Backend** | FastAPI (Python), REST | Minimal API surface — auth, sync, health |
| **Cloud database** | PostgreSQL via Supabase | Persistent cloud storage and synchronization target |
| **Sync mechanism** | Local FIFO queue | Queues offline actions; retries; uses idempotency keys; removes/marks synced only after server confirmation |
| **Voice** | Android STT/TTS | Offline-capable speech models used where available |

### Explicitly Out of Scope (Tech)
- Vector database
- Microservices
- Kubernetes
- Model fine-tuning
- Large analytics platform
- iOS application
- Web application

### Rationale Summary
The stack is chosen to keep the system demoable end-to-end on a single Android device, avoid infrastructure the team can't confidently explain to CTO-level judges, and keep every offline claim genuinely backed by on-device components (SQLite FTS5 for retrieval, llama.cpp for generation) rather than network-dependent services.
