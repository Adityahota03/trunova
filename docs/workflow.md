# Agent Workflow
## Offline-First Agentic Learning Assistant

### 1. State Machine
```
START
  → CAPTURE_INPUT
  → CLASSIFY_INTENT
  → RETRIEVE_LOCAL
  → CHECK_CONTEXT
  → DECIDE_MODE
      ├─ GENERATE_LOCAL   (offline path)
      └─ CALL_BACKEND     (online path)
  → CHECK_GROUNDING
      ├─ ASK_CLARIFICATION  (low confidence)
      └─ RETURN_ANSWER      (sufficient confidence)
  → SAVE_ACTION
      └─ QUEUE_ACTION       (if offline)
  → SYNC_QUEUE               (when online)
  → END
```

### 2. Step-by-Step Decision Logic

| Step | Action |
|------|--------|
| 1 | Capture voice or text question. |
| 2 | Convert voice to text when required. |
| 3 | Classify the learner's intent. |
| 4 | Search SQLite FTS5 for relevant curriculum content. |
| 5 | Check whether sufficient curriculum context was retrieved. |
| 6 | If offline, use llama.cpp with the retrieved context. |
| 7 | If online and additional capability is required, call FastAPI and the online LLM. |
| 8 | Check whether the generated answer is grounded in the retrieved curriculum. |
| 9 | If confidence is low, ask a clarification question or return a safe "not found in curriculum" response. |
| 10 | Return the answer with its source/topic. |
| 11 | Save the learner's action locally. |
| 12 | If offline, add the action to the FIFO sync queue. |
| 13 | When online, synchronize queued actions. |

### 3. Offline/Online Branching
- **Connectivity check:** Android network status is used only as a hint. Actual availability is confirmed via a short, configurable request timeout.
- **Fallback rule:** If the backend cannot be reached within the timeout, the agent automatically switches to the local (llama.cpp + FTS5) path — never hangs or retries indefinitely on the user-facing question flow.

### 4. FIFO Synchronization Rules
- Queue offline actions in order.
- Process the oldest action first (FIFO).
- Retry temporary failures using backoff/retry logic.
- Attach an idempotency key to every queued action.
- Remove or mark an item as synced **only** after successful server confirmation — never delete a queue item speculatively.
- Repeated sync of the same action must not create duplicate server-side records (idempotent by design).
- Queued actions must survive an app restart.

### 5. Grounding & Clarification Loop
- Every generated answer is checked against the retrieved curriculum context before being shown.
- If the grounding/confidence check fails: the agent either asks a targeted clarification question or returns an explicit "not found in curriculum" response — it never fabricates an ungrounded answer.
- Every returned answer carries its source chapter/topic so the learner (and judges) can verify grounding.

### 6. Why This Matters for the Demo
Each state above should be visibly surfaced in the UI during the live demo (state name, retrieval hit, local vs. online path, grounding result) so the architecture is proven live rather than just described in the pitch deck.
