# Security & Reliability Requirements
## Offline-First Agentic Learning Assistant

### 1. Secrets Management
- Never embed Supabase service-role keys in the Flutter application.
- Never embed backend admin credentials in the APK.
- Store all backend secrets only on the server.

### 2. Transport & Storage
- Use HTTPS for all API communication.
- Use secure Android storage for authentication/session information.

### 3. Backend Hardening
- Validate all API requests server-side.
- Use authentication and authorization for protected endpoints.
- Use parameterized queries or an ORM (no raw string-built SQL).
- Use rate limiting on public endpoints.

### 4. Synchronization Integrity
- Use idempotency keys for synchronization so retries and duplicate sync requests never create duplicate records.
- Do not delete or mark a queue item as synced until the server confirms success.

### 5. Data Privacy
- Do not log passwords, access tokens, API keys, or sensitive user information.
- Send only the necessary data to online LLM services — prefer local inference for ordinary curriculum questions to minimize data leaving the device.

### 6. Content Integrity
- Version and verify curriculum/model bundles (so a corrupted or tampered offline bundle can be detected).

### 7. Reliability as a Security-Adjacent Concern
- Application must remain usable when the internet is completely disabled — no crash when the backend is unavailable.
- Offline questions must not attempt endless/uncontrolled network requests (avoids battery/data drain and hangs).
- Queued actions must survive an application restart (no silent data loss).
- Sync must be retryable without side effects (see idempotency, above).
- Local curriculum must remain available after restart.
- The local model must fail gracefully if unavailable (no crash — degrade to a safe retrieval-only response).
- The user must always be able to see whether the current answer came from the local or online path (transparency of data flow).

### 8. Summary Principle
Security here is inseparable from reliability: the biggest risks in this MVP are (1) leaking secrets through the mobile client, and (2) breaking the offline/sync guarantees the entire pitch depends on. Every design decision should be checked against both.
