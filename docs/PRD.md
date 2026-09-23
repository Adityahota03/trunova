# Product Requirements Document (PRD)
## Offline-First Agentic Learning Assistant

### 1. Objective
Build a reliable Android MVP that helps Class 9 learners learn through voice or text even when internet connectivity is unavailable or unreliable. Only must-have features are in scope — the build should demonstrate problem understanding, strong technical architecture, agentic implementation, reliable offline functionality, presentation quality, and pitch-deck readiness.

### 2. Problem Statement
- **Target learner:** Class 9 students, especially in rural or small-town areas with unreliable internet access.
- **Language gap:** Many learners understand a local language better than English-first educational content.
- **Geography gap:** Learners in low-connectivity areas cannot depend on cloud-only AI or continuous internet access.
- **Core problem:** A learner should be able to ask an educational question and receive a useful, curriculum-grounded explanation even without internet.
- **Why it matters:** Learning should not stop because of connectivity or language barriers.

### 3. Target Users
Class 9 students in low-connectivity, multilingual regions who need voice-first, curriculum-grounded learning support.

### 4. Product Scope

#### Must-Have Screens
1. Home
2. Ask Question
3. Answer + Sources
4. Saved Learning
5. Sync Status
6. Settings

#### Must-Have Features
- Voice question input
- Text question input
- English language support
- One selected local language
- Offline curriculum access
- SQLite FTS5 search
- Offline LLM inference
- Online LLM fallback
- Agent state machine
- Source-grounded answers
- Offline save
- FIFO synchronization
- Clear offline/online status

#### Content
- **Initial subject:** Class 9 curriculum
- **Knowledge source:** Curated Class 9 textbook content and glossary
- **Format:** JSON, SQLite
- **Principle:** Use controlled curriculum content instead of unrestricted web search.

#### Out of Scope
- Payments
- Social networking
- Teacher marketplace
- Complex recommendation engine
- Vector database
- Microservices
- Kubernetes
- Model fine-tuning
- Large analytics platform
- Complex gamification
- iOS application
- Web application

### 5. UI Requirements

**Design principles:**
- Large buttons
- Simple navigation
- Minimal text density
- Voice-first interaction
- Clear online/offline indicator
- Clear sync status
- Readable typography
- Low-data design

**Answer screen must show:**
- Question
- Answer
- Source chapter/topic
- Offline/Online indicator
- Ask follow-up
- Save answer

### 6. Evaluation Criteria (Hackathon)
1. **Problem understanding** — clarity on learner, language gap, geography gap, and core problem.
2. **Technicalities** — offline-first architecture, agentic workflow, reliability under real network conditions.
3. **Presentation** — every technical claim must be demonstrable live.
4. **Pitch deck** — must tell a complete story to judges, CTOs, and venture partners.

### 7. Demo Acceptance Criteria
1. Launch the application in airplane mode.
2. Open Class 9 curriculum content.
3. Ask a question using voice or text.
4. Retrieve relevant curriculum content locally.
5. Generate an answer locally using llama.cpp or a safe retrieval-based fallback.
6. Display the source.
7. Show agent state transitions.
8. Save the interaction.
9. Show it in the FIFO sync queue.
10. Restore internet.
11. Synchronize the action with FastAPI.
12. Verify persistence in Supabase PostgreSQL.
13. Repeat the same sync request and demonstrate idempotent behavior.

### 8. Pitch Deck Outline (10 Slides)
| # | Title | Purpose |
|---|-------|---------|
| 1 | Learning Should Not Stop When Internet Stops | Opening problem statement |
| 2 | Who We Are Solving For | Class 9 learners, language/geography barriers |
| 3 | Our Solution | Introduce the offline-first learning assistant |
| 4 | How the Agent Thinks | State machine, retrieval, decision loop, grounding |
| 5 | System Architecture | Flutter, SQLite, llama.cpp, FastAPI, Supabase |
| 6 | Offline → Online Workflow | Local learning and FIFO synchronization |
| 7 | Security & Reliability | Secrets management, validation, idempotency, privacy |
| 8 | Live Demo | Prove offline question answering and synchronization |
| 9 | Impact & Feasibility | Practical deployment and measurable impact |
| 10 | Scale Beyond the MVP | Future languages, subjects, content packs, deployment |

### 9. Final Deliverables
- Flutter Android application
- FastAPI backend
- Supabase PostgreSQL database
- Offline SQLite curriculum bundle
- SQLite FTS5 retrieval
- llama.cpp offline inference
- Explicit agent state machine
- FIFO sync queue
- Security configuration
- Architecture diagram
- 10-slide pitch deck
- Live demo script

### 10. Guiding Development Principles
1. Reliability over feature count.
2. Do not add unnecessary features.
3. Every feature must work offline where it is claimed to work offline.
4. Keep the agent state machine explicit and observable.
5. Keep curriculum knowledge controlled and source-grounded.
6. Do not expose secrets in the mobile application.
7. Build the smallest working end-to-end system before adding UI polish.
8. The final MVP must be demoable on one Android device.
9. Prefer simple components the team can explain confidently to CTO-level judges.
