# RAISC Chatbot

Conversational support for [RAISC](https://raisc.org) — a mental health product we are building in public.

This repo is the chatbot service: a FastAPI backend that talks with people in **English** and **Roman Urdu**, runs a short intake, then stays with them in a therapeutic conversation. It sits next to a Next.js frontend and a Django patient backend. We are a small team shipping this in the open so the work, the tradeoffs, and the unfinished edges are visible.

> This is **not** a replacement for professional mental health care. If you or someone you know is in crisis, contact local emergency services or a trusted clinician.

---

## Where the latest work is

GitHub’s default branch is `main`. **That is not the current product.**

| Branch | What it is |
| --- | --- |
| **`stage`** | Latest chatbot. Follow this branch. |
| **`main`** | Older snapshot. Last real product update was May 2025 (Groq-based text chat, no voice agents, no language/intake work below). |

`stage` is about **50 commits ahead of `main`**. The line of work on `stage` includes:

- English + Roman Urdu in the same conversation
- A smart intake questionnaire, then a therapeutic mode
- Session summaries, clinical-style tags, and clinician notes from the patient backend
- Voice **messages** (LiveKit + Deepgram)
- Voice **calls** (LiveKit agent — still work in progress)
- OpenAI `gpt-4o-mini` for text chat (Groq is used on the voice call path)

Recent landmarks on `stage`:

- **Jan 2026** — `new-year-new-me`: current prompt and conversation behavior
- **Dec 2025** — Roman Urdu + questionnaire marked stable
- **Earlier on this branch** — voice messages, voice-call agent, relevance scoring, unified onboarding/therapy flow

Until we merge `stage` into `main`, clone and run **`stage`**:

```sh
git clone https://github.com/hammad-ahmed-01/RAISC-Chatbot.git
cd RAISC-Chatbot
git checkout stage
```

---

## What it does

A patient session is keyed by `session_key` (issued by the Django backend). Each message goes through two phases:

1. **Intake** — collect current condition, how long it has been going on, mental-health history, and physical activity. Off-topic replies are redirected; on-topic replies are extracted into a structured profile.
2. **Therapeutic chat** — short, language-matched replies grounded in that profile, recent session summaries, and optional clinician notes. Responses are kept brief on purpose.

Around that:

- Chat history in **Firestore**
- User profile + doctor summaries on the **Django** backend
- Language detection (Roman Urdu markers, otherwise English)
- Inactivity summaries after a quiet stretch in the session
- Voice notes transcribed into the same chat pipeline
- Full duplex voice calls (experimental, not parallel with the STT worker yet)

---

## How the pieces fit

```
Next.js app (chat UI, mic, LiveKit room)
        │
        ▼
This repo — FastAPI
  POST /api/chat
  GET  /api/history/{session_key}
  GET  /api/history/{session_key}/{start}/{end}
        │
        ├── OpenAI (text LLM)
        ├── Firestore (transcripts)
        └── Django (patient profile, clinician notes)

LiveKit workers (separate processes)
  stt_agent.py     voice messages → text → /api/chat
  voice_agent.py   live call (Groq STT/LLM + Azure TTS)  [WIP]
```

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/chat` | `{ "session_key": "...", "message": "..." }` → `{ "response": "..." }` |
| `GET` | `/api/history/{session_key}` | Full transcript |
| `GET` | `/api/history/{session_key}/{start}/{end}` | Slice of the transcript |
| `GET` | `/` | Health-style ping (includes CORS allowlist) |

CORS is env-driven (`ALLOWED_ORIGINS`, comma-separated). Defaults include local dev ports and RAISC web origins.

---

## Run it locally

You need Python 3.11+, and on Windows the [MSVC C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (sentence-transformers / Chroma often fail to compile without them).

```sh
git checkout stage
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` in the repo root. **Never commit it.**

```env
# Text chat
OPENAI_API_KEY=

# Firestore — base64-encoded service account JSON
GOOGLE_CREDENTIALS=
PROJECT_ID=

# Patient backend (Django)
DJANGO_BACKEND_URL=http://127.0.0.1:8000

# Optional: comma-separated browser origins
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Voice messages (STT worker)
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
DEEPGRAM_API_KEY=

# Voice calls (WIP)
GROQ_API_KEY=
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
```

`GOOGLE_CREDENTIALS` is the service-account JSON, base64-encoded, matching what `app/utils.py` expects.

Keep the chatbot on a **different port** from Django.

**Text chat only**

```sh
uvicorn main:app --reload --port 8001
```

**Voice messages** (second terminal)

```sh
python run_stt_agent.py download-files
python run_stt_agent.py dev
```

**Voice calls** (experimental; do not run next to the STT worker)

```sh
python voice_agent.py download-files
python voice_agent.py dev
# or: python voice_agent.py console
```

A registered patient in the Django backend is required so `session_key` resolves to a profile.

---

## Repo layout

```
main.py                 FastAPI app, CORS, routers
app/routers/            /chat and /history
app/services/
  chat_service.py       session flow, intake → therapy, summaries
  smart_questionnaire.py
  language_service.py   English / Roman Urdu
  rag_service.py        context-grounded replies
  relevance.py          on-topic vs off-topic during intake
  firestore_service.py
  user_service.py       Django patient API
app/rag_config.py       therapeutic system prompt + LLM
stt_agent.py            LiveKit voice-message worker
voice_agent.py          LiveKit voice-call agent (WIP)
```

---

## Team

Built by the RAISC engineering group, in public:

- [Hammad Ahmed](https://github.com/hammad-ahmed-01)
- [Abdullah Khan](https://github.com/921abdullah)
- [Usman Javaid](https://github.com/Usman-Javaid1234)

The product lives at [raisc.org](https://raisc.org). This chatbot is one of three services in that stack (web app, patient backend, this API).

If you are reading this because we opened the repo: issues and thoughtful PRs against **`stage`** are welcome. Please do not open PRs that add secrets, real credentials, or production dumps.

---

## Status

Honest snapshot while we build in public:

- [x] Text chat with intake + therapeutic modes
- [x] English and Roman Urdu
- [x] History, summaries, clinician-note context
- [x] Voice messages
- [ ] Voice calls production-ready
- [ ] `stage` merged back into `main`
- [ ] Knowledge-base retrieval fully wired (Chroma is in dependencies; the live path currently grounds on session context rather than a vector index)

Keys and cloud credentials belong in environment variables or your host’s secret store (e.g. Railway), never in source.
