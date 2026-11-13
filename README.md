# Free-tier Voice AI Agent

Aria is a voice receptionist demo for a fictional dental clinic. It transcribes microphone audio, answers with a Groq-hosted open-weight model, calls local appointment and knowledge-base tools, and speaks replies using the browser's built-in voice.

## Free-use design

- **Chat:** Groq API Free plan using `openai/gpt-oss-20b`, an open-weight model that supports tool use. The `openai/` prefix is the model ID; the app calls Groq and does not use an OpenAI API key.
- **Speech-to-text:** Groq Whisper Large V3 Turbo through the same Groq API key.
- **Text-to-speech:** Browser Speech Synthesis API. No paid TTS API or server-side voice generation.
- **Knowledge base:** Local Markdown files with keyword/IDF ranking. No embedding service or embedding API key.
- **Appointments:** Local SQLite database.
- **Weather:** Open-Meteo public geocoding and forecast APIs; no API key is configured.
- **Hosting:** Runs on your computer or with Docker. This repository does not provision paid hosting.

Groq's Free plan is $0 and has usage/model limits. Check the [current Groq plan and rate limits](https://console.groq.com/settings/billing/plans) before use; they can change. Do not upgrade or enable paid billing if you want to avoid API charges. The Groq API key is a private account credential, not an open-source key. Store it only in `.env`, which is ignored by Git.

Audio is sent to Groq for transcription, and the resulting text is sent to Groq for chat. Browser speech output depends on the voice support provided by the user's browser and operating system.

## Setup on Windows

1. Create a free Groq account and generate a key at [Groq Console](https://console.groq.com/keys).
2. In PowerShell, from the project directory:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   Copy-Item .env.example .env
   ```

3. Edit `.env` and replace `replace_with_your_groq_api_key` with your key.
4. Install dependencies and start the app:

   ```powershell
   python -m pip install -r requirements.txt
   python -m uvicorn app.main:app --reload
   ```

5. Open <http://localhost:8000>. Microphone access requires localhost or HTTPS. Use a browser with microphone and speech-synthesis support (for example, Chrome or Edge).

Run tests with:

```powershell
python -m pytest -q
```

Docker is optional: after creating `.env`, run `docker compose up --build`.

## Technologies used

| Area | Technologies |
|---|---|
| Language/runtime | Python 3.10+ (Docker image uses Python 3.12), HTML, CSS, JavaScript |
| Web server/API | FastAPI, Pydantic, Uvicorn |
| LLM and transcription | Groq Cloud API, Groq Python SDK (`groq`), GPT-OSS 20B, Whisper Large V3 Turbo |
| Browser audio | MediaRecorder API for microphone capture; Speech Synthesis API for spoken replies |
| Knowledge retrieval | Local Markdown/TXT files, regular-expression tokenization, IDF keyword ranking |
| Data and tools | SQLite, Python `zoneinfo`, HTTPX, Open-Meteo API |
| Configuration and testing | `python-dotenv`, pytest, pytest-asyncio |
| Optional local container | Docker, Docker Compose |

## Features

- Voice and text chat endpoints.
- Tool-calling loop with clinic knowledge search, appointment booking/listing/cancellation, current time, and weather.
- Appointment checks for clinic hours, Sundays, past dates, and duplicate slots.
- Conversation memory per session and reset endpoint.
- Browser speech fallback, latency timings, logs, and a health endpoint.
- No OpenAI API, paid TTS, or paid embeddings integration.

## Run and API

The app starts with:

```powershell
python -m uvicorn app.main:app --reload
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Voice console |
| `GET` | `/api/health` | Model and knowledge-base status |
| `POST` | `/api/voice` | Transcribe audio, respond, and optionally speak |
| `POST` | `/api/chat` | Text chat |
| `POST` | `/api/reset` | Clear a session's conversation memory |

## Project layout

```text
app/
  main.py              FastAPI routes and pipeline
  agent.py             Groq chat/tool-calling loop and session memory
  tools.py             Tool schemas, SQLite booking, Open-Meteo weather
  rag.py               Local document chunking and keyword search
  speech.py            Groq transcription and browser voice response
  config.py            Environment configuration
  db.py                SQLite setup
  static/index.html    Browser voice console
knowledge_base/        Clinic facts
tests/                 Pytest suite
```
