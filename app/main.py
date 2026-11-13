"""FastAPI entrypoint: voice pipeline = STT -> LLM agent (+tools/RAG) -> TTS."""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db
from .agent import VoiceAgent
from .config import settings
from .rag import KnowledgeBase
from .speech import SpeechToText, TextToSpeech, build_client
from .tools import ToolRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("api")
STATIC = Path(__file__).parent / "static"
MAX_AUDIO_BYTES = 10 * 1024 * 1024
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    client = build_client()
    kb = KnowledgeBase(settings.kb_dir)
    await kb.build()
    state.update(
        kb=kb,
        agent=VoiceAgent(client, settings.llm_model, ToolRegistry(kb)),
        stt=SpeechToText(client, settings.stt_model),
        tts=TextToSpeech(),
    )
    log.info("Ready | llm=%s stt=%s tts=%s kb=%s",
             settings.llm_model, settings.stt_model, settings.tts_provider, kb.mode)
    yield


app = FastAPI(title="Voice AI Agent", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    speak: bool = True


async def run_turn(session_id: str, user_text: str, speak: bool, stt_ms: int = 0) -> dict:
    t0 = time.perf_counter()
    try:
        result = await state["agent"].respond(session_id, user_text)
    except Exception as exc:
        log.exception("LLM failed")
        raise HTTPException(502, f"Language model error: {exc}") from exc

    audio = {"audio_b64": None, "mime": None, "engine": "off"}
    tts_ms = 0
    if speak:
        t1 = time.perf_counter()
        try:
            audio = await state["tts"].synthesize()
        except Exception as exc:  # degrade gracefully: browser voice
            log.warning("TTS failed, browser fallback: %s", exc)
            audio = {"audio_b64": None, "mime": None, "engine": "browser"}
        tts_ms = round((time.perf_counter() - t1) * 1000)

    agent_ms = round((time.perf_counter() - t0) * 1000) - tts_ms
    timings = {"stt_ms": stt_ms, "llm_ms": result["llm_ms"], "tools_ms": result["tools_ms"],
               "tts_ms": tts_ms, "total_ms": stt_ms + agent_ms + tts_ms}
    log.info("session=%s timings=%s tools=%s", session_id[:8], timings,
             [t["name"] for t in result["tool_calls"]])
    return {"session_id": session_id, "transcript": user_text, "reply": result["reply"],
            "tool_calls": result["tool_calls"], "timings": timings, **audio}


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "llm": settings.llm_model, "stt": settings.stt_model,
            "tts": settings.tts_provider, "kb_chunks": len(state["kb"].chunks),
            "kb_mode": state["kb"].mode}


@app.post("/api/voice")
async def voice(audio: UploadFile = File(...), session_id: str = Form(None), speak: bool = Form(True)):
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Empty audio")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio too long - keep it under about a minute")
    session_id = session_id or str(uuid.uuid4())

    t0 = time.perf_counter()
    try:
        text = await state["stt"].transcribe(data, audio.filename or "audio.webm",
                                             audio.content_type or "audio/webm")
    except Exception as exc:
        log.exception("STT failed")
        raise HTTPException(502, f"Speech-to-text error: {exc}") from exc
    stt_ms = round((time.perf_counter() - t0) * 1000)

    if len(text) < 2:
        return {"session_id": session_id, "transcript": "", "reply": "", "tool_calls": [],
                "timings": {"stt_ms": stt_ms}, "audio_b64": None, "mime": None,
                "engine": "none", "note": "No speech detected"}
    return await run_turn(session_id, text, speak, stt_ms)


@app.post("/api/chat")
async def chat(body: ChatIn):
    return await run_turn(body.session_id or str(uuid.uuid4()), body.text.strip(), body.speak)


@app.post("/api/reset")
async def reset(session_id: str = Form(...)):
    state["agent"].reset(session_id)
    return {"status": "reset"}
