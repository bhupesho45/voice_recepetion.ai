import asyncio
from datetime import datetime, timedelta

import pytest

from app import db
from app.config import settings
from app.rag import KnowledgeBase, chunk_text
from app.speech import TextToSpeech, build_client
from app.tools import ToolRegistry


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    path = str(tmp_path / "test.db")
    monkeypatch.setattr(settings, "db_path", path)
    db.init_db(path)


def next_weekday_at(hour=11):
    d = datetime.now() + timedelta(days=2)
    while d.weekday() == 6:
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d"), f"{hour:02d}:00"


def run(coro):
    return asyncio.run(coro)


def test_chunking_keeps_heading_context():
    chunks = chunk_text("# Prices\n\n" + "Cleaning costs 1500.\n\n" * 60, "x.md", max_chars=200)
    assert len(chunks) > 1
    assert chunks[1]["text"].startswith("[Prices]")


def test_keyword_search_finds_root_canal_price():
    kb = KnowledgeBase(settings.kb_dir)
    run(kb.build())
    hits = run(kb.search("how much does root canal treatment cost"))
    assert any("6,000 to 9,000" in h["text"] for h in hits)


def test_book_list_cancel_flow():
    tools = ToolRegistry()
    day, time = next_weekday_at(11)
    booked = run(tools.book_appointment("Ravi Kumar", day, time, "cleaning"))
    assert booked["status"] == "booked"

    clash = run(tools.book_appointment("Someone Else", day, time, "check-up"))
    assert "already taken" in clash["error"]

    listed = run(tools.list_appointments("ravi"))
    assert len(listed["appointments"]) == 1

    cancelled = run(tools.cancel_appointment(booked["appointment_id"]))
    assert cancelled["status"] == "cancelled"


def test_booking_rules():
    tools = ToolRegistry()
    day, _ = next_weekday_at()
    assert "hours" in run(tools.book_appointment("A", day, "21:00", "x"))["error"]
    assert "past" in run(tools.book_appointment("A", "2020-01-01", "10:00", "x"))["error"]


def test_execute_never_raises():
    tools = ToolRegistry()
    out = run(tools.execute("book_appointment", '{"name": "A"}'))
    assert "error" in out["result"]
    out = run(tools.execute("does_not_exist", "{}"))
    assert "Unknown tool" in out["result"]["error"]


def test_groq_client_requires_a_real_api_key(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "replace_with_your_groq_api_key")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        build_client()


def test_tts_uses_browser_without_server_audio():
    assert run(TextToSpeech().synthesize()) == {
        "audio_b64": None, "mime": None, "engine": "browser"
    }


# ---------- agent loop with a fake LLM (no API key needed) ----------
from types import SimpleNamespace as NS

from app.agent import VoiceAgent


class FakeLLM:
    """First call asks for a KB search, second call answers using the tool result."""

    def __init__(self):
        self.calls = []
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs["messages"])
        if len(self.calls) == 1:
            tc = NS(id="call_1", function=NS(name="search_knowledge_base",
                                             arguments='{"query": "cleaning price"}'))
            msg = NS(content=None, tool_calls=[tc])
        else:
            msg = NS(content="A cleaning is fifteen hundred rupees.", tool_calls=None)
        return NS(choices=[NS(message=msg)])


def test_agent_tool_loop_and_memory():
    kb = KnowledgeBase(settings.kb_dir)
    run(kb.build())
    llm = FakeLLM()
    agent = VoiceAgent(llm, "fake", ToolRegistry(kb))
    out = run(agent.respond("s1", "How much is a cleaning?"))

    assert out["reply"].startswith("A cleaning")
    assert out["tool_calls"][0]["name"] == "search_knowledge_base"
    tool_msg = llm.calls[1][-1]
    assert tool_msg["role"] == "tool" and "1,500" in tool_msg["content"]
    assert [m["role"] for m in agent.sessions["s1"]] == ["user", "assistant"]
