"""LLM agent with tool-calling loop and per-session conversation memory."""
import logging
import time
from collections import defaultdict

from groq import AsyncGroq

from .config import settings
from .tools import ToolRegistry

log = logging.getLogger("agent")

SYSTEM_PROMPT = """You are Aria, the voice receptionist for BrightSmile Dental Clinic in Bengaluru.
You are speaking out loud on a call, so:
- Reply in 1 to 3 short sentences. No markdown, bullet points, emojis or URLs.
- Say numbers naturally: "four thirty in the afternoon", "eight hundred rupees".
- Ask one question at a time.

How you work:
- For any question about the clinic (services, prices, hours, doctors, policies, location),
  call search_knowledge_base first and answer only from what it returns. If it has nothing, say
  you're not sure and offer to have the front desk call back. Never invent prices or policies.
- To book: collect name, purpose, date and time (and phone if offered). Call get_current_time to
  resolve words like "tomorrow" or "next Monday". Read the details back and get a clear yes
  BEFORE calling book_appointment. Then confirm with the appointment number.
- If a tool returns an error, explain it simply and offer an alternative.
- For medical emergencies (heavy bleeding, facial swelling with fever, trauma), tell them to go
  to the nearest emergency room now.
- Stay on topic. Politely decline unrelated requests, except small talk like weather or time."""

MAX_TOOL_STEPS = 5


class VoiceAgent:
    def __init__(self, client: AsyncGroq, model: str, tools: ToolRegistry):
        self.client = client
        self.model = model
        self.tools = tools
        self.sessions: dict[str, list[dict]] = defaultdict(list)

    def reset(self, session_id: str):
        self.sessions.pop(session_id, None)

    async def respond(self, session_id: str, user_text: str) -> dict:
        history = self.sessions[session_id]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
                    {"role": "user", "content": user_text}]
        tool_trace, llm_ms, reply = [], 0, ""

        for _ in range(MAX_TOOL_STEPS):
            t0 = time.perf_counter()
            resp = await self.client.chat.completions.create(
                model=self.model, messages=messages, tools=self.tools.schemas,
                tool_choice="auto", temperature=0.3, max_completion_tokens=300,
            )
            llm_ms += (time.perf_counter() - t0) * 1000
            msg = resp.choices[0].message

            if not msg.tool_calls:
                reply = (msg.content or "").strip()
                break

            messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name,
                                             "arguments": tc.function.arguments}}
                               for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                out = await self.tools.execute(tc.function.name, tc.function.arguments)
                tool_trace.append(out)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": _to_json(out["result"])})
        else:
            reply = "Sorry, I got stuck on that one. Could you say it another way?"

        reply = reply or "Sorry, could you repeat that?"
        # Keep only user/assistant text in memory - compact and always valid for the API.
        history += [{"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply}]
        del history[: max(0, len(history) - settings.max_history_turns * 2)]

        tools_ms = sum(t["ms"] for t in tool_trace)
        return {"reply": reply, "tool_calls": tool_trace,
                "llm_ms": round(llm_ms), "tools_ms": tools_ms}


def _to_json(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, default=str)[:6000]
