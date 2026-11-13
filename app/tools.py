"""Backend tools the LLM can call. Each tool = JSON schema + async function."""
import json
import logging
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import httpx

from . import db
from .config import settings

log = logging.getLogger("tools")

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
    48: "fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain",
    63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


class ToolRegistry:
    def __init__(self, kb=None):
        self.kb = kb
        self.functions = {
            "search_knowledge_base": self.search_knowledge_base,
            "book_appointment": self.book_appointment,
            "list_appointments": self.list_appointments,
            "cancel_appointment": self.cancel_appointment,
            "get_current_time": self.get_current_time,
            "get_weather": self.get_weather,
        }

    # ---------- schemas sent to the LLM ----------
    @property
    def schemas(self) -> list[dict]:
        def fn(name, desc, props, required):
            return {"type": "function", "function": {
                "name": name, "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required}}}

        s = {"type": "string"}
        return [
            fn("search_knowledge_base",
               "Search the clinic knowledge base (services, prices, hours, policies, doctors, location). "
               "Use for ANY factual question about the clinic before answering.",
               {"query": {**s, "description": "Short search query"}}, ["query"]),
            fn("book_appointment",
               "Book an appointment. Only call after the caller has confirmed name, date, time and purpose.",
               {"name": s, "phone": s,
                "date": {**s, "description": "YYYY-MM-DD"},
                "time": {**s, "description": "HH:MM, 24-hour"},
                "purpose": s}, ["name", "date", "time", "purpose"]),
            fn("list_appointments", "List upcoming booked appointments for a person.",
               {"name": s}, ["name"]),
            fn("cancel_appointment", "Cancel an appointment by its id.",
               {"appointment_id": {"type": "integer"}}, ["appointment_id"]),
            fn("get_current_time", "Get today's date, weekday and current time.", {}, []),
            fn("get_weather", "Get current weather for a city.", {"city": s}, ["city"]),
        ]

    async def execute(self, name: str, arguments: str) -> dict:
        """Run a tool call from the LLM. Never raises - errors go back to the LLM as data."""
        start = time.perf_counter()
        try:
            args = json.loads(arguments or "{}")
            func = self.functions.get(name)
            result = await func(**args) if func else {"error": f"Unknown tool {name}"}
        except Exception as exc:
            log.exception("Tool %s failed", name)
            result = {"error": f"{type(exc).__name__}: {exc}"}
        ms = round((time.perf_counter() - start) * 1000)
        log.info("tool=%s args=%s ms=%d", name, arguments, ms)
        return {"name": name, "arguments": arguments, "result": result, "ms": ms}

    # ---------- implementations ----------
    async def search_knowledge_base(self, query: str) -> dict:
        hits = await self.kb.search(query, k=3) if self.kb else []
        if not hits:
            return {"results": [], "note": "Nothing relevant found. Say you don't know."}
        return {"results": [{"source": h["source"], "text": h["text"], "score": h["score"]} for h in hits]}

    async def book_appointment(self, name: str, date: str, time: str, purpose: str, phone: str = "") -> dict:
        when = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        now = datetime.now(ZoneInfo(settings.timezone)).replace(tzinfo=None)
        if when < now:
            return {"error": "That time is in the past. Ask for a future date and time."}
        if when.weekday() == 6:
            return {"error": "Clinic is closed on Sundays."}
        if not (9 <= when.hour < 19):
            return {"error": "Clinic hours are 09:00 to 19:00."}
        with db.connect() as c:
            clash = c.execute(
                "SELECT id FROM appointments WHERE date=? AND time=? AND status='booked'", (date, time)
            ).fetchone()
            if clash:
                return {"error": f"{date} {time} is already taken. Offer a different slot."}
            cur = c.execute(
                "INSERT INTO appointments (name, phone, date, time, purpose) VALUES (?,?,?,?,?)",
                (name.strip(), phone, date, time, purpose),
            )
            return {"status": "booked", "appointment_id": cur.lastrowid,
                    "name": name, "date": date, "time": time, "purpose": purpose}

    async def list_appointments(self, name: str) -> dict:
        with db.connect() as c:
            rows = c.execute(
                "SELECT id, name, date, time, purpose FROM appointments "
                "WHERE lower(name) LIKE ? AND status='booked' AND date >= ? ORDER BY date, time",
                (f"%{name.strip().lower()}%", date.today().isoformat()),
            ).fetchall()
        return {"appointments": [dict(r) for r in rows]}

    async def cancel_appointment(self, appointment_id: int) -> dict:
        with db.connect() as c:
            cur = c.execute(
                "UPDATE appointments SET status='cancelled' WHERE id=? AND status='booked'",
                (appointment_id,),
            )
        return {"status": "cancelled"} if cur.rowcount else {"error": "No booked appointment with that id."}

    async def get_current_time(self) -> dict:
        now = datetime.now(ZoneInfo(settings.timezone))
        return {"date": now.strftime("%Y-%m-%d"), "weekday": now.strftime("%A"),
                "time": now.strftime("%H:%M"), "timezone": settings.timezone}

    async def get_weather(self, city: str) -> dict:
        async with httpx.AsyncClient(timeout=8) as client:
            geo = (await client.get("https://geocoding-api.open-meteo.com/v1/search",
                                    params={"name": city, "count": 1})).json()
            if not geo.get("results"):
                return {"error": f"Could not find a city called {city}."}
            place = geo["results"][0]
            cur = (await client.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": place["latitude"], "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            })).json()["current"]
        return {"city": place["name"], "country": place.get("country"),
                "temperature_c": cur["temperature_2m"], "humidity_pct": cur["relative_humidity_2m"],
                "wind_kmh": cur["wind_speed_10m"],
                "conditions": WEATHER_CODES.get(cur["weather_code"], "unknown")}
