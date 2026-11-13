"""Groq speech-to-text and browser-based speech output."""
from groq import AsyncGroq

from .config import settings


class SpeechToText:
    def __init__(self, client: AsyncGroq, model: str):
        self.client, self.model = client, model

    async def transcribe(self, audio: bytes, filename: str, mime: str) -> str:
        resp = await self.client.audio.transcriptions.create(
            model=self.model, file=(filename, audio, mime),
            prompt="BrightSmile Dental Clinic, appointment, root canal, cleaning, Bengaluru.",
        )
        return (resp.text or "").strip()


class TextToSpeech:
    async def synthesize(self) -> dict:
        return {"audio_b64": None, "mime": None, "engine": "browser"}


def build_client() -> AsyncGroq:
    if not settings.groq_api_key or settings.groq_api_key == "replace_with_your_groq_api_key":
        raise RuntimeError("GROQ_API_KEY is missing. Add your key to the .env file.")
    return AsyncGroq(api_key=settings.groq_api_key)
