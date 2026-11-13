import os
from dotenv import load_dotenv

load_dotenv()


def env(key: str, default=None):
    value = os.getenv(key)
    return value if value not in (None, "") else default


class Settings:
    groq_api_key = env("GROQ_API_KEY")
    llm_model = env("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")
    stt_model = env("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    tts_provider = "browser"

    timezone = env("TIMEZONE", "Asia/Kolkata")
    db_path = env("DB_PATH", "data/agent.db")
    kb_dir = env("KB_DIR", "knowledge_base")
    max_history_turns = int(env("MAX_HISTORY_TURNS", "10"))


settings = Settings()
