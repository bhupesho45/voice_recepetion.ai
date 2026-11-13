"""SQLite store for appointments (stands in for a real business system / MySQL)."""
import os
import sqlite3
from contextlib import contextmanager

from .config import settings


@contextmanager
def connect(path: str | None = None):
    path = path or settings.db_path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: str | None = None):
    with connect(path) as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                phone TEXT,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                purpose TEXT,
                status TEXT NOT NULL DEFAULT 'booked',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
