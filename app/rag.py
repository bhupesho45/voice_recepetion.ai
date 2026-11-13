"""Local knowledge-base chunking and IDF-weighted keyword search."""
import logging
import math
import re
from pathlib import Path

log = logging.getLogger("rag")


def chunk_text(text: str, source: str, max_chars: int = 700) -> list[dict]:
    """Split on blank lines, keep the latest heading as context, pack into ~max_chars chunks."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks, current, heading = [], "", ""
    for block in blocks:
        if block.startswith("#"):
            heading = block.lstrip("# ").split("\n")[0]
        if len(current) + len(block) > max_chars and current:
            chunks.append(current.strip())
            current = f"[{heading}]\n" if heading else ""
        current += block + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return [{"source": source, "text": c} for c in chunks]


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


class KnowledgeBase:
    def __init__(self, folder: str):
        self.folder = Path(folder)
        self.chunks: list[dict] = []

    @property
    def mode(self) -> str:
        return "keyword"

    async def build(self):
        self.chunks = []
        for path in sorted(self.folder.glob("**/*")):
            if path.suffix.lower() in {".md", ".txt"}:
                self.chunks += chunk_text(path.read_text(encoding="utf-8"), path.name)
        if not self.chunks:
            log.warning("Knowledge base is empty: %s", self.folder)
            return
        log.info("KB ready: %d chunks (%s)", len(self.chunks), self.mode)

    async def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.chunks:
            return []
        # Keyword fallback: IDF-weighted term overlap (rare words like "canal" count more).
        q = _tokens(query)
        docs = [_tokens(c["text"]) for c in self.chunks]
        n = len(docs)
        idf = {t: math.log((n + 1) / (1 + sum(t in d for d in docs))) + 1 for t in q}
        total = sum(idf.values()) or 1
        scored = [(sum(idf[t] for t in q & d) / total, c) for d, c in zip(docs, self.chunks)]
        scored.sort(key=lambda x: -x[0])
        return [{**c, "score": round(s, 3)} for s, c in scored[:k] if s > 0]
