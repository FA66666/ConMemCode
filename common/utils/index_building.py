import hashlib
import re


PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
WHITESPACE_RE = re.compile(r"\s+")


def stable_hash(*parts: str, length: int = 16) -> str:
    payload = "\n".join(part or "" for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def _normalize_unit(text: str) -> str:
    return WHITESPACE_RE.sub(" ", (text or "").strip())


def _split_long_text(text: str, chunk_chars: int, overlap_chars: int, min_chunk_chars: int) -> list[str]:
    text = _normalize_unit(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_chars, text_len)
        if end < text_len:
            window = text[start:end]
            last_break = max(window.rfind(" "), window.rfind("."), window.rfind(","), window.rfind(";"))
            if last_break >= int(chunk_chars * 0.6):
                end = start + last_break + 1

        chunk = text[start:end].strip()
        if chunk and (len(chunk) >= min_chunk_chars or not chunks):
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(end - overlap_chars, 0)
        while start < text_len and text[start].isspace():
            start += 1

    return chunks


def chunk_text(
    text: str,
    chunk_chars: int = 900,
    overlap_chars: int = 120,
    min_chunk_chars: int = 160,
) -> list[str]:
    """Chunk text into paragraph-aware passages.

    Preference order:
    1. Preserve paragraphs when they fit into a chunk.
    2. Fall back to whitespace-aware sliding windows for long paragraphs.
    """
    text = (text or "").replace("\r", "\n")
    paragraphs = [_normalize_unit(part) for part in PARAGRAPH_SPLIT_RE.split(text)]
    paragraphs = [part for part in paragraphs if part]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current_parts, current_len
        if not current_parts:
            return
        chunk = " ".join(current_parts).strip()
        if chunk and (len(chunk) >= min_chunk_chars or not chunks):
            chunks.append(chunk)
        current_parts = []
        current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > chunk_chars:
            flush_current()
            chunks.extend(_split_long_text(paragraph, chunk_chars, overlap_chars, min_chunk_chars))
            continue

        candidate_len = current_len + (1 if current_parts else 0) + len(paragraph)
        if current_parts and candidate_len > chunk_chars:
            flush_current()
        current_parts.append(paragraph)
        current_len = len(" ".join(current_parts))

    flush_current()

    deduped: list[str] = []
    seen = set()
    for chunk in chunks:
        key = stable_hash(chunk)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped
