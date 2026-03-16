from __future__ import annotations

import re


URL_RE = re.compile(r"https?://[^\s<>\")\]]+")


def extract_urls(text: str, *, unique: bool = False) -> list[str]:
    urls = URL_RE.findall(text or "")
    if not unique:
        return urls

    seen: list[str] = []
    for url in urls:
        if url not in seen:
            seen.append(url)
    return seen


def sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", (text or "").strip())
    return len([part for part in parts if part.strip()])
