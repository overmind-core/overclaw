"""NLP-ish helpers not used by the router agent."""

import re
import unicodedata


def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "empty"


def rough_reading_time_words(body: str, wpm: int = 220) -> float:
    words = len(re.findall(r"\b\w+\b", body))
    return words / float(wpm)
