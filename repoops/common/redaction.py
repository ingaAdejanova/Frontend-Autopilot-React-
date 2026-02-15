import re

_SECRET_PATTERNS = [
    re.compile(r"(OPENAI_API_KEY=)([A-Za-z0-9_\-]+)"),
    re.compile(r"(sk-[A-Za-z0-9]{10,})"),
]

def redact(text: str) -> str:
    out = text
    for pat in _SECRET_PATTERNS:
        out = pat.sub(r"\1***REDACTED***", out)
    return out
