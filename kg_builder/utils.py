"""Small helpers shared across the project."""
from __future__ import annotations

import json
import re


def decode_text_bytes(data: bytes) -> str:
    """Decode raw file bytes to text, auto-detecting the encoding.

    Order: BOM sniff (UTF-8/UTF-16/UTF-32) -> strict UTF-8 (reject if it yields
    NUL/mojibake) -> UTF-16/32 without BOM (alternating-NUL heuristic) -> chardet
    guess (reject mojibake) -> CJK fallbacks (reject mojibake) -> lossy UTF-8.

    Without BOM/sanity checks a wide encoding (e.g. UTF-16-LE, common for files
    exported from Windows/Office) is force-read as UTF-8/GBK and produces NUL-
    littered mojibake ("c\\x00y\\x00..."); the LLM then extracts garbage entities
    and the whole graph is poisoned. NUL bytes never appear in genuine text, so we
    use their presence (and U+FFFD) as a mojibake signal at every fallback step.
    """
    if not data:
        return ""

    # 1) BOM-based detection (most reliable)
    if data[:3] == b"\xef\xbb\xbf":
        return data.decode("utf-8-sig", errors="ignore")
    if data[:4] in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        enc = "utf-32-le" if data[:4] == b"\xff\xfe\x00\x00" else "utf-32-be"
        return data[4:].decode(enc, errors="ignore")
    if data[:2] == b"\xff\xfe":
        return data[2:].decode("utf-16-le", errors="ignore")
    if data[:2] == b"\xfe\xff":
        return data[2:].decode("utf-16-be", errors="ignore")

    # 2) strict UTF-8 — only accept if it is genuinely text (no U+FFFD, no NUL)
    try:
        s = data.decode("utf-8")
        if "\ufffd" not in s and "\x00" not in s:
            return s
    except UnicodeDecodeError:
        pass

    # 3) UTF-16/UTF-32 without a BOM: every other (or every 3rd) byte is NUL
    if len(data) >= 2 and len(data) % 2 == 0:
        if data[1::2] == b"\x00" * (len(data) // 2):
            return data.decode("utf-16-le", errors="ignore")
        if data[0::2] == b"\x00" * (len(data) // 2):
            return data.decode("utf-16-be", errors="ignore")
    if len(data) >= 4 and len(data) % 4 == 0:
        if data[1::4] == b"\x00" * (len(data) // 4) and \
           data[2::4] == b"\x00" * (len(data) // 4) and \
           data[3::4] == b"\x00" * (len(data) // 4):
            return data.decode("utf-32-le", errors="ignore")

    # 4) chardet guess — but reject mojibake
    try:
        import chardet

        det = chardet.detect(data)
        enc = (det or {}).get("encoding")
        if enc:
            try:
                s = data.decode(enc, errors="strict")
                if s and "\ufffd" not in s and "\x00" not in s and s.strip():
                    return s
            except Exception:
                pass
    except Exception:
        pass

    # 5) CJK fallbacks — but reject mojibake
    for enc in ("gb18030", "gbk", "big5"):
        try:
            s = data.decode(enc, errors="strict")
            if s and "\ufffd" not in s and "\x00" not in s and s.strip():
                return s
        except Exception:
            pass

    return data.decode("utf-8", errors="ignore")


def extract_json(text: str | None):
    """Best-effort extraction of a JSON object/array from an LLM response.

    Handles ```json fenced blocks, trailing prose, and (best-effort) truncated
    JSON caused by max_tokens limits by appending the missing closing brackets.
    """
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()

    candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not candidates:
        return None
    start = min(candidates)
    body = text[start:]

    # 1) try the whole body, then progressively shorter suffixes
    for end in range(len(body), 0, -1):
        try:
            return json.loads(body[:end])
        except Exception:
            continue
    # 2) truncated? repair by closing open braces/brackets
    try:
        return _repair_json(body)
    except Exception:
        return None


def _repair_json(s: str):
    """Best-effort repair of a JSON string truncated by an output token limit.

    Handles: unterminated string, a dangling trailing property (e.g. ``"type``
    or ``:"partial``), a trailing comma, and unclosed braces/brackets.
    """
    # 1) close an unterminated string
    in_str = False
    esc = False
    for ch in s:
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            in_str = not in_str
    if in_str:
        s = s + '"'

    # 2) trim dangling trailing tokens
    s = s.rstrip()
    s = re.sub(r",\s*$", "", s)            # trailing comma
    s = re.sub(r':\s*"[^"]*$', "", s)      # trailing :"partial
    s = re.sub(r',\s*"[^"]*"?\s*$', "", s)  # trailing ,"key"(incomplete)

    # 3) close any still-open braces/brackets (ignoring string contents)
    stack = []
    in_str = False
    esc = False
    for ch in s:
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif in_str:
            continue
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    if stack:
        s = s + "".join("]" if t == "[" else "}" for t in reversed(stack))
    return json.loads(s)


def normalize_name(name: str) -> str:
    """Normalize an entity name for de-duplication (case/space insensitive)."""
    if not name:
        return ""
    return re.sub(r"\s+", "", str(name)).lower()


def clean_text(text: str) -> str:
    """Collapse whitespace and strip encoding artifacts / placeholder glyphs.

    Removes U+FFFD replacement chars (from any bad decode) and runs of 2+ ASCII
    '?' (e.g. the literal ``??`` that chat/office exports leave where an emoji or
    icon could not be represented) so they don't pollute entity/relation extraction.
    """
    if not text:
        return ""
    text = text.replace("\ufffd", "")
    text = re.sub(r"\?{2,}", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
