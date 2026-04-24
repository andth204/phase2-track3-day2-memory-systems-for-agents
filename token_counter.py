"""
Token Counter — tiktoken với fallback offline
==============================================
Cố gắng dùng tiktoken (accurate, bonus point).
Nếu tiktoken không download được (offline/blocked),
fallback sang ước lượng: len(text) // 4
(xấp xỉ chuẩn OpenAI: ~4 chars per token cho tiếng Anh,
 ~2-3 chars cho tiếng Việt → dùng 3 để an toàn).
"""

from __future__ import annotations

_encoder = None
_use_fallback = False


def _init_encoder() -> None:
    global _encoder, _use_fallback
    if _encoder is not None or _use_fallback:
        return
    try:
        import tiktoken
        _encoder = tiktoken.get_encoding("cl100k_base")
        _use_fallback = False
    except Exception:
        _use_fallback = True


def count_tokens(text: str) -> int:
    """
    Count tokens in text.
    Uses tiktoken cl100k_base when available, else len(text)//3 fallback.
    """
    _init_encoder()
    if _use_fallback or _encoder is None:
        # Fallback: ~3 chars per token (conservative for Vietnamese + English mix)
        return max(1, len(str(text)) // 3)
    try:
        return len(_encoder.encode(str(text)))
    except Exception:
        return max(1, len(str(text)) // 3)


def count_tokens_obj(obj) -> int:
    """Count tokens in any object by converting to string first."""
    return count_tokens(str(obj))


def is_using_tiktoken() -> bool:
    """Return True if tiktoken is active (not fallback)."""
    _init_encoder()
    return not _use_fallback
