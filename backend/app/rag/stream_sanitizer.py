"""Streaming text sanitizers for LLM output.

Foldex is citation-driven and should never expose model "thinking"/scratchpad text.
Some local models emit reasoning wrapped in tags like `<think>...</think>`.

This module provides a *stateful* stream filter that can safely be applied to
token-by-token streaming, including cases where tags are split across tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass
class _TagMatch:
    idx: int
    tag: str


class ThinkTagStreamFilter:
    """Remove `<think>...</think>` (and similar) from a streaming token sequence.

    Key properties:
    - Handles tags split across arbitrary token boundaries.
    - Drops *all* content inside the thinking region.
    - If a thinking tag appears early, we buffer a small "preface" window so we
      can discard any pre-tag meta-reasoning without leaking it to the UI.
    """

    _OPEN_TAGS: Tuple[str, ...] = ("<think>", "<antml:thinking>")
    _CLOSE_TAGS: Tuple[str, ...] = ("</think>", "</antml:thinking>")

    def __init__(self, *, preface_buffer_limit: int = 512) -> None:
        self._buf: str = ""
        self._in_think: bool = False
        self._preface_mode: bool = True
        self._preface_buffer_limit: int = max(0, int(preface_buffer_limit))

        # Keep enough tail to detect split tags.
        self._keep_tail: int = max(
            (len(t) - 1 for t in (self._OPEN_TAGS + self._CLOSE_TAGS)),
            default=0,
        )

    def process(self, token: Optional[str]) -> str:
        """Process a streaming token and return a safe-to-display chunk."""
        if not token:
            return ""

        self._buf += token

        # Preface buffering: if an opening think tag appears early, discard
        # everything before it (this is almost always meta-reasoning).
        if self._preface_mode:
            open_match = self._find_first(self._buf, self._OPEN_TAGS)
            if open_match is not None:
                # Discard any preface before the opening tag.
                self._buf = self._buf[open_match.idx :]
                self._preface_mode = False
            elif len(self._buf) >= self._preface_buffer_limit:
                # No opening tag seen in a reasonable window; stop buffering.
                self._preface_mode = False
            else:
                # Keep buffering until we can decide.
                return ""

        out: List[str] = []
        while self._buf:
            if self._in_think:
                close_match = self._find_first(self._buf, self._CLOSE_TAGS)
                if close_match is None:
                    # Don't leak any partial thinking content; keep tail so we
                    # can still detect a split closing tag on the next token.
                    if self._keep_tail and len(self._buf) > self._keep_tail:
                        self._buf = self._buf[-self._keep_tail :]
                    break

                # Drop everything through the closing tag.
                end = close_match.idx + len(close_match.tag)
                self._buf = self._buf[end:]
                self._in_think = False
                continue

            # Not currently in a think region.
            open_match = self._find_first(self._buf, self._OPEN_TAGS)
            close_match = self._find_first(self._buf, self._CLOSE_TAGS)

            # If we see a stray closing tag before any opening tag, drop it.
            if close_match is not None and (open_match is None or close_match.idx < open_match.idx):
                start = close_match.idx
                end = close_match.idx + len(close_match.tag)
                self._buf = self._buf[:start] + self._buf[end:]
                continue

            if open_match is None:
                # No opening tag found; emit safe portion, keep tail to catch split tags.
                if self._keep_tail and len(self._buf) > self._keep_tail:
                    out.append(self._buf[:-self._keep_tail])
                    self._buf = self._buf[-self._keep_tail :]
                else:
                    out.append(self._buf)
                    self._buf = ""
                break

            # Emit everything before the opening tag, then enter think mode.
            if open_match.idx > 0:
                out.append(self._buf[: open_match.idx])
            self._buf = self._buf[open_match.idx + len(open_match.tag) :]
            self._in_think = True

        return "".join(out)

    def flush(self) -> str:
        """Flush any remaining buffered content.
        
        This should be called when the stream ends to ensure no content
        is lost in the buffer.
        
        Returns:
            Any remaining safe content from the buffer
        """
        if not self._buf:
            return ""
        
        # If we're in preface mode, just return the buffer (it's safe)
        if self._preface_mode:
            result = self._buf
            self._buf = ""
            self._preface_mode = False
            return result
        
        # If we're in think mode, drop everything (it's thinking content)
        if self._in_think:
            self._buf = ""
            self._in_think = False
            return ""
        
        # Otherwise, return the buffer (it's safe content)
        result = self._buf
        self._buf = ""
        return result

    @staticmethod
    def _find_first(text: str, tags: Iterable[str]) -> Optional[_TagMatch]:
        low = text.lower()
        best: Optional[_TagMatch] = None
        for tag in tags:
            idx = low.find(tag.lower())
            if idx == -1:
                continue
            if best is None or idx < best.idx:
                best = _TagMatch(idx=idx, tag=tag)
        return best


