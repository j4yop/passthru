"""Split a spoken text and a received text into matched utterance pairs.

Scoring a whole document as one blob tells you that something was lost. It does not tell
you *where*. This module cuts both sides into utterances and aligns them so each one can
be scored on its own, which is what turns "40% gone" into "the filename was in the third
sentence and it is gone".

Utterances come from the capture marker `end utterance` when it is present, otherwise from
sentence boundaries. Markers are authoritative: they are written at the moment of capture,
so they describe how the speech was actually delivered, which sentence splitting can only
guess at.

Two properties matter and are tested:

- One pair per utterance, always. An utterance that found no match yields an empty
  received side rather than being dropped, because a missing pair is the most important
  thing this module can report.
- Segmenting must not change the answer. If you add up survived tokens across every pair
  and divide by the total spoken, you get the same number as scoring the document whole.
  If it does not, the segmentation is distorting the measurement and every number
  downstream is suspect.

Pure: no file access, no network, standard library only.
"""

from __future__ import annotations

import difflib
import re
from typing import NamedTuple

from .score import DEFAULT_STRIP, Survival, score_stage, tokenize_spans

_MARKER = "end utterance"
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class Pair(NamedTuple):
    """One spoken utterance matched to whatever the received text kept of it."""

    index: int
    spoken: str
    received: str
    survival: Survival

    @property
    def matched(self) -> bool:
        """False when nothing of this utterance reached the received side."""
        return self.survival.spoken > 0 and self.survival.ratio > 0.0


def split_utterances(text: str, strip: tuple[str, ...] = DEFAULT_STRIP) -> list[str]:
    """Cut text into utterances on capture markers, else on sentence boundaries."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    for phrase in strip + (_MARKER,):
        if re.search(re.escape(phrase), cleaned, flags=re.IGNORECASE):
            parts = re.split(re.escape(phrase), cleaned, flags=re.IGNORECASE)
            return [p.strip() for p in parts if p.strip()]

    parts = [p.strip() for p in _SENTENCE_BOUNDARY.split(cleaned) if p.strip()]
    return parts or [cleaned]


def _strip_markers(text: str, strip: tuple[str, ...]) -> str:
    cleaned = text or ""
    for phrase in strip + (_MARKER,):
        cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def align_utterances(
    spoken: str,
    received: str,
    strip: tuple[str, ...] = DEFAULT_STRIP,
) -> list[Pair]:
    """Return one Pair per spoken utterance, in spoken order.

    Never raises on empty or one-sided input: no utterances means an empty list, and a
    received side that is missing entirely produces pairs with empty received text.
    """
    utterances = split_utterances(spoken, strip)
    if not utterances:
        return []

    spoken_clean = _strip_markers(spoken, strip)
    received_clean = _strip_markers(received, strip)

    # Character offset where each utterance begins in the stripped spoken text, so a
    # token can be attributed to exactly one utterance.
    spans: list[tuple[str, int, int, int]] = []  # token, start, end, utterance index
    cursor = 0
    for index, utterance in enumerate(utterances):
        found_at = spoken_clean.find(utterance, cursor)
        if found_at < 0:
            found_at = cursor
        base = found_at
        for token, start, end in tokenize_spans(utterance, strip):
            spans.append((token, base + start, base + end, index))
        cursor = base + len(utterance)

    received_tokens = tokenize_spans(received_clean, strip)
    received_texts = [t for t, _, _ in received_tokens]

    buckets: dict[int, list[tuple[int, int]]] = {}
    lost: dict[int, list[str]] = {}

    matcher = difflib.SequenceMatcher(
        None, [t for t, _, _, _ in spans], received_texts, autojunk=False
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                spoken_index = i1 + offset
                received_index = j1 + offset
                token, _, _, utterance_index = spans[spoken_index]
                buckets.setdefault(utterance_index, []).append(
                    (received_tokens[received_index][1], received_tokens[received_index][2])
                )
        elif tag == "replace":
            # Pair positionally inside the span and compare. A spoken token that has no
            # identical counterpart at its own position was overwritten, not moved.
            for offset, spoken_index in enumerate(range(i1, i2)):
                token, _, _, utterance_index = spans[spoken_index]
                received_index = j1 + offset
                if received_index < j2 and received_texts[received_index] == token:
                    buckets.setdefault(utterance_index, []).append(
                        (
                            received_tokens[received_index][1],
                            received_tokens[received_index][2],
                        )
                    )
                else:
                    lost.setdefault(utterance_index, []).append(token)
        elif tag == "delete":
            for spoken_index in range(i1, i2):
                token, _, _, utterance_index = spans[spoken_index]
                lost.setdefault(utterance_index, []).append(token)
        # "insert" is new text the speaker never said. Not a loss.

    pairs: list[Pair] = []
    for index, utterance in enumerate(utterances):
        chunks = sorted(buckets.get(index, []))
        if chunks:
            merged: list[str] = []
            # Start at the first matched token, not at offset zero, so each pair shows
            # only the text attributable to its own utterance rather than everything
            # before it as well.
            cursor_at = chunks[0][0]
            for start, end in chunks:
                if start > cursor_at:
                    merged.append(received_clean[cursor_at:start])
                merged.append(received_clean[start:end])
                cursor_at = end
            received_piece = " ".join("".join(merged).split())
        else:
            received_piece = ""

        survival = score_stage(utterance, received_piece, strip)
        pairs.append(
            Pair(
                index=index,
                spoken=utterance,
                received=received_piece,
                survival=survival,
            )
        )
    return pairs


def aggregate(pairs: list[Pair]) -> float:
    """Token-weighted survival across pairs.

    Weighting by token count rather than averaging per-pair ratios keeps a three-word
    utterance from counting as much as a thirty-word one, which is what makes this
    comparable to scoring the whole document at once.
    """
    spoken = sum(p.survival.spoken for p in pairs)
    if spoken == 0:
        return 0.0
    kept = sum(len(p.survival.survived) for p in pairs)
    return kept / spoken
