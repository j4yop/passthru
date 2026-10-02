"""Token survival across a single stage of a pipeline.

A *stage* is any boundary where text is rewritten: speech to transcript, transcript to
formatted prompt, prompt to a field in an editor. The question this module answers is
narrow and literal: of the tokens present before the stage, how many are still present
after it, and which ones.

Deliberate choices:

- Tokens are lowercased and split on non-word characters, except that `.` and `_` are
  kept so that `score.py`, `use_case`, and `400` survive as single tokens. A filename
  that gets split into `score` and `py` is no longer a filename, so keeping it intact is
  the difference between detecting a loss and missing it.
- Matching is exact after normalisation. Two runs will disagree about near misses such as
  `their` / `there`, which are perceptually free to a reader but are not the same token.
  That means this module **over**-counts loss relative to a perceptual measure. The
  over-count is the safe direction: it cannot hide a loss, only over-report one. Which
  specific near misses matter is a question for constraint extraction, not for counting.
- Insertions are not losses. A stage that adds words has not destroyed anything.

Pure: no file access, no network, no model calls, standard library only.
"""

from __future__ import annotations

import difflib
import re
from typing import NamedTuple

_TOKEN = re.compile(r"[A-Za-z0-9_.]+")
# Trailing `.` and `_` are sentence punctuation, not part of a name: `score.py.` is the
# filename `score.py` followed by a full stop, and treating them as different tokens
# reports a filename as lost when it survived. Leading dots are kept, because `.env`
# and `.gitignore` are named that way.
_TRAILING = "._"

# Utterance markers used to delimit captured dictation. They are scaffolding for the
# capture format, not content the speaker intended, so they never count as losses.
DEFAULT_STRIP = ("end utterance",)


class Survival(NamedTuple):
    """Result of scoring one stage."""

    ratio: float
    """Fraction of spoken tokens still present after the stage, 0.0 to 1.0."""

    spoken: int
    """How many tokens the stage started with."""

    survived: list[str]
    """Tokens present on both sides, in spoken order."""

    lost: list[str]
    """Tokens dropped or overwritten by the stage, in spoken order."""


def tokenize(text: str, strip: tuple[str, ...] = DEFAULT_STRIP) -> list[str]:
    """Lowercase word-ish tokens, with filenames kept whole.

    `strip` holds phrases to remove before tokenising, used to drop capture markers.
    """
    cleaned = text or ""
    for phrase in strip:
        cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
    return [t.strip(_TRAILING) for t in _TOKEN.findall(cleaned.lower()) if t.strip(_TRAILING)]


def score_stage(before: str, after: str, strip: tuple[str, ...] = DEFAULT_STRIP) -> Survival:
    """Score how much of `before` survives into `after`.

    Works on any pair of stages; nothing here knows what the text means.
    """
    src = tokenize(before, strip)
    dst = tokenize(after, strip)

    if not src:
        return Survival(ratio=1.0 if not dst else 0.0, spoken=0, survived=[], lost=[])

    matcher = difflib.SequenceMatcher(None, src, dst, autojunk=False)
    survived: list[str] = []
    lost: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        window = set(dst[j1:j2])
        if tag in ("equal", "replace"):
            for index in range(i1, i2):
                token = src[index]
                if token in window:
                    survived.append(token)
                else:
                    lost.append(token)
        elif tag == "delete":
            lost.extend(src[i1:i2])
        # "insert" adds tokens the speaker never said; that is not a loss.

    ratio = len(survived) / len(src)
    return Survival(ratio=ratio, spoken=len(src), survived=survived, lost=lost)
